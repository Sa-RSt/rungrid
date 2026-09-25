"""Storage systems and export sinks for recording experimental trials on disk."""

import csv
import os
import re
import time
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import IO, Literal, Protocol, Sequence

from optuna.storages.journal import JournalFileOpenLock

from rungrid.experiment import StaleTrial, Trial
from rungrid.plumbing import LoaderDumper, PredicateType, Sink, Source
from rungrid.warn import rungrid_warn

_STRIP_PATH_CHARS = re.compile(r"[^a-zA-Z0-9_.-]")


LOCK_FILE_SUFFIX = ".lock"
RENAME_FILE_SUFFIX = ".rename"


class SupportsAcquireRelease(Protocol):
    """Protocol defining a locking mechanism with acquire and release capabilities."""

    def acquire(self) -> bool:
        """Acquire the lock.

        :return: True if the lock was successfully acquired, False otherwise.
        :rtype: bool
        """
        return NotImplemented

    def release(self) -> None:
        """Release the lock."""
        pass


@contextmanager
def _lock(sar: SupportsAcquireRelease):
    wait = 1e-3
    while not sar.acquire():
        time.sleep(wait)
        wait *= 2
    yield
    sar.release()


class BucketFileStorage(Sink, Source):
    """A trial storage system that buckets trials into files on the filesystem.

    Trials are organized by their tag values into distinct bucket files, separating
    ongoing (unfinished) and completed (finished) trials.
    """

    def __init__(
        self,
        finished_dir: os.PathLike,
        unfinished_dir: os.PathLike,
        serializing_backend: Literal["torch", "pickle", "joblib", "default"]
        | LoaderDumper = "default",
        lock_factory: Callable[[str], "SupportsAcquireRelease"] = JournalFileOpenLock,
    ) -> None:
        """Initialize the BucketFileStorage.

        :param finished_dir: Directory path where completed trials are stored.
        :type finished_dir: os.PathLike
        :param unfinished_dir: Directory path where unfinished trials are stored.
        :type unfinished_dir: os.PathLike
        :param serializing_backend: The serialization backend to use.
        :type serializing_backend: Literal["torch", "pickle", "joblib", "default"] | LoaderDumper
        :param lock_factory: A callable that provides file locking primitives.
        :type lock_factory: collections.abc.Callable[[str], SupportsAcquireRelease]
        :raises TypeError: If serializing_backend is not a string or LoaderDumper.
        """
        super().__init__()
        self._finished_dir = Path(finished_dir)
        self._unfinished_dir = Path(unfinished_dir)
        self._finished_dir.mkdir(parents=True, exist_ok=True)
        self._unfinished_dir.mkdir(parents=True, exist_ok=True)
        self._lock_factory = lock_factory
        self._io_workers = ThreadPoolExecutor(os.cpu_count())
        if isinstance(serializing_backend, str):
            self._ld = LoaderDumper.make(serializing_backend, (), {}, (), {})
        elif isinstance(serializing_backend, LoaderDumper):
            self._ld = serializing_backend
        else:
            raise TypeError(
                f"serializing_backend: expected type str or LoaderDumper, got {type(serializing_backend)!r}"
            )

    @lru_cache
    def _tag_mangle(self, concat: str) -> str:
        mangle = _STRIP_PATH_CHARS.sub("", concat)
        if len(mangle) >= 120:
            rungrid_warn(
                f"the tags {concat} mangle to a large path component {mangle}"
                + "which may not be supported by the filesystem, try using shorter tag names",
                RuntimeWarning,
            )
        return mangle

    def _base_dir(self, trial: Trial) -> Path:
        if trial.is_finished():
            return self._finished_dir
        else:
            return self._unfinished_dir

    def _bucket_path(self, trial: Trial, *, assume_unfinished: bool = False) -> Path:
        if assume_unfinished:
            base_dir = self._unfinished_dir
        else:
            base_dir = self._base_dir(trial)
        concat = ".".join(sorted(trial.tags)) or "untagged"
        mangle = self._tag_mangle(concat)
        dir_name = base_dir / mangle
        return (dir_name / str(trial.uuid)[:4]).with_suffix(".bin")

    def _bucket_lock(self, bucket: Path):
        bucket.parent.mkdir(parents=True, exist_ok=True)
        return _lock(self._lock_factory(str(bucket.resolve())))

    def _read_bucket(self, bucket: Path) -> list[StaleTrial]:
        with self._bucket_lock(bucket):
            if not bucket.is_file():
                return []
            with open(bucket, "r+b") as f:
                return self._ld.load(f)

    def _prefetch_bucket(self, bucket: Path) -> Future[list[StaleTrial]]:
        return self._io_workers.submit(self._read_bucket, bucket)

    def _prefetching_bucket_iter(
        self, buckets: Iterable[Path]
    ) -> Iterable[list[StaleTrial]]:
        iterator = iter(buckets)
        try:
            first_bucket = next(iterator)
        except StopIteration:
            return
        future = self._prefetch_bucket(first_bucket)
        try:
            while 1:
                bucket = next(iterator)
                if future.done():
                    res = future.result(timeout=0)
                    future = self._prefetch_bucket(bucket)
                    yield res
                else:
                    yield self._read_bucket(bucket)
        except StopIteration:
            pass
        yield future.result(timeout=10)

    def _write_bucket(self, bucket: Path, trials: list[StaleTrial]) -> None:
        with self._bucket_lock(bucket):
            if trials:
                with open(bucket, "wb") as f:
                    self._ld.dump(trials, f)
            else:
                bucket.unlink()

    def _get_trials_from_base(self, base: Path) -> Iterable[StaleTrial]:
        for mangle in base.iterdir():
            for read_bucket in self._prefetching_bucket_iter(mangle.iterdir()):
                yield from read_bucket

    def _get_trials_with_tag(self, tag: str, base: Path) -> Iterable[StaleTrial]:
        mangled_tag = self._tag_mangle(tag)
        for mangle in base.iterdir():
            if mangled_tag in mangle.name:
                for read_bucket in self._prefetching_bucket_iter(mangle.iterdir()):
                    for trial in read_bucket:
                        if tag in trial.tags:
                            yield trial

    def get_trials(
        self, *, finished: bool | None = None, search_tag: str | None = None
    ) -> Iterable[StaleTrial]:
        """Retrieve stored trials matching specified criteria from the filesystem.

        :param finished: Filter by finished state (True for finished, False for unfinished, None for both).
        :type finished: bool | None
        :param search_tag: Optional tag string filter.
        :type search_tag: str | None
        :return: An iterable of stored trials.
        :rtype: collections.abc.Iterable[StaleTrial]
        """
        if finished is None:
            bases = [self._finished_dir, self._unfinished_dir]
        elif finished:
            bases = [self._finished_dir]
        else:
            bases = [self._unfinished_dir]
        if search_tag is None:
            for base in bases:
                yield from self._get_trials_from_base(base)
        else:
            for base in bases:
                yield from self._get_trials_with_tag(search_tag, base)

    def put_trial(self, trial: Trial) -> None:
        """Save a trial to storage, updating its bucket on the filesystem.

        If the trial has finished, it is moved from the unfinished directory
        to the finished directory.

        :param trial: The trial instance to save.
        :type trial: Trial
        """
        bucket = self._bucket_path(trial)
        bucket.parent.mkdir(parents=True, exist_ok=True)
        trials = self._read_bucket(bucket)
        trials_new = [x for x in trials if x.uuid != trial.uuid]
        trials_new.append(trial.as_stale())
        self._write_bucket(bucket, trials_new)

        if trial.is_finished():
            unfinished_bucket = self._bucket_path(trial, assume_unfinished=True)
            trials = self._read_bucket(unfinished_bucket)
            trials_new = [x for x in trials if x.uuid != trial.uuid]
            if len(trials) != len(trials_new):
                self._write_bucket(unfinished_bucket, trials_new)

    def _consume_lock(self):
        lockfile = self._unfinished_dir / "(consume)"
        return _lock(self._lock_factory(str(lockfile.resolve())))

    def consume_and_tag(
        self,
        applied_tag: str,
        *,
        finished: bool | None = None,
        search_tag: str | None = None,
        predicate: PredicateType = lambda _: True,
    ) -> StaleTrial | None:
        """Atomically find, tag, and save a trial matching the criteria.

        :param applied_tag: The tag to apply to the found trial.
        :type applied_tag: str
        :param finished: Filter by finished state.
        :type finished: bool | None
        :param search_tag: Optional tag string filter to apply before selection.
        :type search_tag: str | None
        :return: The tagged trial, or None if no match was found.
        :rtype: StaleTrial | None
        """
        sampled: StaleTrial | None = None
        with self._consume_lock():
            for trial in self.get_trials(finished=finished, search_tag=search_tag):
                if applied_tag not in trial.tags and predicate(trial):
                    sampled = trial
                    break
            if sampled is not None:
                old_bucket = self._bucket_path(sampled)
                old_trials = self._read_bucket(old_bucket)
                old_trials_new = [x for x in old_trials if x.uuid != sampled.uuid]
                if len(old_trials) != len(old_trials_new):
                    self._write_bucket(old_bucket, old_trials_new)

                sampled.add_tag(applied_tag)
                self.put_trial(sampled)
        return sampled


class CSVSink(Sink):
    """A trial Sink that writes trial summaries to a CSV file or stream."""

    def __init__(self, fp: IO) -> None:
        """Initialize the CSVSink with a file-like write stream.

        :param fp: A writeable file-like object to write CSV rows to.
        :type fp: typing.IO
        """
        super().__init__()
        self._fp = fp
        self._csv = csv.writer(fp)
        self._csv.writerow(
            [
                "uuid",
                "optuna_trial_number",
                "tags",
                "step_names",
                "step_start",
                "step_stop",
                "result",
                "error",
                "prune_reason",
            ]
        )

    def put_trials(self, trials: Sequence[Trial]) -> None:
        """Write a sequence of trials to the CSV stream.

        :param trials: The sequence of Trial instances to write.
        :type trials: collections.abc.Sequence[Trial]
        """
        stales = (x.as_stale() for x in trials)
        self._csv.writerows(
            [
                [
                    s.uuid,
                    s.optuna_trial_number,
                    ";".join(s.tags),
                    ";".join(r.step.name for r in s.step_records),
                    ";".join(r.start.isoformat() for r in s.step_records),
                    ";".join(r.end.isoformat() for r in s.step_records),
                    repr(s.result.result)
                    if s.result is not None and s.result.result is not None
                    else "",
                    repr(s.result.error)
                    if s.result is not None and s.result.error is not None
                    else "",
                    repr(s.result.prune_reason)
                    if s.result is not None and s.result.prune_reason is not None
                    else "",
                ]
                for s in stales
            ]
        )
        self._fp.flush()

    def put_trial(self, trial: Trial) -> None:
        """Write a single trial summary to the CSV stream.

        :param trial: The trial instance to write.
        :type trial: Trial
        """
        self.put_trials([trial])


class InMemoryStorage(Source, Sink):
    def __init__(self, trials: list[Trial] | None = None) -> None:
        super().__init__()
        if trials is None:
            trials = []
        self.trials = trials
        self._lock = Lock()

    def get_trials(
        self, *, finished: bool | None = None, search_tag: str | None = None
    ) -> Iterable[Trial]:
        for trial in list(self.trials):
            if finished is not None and finished != trial.is_finished():
                continue
            if search_tag is not None and search_tag not in trial.tags:
                continue
            yield trial

    def consume_and_tag(
        self,
        applied_tag: str,
        *,
        finished: bool | None = None,
        search_tag: str | None = None,
        predicate: PredicateType = lambda _: True,
    ) -> Trial | None:
        with self._lock:
            for trial in self.trials:
                if finished is not None and finished != trial.is_finished():
                    continue
                if search_tag is not None and search_tag not in trial.tags:
                    continue
                if applied_tag in trial.tags:
                    continue
                if not predicate(trial):
                    continue
                trial.add_tag(applied_tag)
                return trial

    def put_trial(self, trial: Trial) -> None:
        self.trials.append(trial)

    def put_trials(self, trials: Sequence[Trial]) -> None:
        self.trials.extend(trials)
