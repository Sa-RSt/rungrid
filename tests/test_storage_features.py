"""Additional tests for CSV, in-memory and bucket-file storage."""

import csv
import datetime
import io
import json
from pathlib import Path
from uuid import uuid4

import numpy
import pytest

from rungrid.experiment import StaleTrial, TrialResult, VarNamespace
from rungrid.plumbing import LoaderDumper
from rungrid.plumbing.storage import (
    BucketFileStorage,
    CSVSink,
    InMemoryStorage,
    _lock,
)


def make_trial(
    number: int = 1,
    *,
    tags=(),
    result=None,
    variables: dict | None = None,
) -> StaleTrial:
    """Create a StaleTrial for storage tests."""
    v = VarNamespace(set())
    if variables:
        v._variables_dict = dict(variables)
    trial = StaleTrial(number, uuid4(), v, tags=set(tags))
    if result is not None:
        trial._result = result
    return trial


def ok_trial(number=1, value=1.0, *, tags=()) -> StaleTrial:
    return make_trial(
        number,
        tags=tags,
        result=TrialResult.make_ok(value, datetime.datetime.now()),
    )


def make_storage(tmp_path: Path, **kwargs) -> BucketFileStorage:
    kwargs.setdefault("serializing_backend", "pickle")
    return BucketFileStorage(
        finished_dir=tmp_path / "finished",
        unfinished_dir=tmp_path / "unfinished",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# InMemoryStorage
# ---------------------------------------------------------------------------


def test_in_memory_storage_uses_given_list():
    trials = [make_trial(1)]
    storage = InMemoryStorage(trials)

    assert storage.trials is trials


def test_in_memory_storage_put_trial_and_put_trials():
    storage = InMemoryStorage()
    a = make_trial(1)
    b = make_trial(2)

    storage.put_trial(a)
    storage.put_trials([b])

    assert storage.trials == [a, b]
    assert repr(storage) == "InMemoryStorage(len=2)"


def test_in_memory_storage_get_trials_filters_finished_state():
    unfinished = make_trial(1)
    finished = ok_trial(2)
    storage = InMemoryStorage([unfinished, finished])

    assert list(storage.get_trials(finished=None)) == [unfinished, finished]
    assert list(storage.get_trials(finished=True)) == [finished]
    assert list(storage.get_trials(finished=False)) == [unfinished]


def test_in_memory_storage_get_trials_filters_tags():
    keep = make_trial(1, tags={"keep"})
    drop = make_trial(2, tags={"drop"})
    storage = InMemoryStorage([keep, drop])

    assert list(storage.get_trials(search_tag="keep")) == [keep]
    assert list(storage.get_trials(search_tag="missing")) == []


def test_in_memory_storage_consume_and_tag():
    trial = make_trial(1)
    storage = InMemoryStorage([trial])

    consumed = storage.consume_and_tag("processed")

    assert consumed is trial
    assert "processed" in trial.tags


def test_in_memory_storage_consume_and_tag_skips_already_tagged():
    first = make_trial(1, tags={"processed"})
    second = make_trial(2)
    storage = InMemoryStorage([first, second])

    consumed = storage.consume_and_tag("processed")

    assert consumed is second


def test_in_memory_storage_consume_and_tag_respects_filters():
    unfinished = make_trial(1)
    finished = ok_trial(2, tags={"tag"})
    storage = InMemoryStorage([unfinished, finished])

    assert storage.consume_and_tag("x", finished=True) is finished
    assert storage.consume_and_tag("other", search_tag="missing") is None


def test_in_memory_storage_consume_and_tag_respects_predicate():
    first = make_trial(1)
    second = make_trial(2)
    storage = InMemoryStorage([first, second])

    consumed = storage.consume_and_tag(
        "x", predicate=lambda trial: trial.optuna_trial_number == 2
    )

    assert consumed is second
    assert "x" not in first.tags


def test_in_memory_storage_consume_and_tag_returns_none_when_exhausted():
    storage = InMemoryStorage([make_trial(1, tags={"taken"})])

    assert storage.consume_and_tag("taken") is None


# ---------------------------------------------------------------------------
# CSVSink
# ---------------------------------------------------------------------------


def read_csv(stream: io.StringIO) -> tuple[list[str], list[list[str]]]:
    stream.seek(0)
    reader = csv.reader(stream)
    return next(reader), list(reader)


def test_csv_sink_header():
    stream = io.StringIO()
    CSVSink(stream)

    header, rows = read_csv(stream)

    assert header == [
        "uuid",
        "optuna_trial_number",
        "tags",
        "step_names",
        "step_start",
        "step_stop",
        "result",
        "error",
        "prune_reason",
        "vars",
        "trace",
    ]
    assert rows == []


def test_csv_sink_writes_multiple_trials():
    stream = io.StringIO()
    sink = CSVSink(stream)

    sink.put_trials([make_trial(1), make_trial(2)])

    header, rows = read_csv(stream)
    assert len(rows) == 2
    assert rows[0][1] == "1"
    assert rows[1][1] == "2"


def test_csv_sink_writes_result_error_and_prune_columns():
    now = datetime.datetime.now()
    ok = make_trial(1, result=TrialResult.make_ok("value", now))
    error = make_trial(2, result=TrialResult.make_error(ValueError("bad"), now))
    pruned = make_trial(3, result=TrialResult.make_pruned("no good", now))
    no_result = make_trial(4)

    stream = io.StringIO()
    CSVSink(stream).put_trials([ok, error, pruned, no_result])
    header, rows = read_csv(stream)

    result_col = header.index("result")
    error_col = header.index("error")
    prune_col = header.index("prune_reason")
    assert rows[0][result_col] == repr("value")
    assert rows[1][error_col] == repr(ValueError("bad"))
    assert rows[2][prune_col] == repr("no good")
    assert rows[3][result_col] == ""
    assert rows[3][error_col] == ""
    assert rows[3][prune_col] == ""


def test_csv_sink_writes_variables_as_json():
    trial = make_trial(1, variables={"x": 1, "y": "a"})
    stream = io.StringIO()
    CSVSink(stream).put_trial(trial)
    header, rows = read_csv(stream)

    assert json.loads(rows[0][header.index("vars")]) == {"x": 1, "y": "a"}


def test_csv_sink_writes_traces_with_jsonized_values():
    trial = make_trial(1)
    trial.append_trace("loss", numpy.float32(0.5), epoch=numpy.int64(1))
    trial.append_trace("bytes", b"abc")

    stream = io.StringIO()
    CSVSink(stream).put_trial(trial)
    header, rows = read_csv(stream)

    trace = json.loads(rows[0][header.index("trace")])
    assert trace["loss"] == [[0.5, {"epoch": 1}]]
    assert trace["bytes"] == [["abc", {}]]


def test_csv_sink_flushes_stream():
    class FlushCountingStream(io.StringIO):
        def __init__(self):
            super().__init__()
            self.flush_count = 0

        def flush(self):
            self.flush_count += 1
            return super().flush()

    stream = FlushCountingStream()
    sink = CSVSink(stream)
    before = stream.flush_count

    sink.put_trial(make_trial(1))

    assert stream.flush_count > before


def test_csv_jsonize_scalars_and_containers():
    sink = CSVSink(io.StringIO())

    assert sink._jsonize(3) == 3
    assert sink._jsonize(True) is True
    assert sink._jsonize(False) is False
    assert sink._jsonize(3.5) == 3.5
    assert sink._jsonize("abc") == "abc"
    assert sink._jsonize(b"abc") == "abc"
    assert sink._jsonize(None) is None
    assert sink._jsonize([1, 2]) == [1, 2]
    assert sink._jsonize((1, 2)) == [1, 2]
    assert sink._jsonize({1, 2}) in ([1, 2], [2, 1])
    assert sink._jsonize({1: "a"}) == {"1": "a"}
    assert sink._jsonize(object()).startswith("<object object at")


def test_csv_jsonize_numpy_values():
    sink = CSVSink(io.StringIO())

    assert sink._jsonize(numpy.int64(5)) == 5
    assert isinstance(sink._jsonize(numpy.float32(1.25)), float)


def test_csv_format_trace():
    trial = make_trial(1)
    trial.append_trace("loss", 1, epoch=2)

    formatted = CSVSink._format_trace(trial, "loss")

    assert formatted == repr([(1, {"epoch": 2})])


# ---------------------------------------------------------------------------
# BucketFileStorage basics
# ---------------------------------------------------------------------------


def test_bucket_storage_creates_directories_and_repr(tmp_path):
    storage = make_storage(tmp_path)

    assert (tmp_path / "finished").is_dir()
    assert (tmp_path / "unfinished").is_dir()
    assert "BucketFileStorage" in repr(storage)
    assert "finished" in repr(storage)


def test_bucket_storage_accepts_loader_dumper_instance(tmp_path):
    ld = LoaderDumper.make_pickle((), {}, (), {})
    storage = make_storage(tmp_path, serializing_backend=ld)
    trial = make_trial(1)

    storage.put_trial(trial)

    assert [t.uuid for t in storage.get_trials()] == [trial.uuid]


@pytest.mark.parametrize("backend", ["pickle", "joblib"])
def test_bucket_storage_roundtrips_backends(tmp_path, backend):
    storage = make_storage(tmp_path, serializing_backend=backend)
    trial = make_trial(1, tags={"tag"})

    storage.put_trial(trial)

    assert [t.uuid for t in storage.get_trials()] == [trial.uuid]


def test_bucket_tag_mangle_strips_unsafe_characters(tmp_path):
    storage = make_storage(tmp_path)

    assert storage._tag_mangle("a/b c.d-e") == "abc.d-e"


def test_bucket_tag_mangle_warns_for_long_tags(tmp_path):
    storage = make_storage(tmp_path)
    long_tag = "a" * 130

    with pytest.warns(RuntimeWarning, match="mangle to a large path component"):
        storage._tag_mangle(long_tag)


def test_bucket_path_uses_sorted_tags_and_uuid_prefix(tmp_path):
    storage = make_storage(tmp_path, uuid_prefix_length=4)
    trial = make_trial(1, tags={"b", "a"})

    bucket = storage._bucket_path(trial)

    assert bucket.parent.name == "a.b"
    assert bucket.name == str(trial.uuid).replace("-", "")[:4] + ".bin"
    assert bucket.parent.parent == tmp_path / "unfinished"


def test_bucket_path_untagged(tmp_path):
    storage = make_storage(tmp_path)
    trial = make_trial(1)

    assert storage._bucket_path(trial).parent.name == "untagged"


def test_bucket_path_for_finished_trial(tmp_path):
    storage = make_storage(tmp_path)
    trial = ok_trial(1, tags={"done"})

    assert storage._bucket_path(trial).parent.parent == tmp_path / "finished"
    assert (
        storage._bucket_path(trial, assume_unfinished=True).parent.parent
        == tmp_path / "unfinished"
    )


def test_bucket_storage_replaces_existing_trial_with_same_uuid(tmp_path):
    storage = make_storage(tmp_path)
    trial = make_trial(1)
    storage.put_trial(trial)
    storage.put_trial(trial)

    assert len(list(storage.get_trials())) == 1


def test_bucket_storage_get_trials_filters_finished_and_tag(tmp_path):
    storage = make_storage(tmp_path)
    unfinished = make_trial(1, tags={"keep"})
    finished = ok_trial(2, tags={"keep"})
    storage.put_trial(unfinished)
    storage.put_trial(finished)

    assert {t.uuid for t in storage.get_trials()} == {unfinished.uuid, finished.uuid}
    assert [t.uuid for t in storage.get_trials(finished=True)] == [finished.uuid]
    assert [t.uuid for t in storage.get_trials(finished=False)] == [unfinished.uuid]
    assert {t.uuid for t in storage.get_trials(search_tag="keep")} == {
        unfinished.uuid,
        finished.uuid,
    }
    assert list(storage.get_trials(search_tag="missing")) == []


def test_bucket_storage_tag_search_is_exact(tmp_path):
    storage = make_storage(tmp_path)
    trial_a = make_trial(1, tags={"a"})
    trial_aa = make_trial(2, tags={"aa"})
    storage.put_trial(trial_a)
    storage.put_trial(trial_aa)

    results = list(storage.get_trials(search_tag="a"))

    assert [t.uuid for t in results] == [trial_a.uuid]


def test_bucket_storage_moves_finished_trial_from_unfinished(tmp_path):
    storage = make_storage(tmp_path)
    trial = make_trial(1)
    storage.put_trial(trial)

    finished = trial.with_result(
        TrialResult.make_ok("done", datetime.datetime.now())
    )
    storage.put_trial(finished)

    assert [t.uuid for t in storage.get_trials(finished=True)] == [trial.uuid]
    assert list(storage.get_trials(finished=False)) == []


def test_bucket_storage_consume_and_tag(tmp_path):
    storage = make_storage(tmp_path)
    trial = make_trial(1, tags={"untagged"})
    storage.put_trial(trial)

    consumed = storage.consume_and_tag("processed", search_tag="untagged")

    assert consumed is not None
    assert consumed.uuid == trial.uuid
    assert "processed" in consumed.tags
    stored = list(storage.get_trials(search_tag="processed"))
    assert len(stored) == 1
    assert "untagged" in stored[0].tags


def test_bucket_storage_consume_and_tag_respects_predicate(tmp_path):
    storage = make_storage(tmp_path)
    first = make_trial(1)
    second = make_trial(2)
    storage.put_trial(first)
    storage.put_trial(second)

    consumed = storage.consume_and_tag(
        "picked", predicate=lambda trial: trial.optuna_trial_number == 2
    )

    assert consumed is not None
    assert consumed.optuna_trial_number == 2


def test_bucket_storage_consume_and_tag_skips_already_tagged(tmp_path):
    storage = make_storage(tmp_path)
    trial = make_trial(1, tags={"processed"})
    storage.put_trial(trial)

    assert storage.consume_and_tag("processed") is None


def test_bucket_storage_consume_and_tag_returns_none_when_empty(tmp_path):
    storage = make_storage(tmp_path)

    assert storage.consume_and_tag("processed") is None


def test_bucket_storage_consume_and_tag_respects_finished_filter(tmp_path):
    storage = make_storage(tmp_path)
    unfinished = make_trial(1)
    finished = ok_trial(2)
    storage.put_trial(unfinished)
    storage.put_trial(finished)

    consumed = storage.consume_and_tag("picked", finished=True)

    assert consumed is not None
    assert consumed.uuid == finished.uuid


def test_bucket_storage_ignores_corrupted_bucket(tmp_path, capfd):
    storage = make_storage(tmp_path)
    trial = make_trial(1)
    storage.put_trial(trial)
    bucket = storage._bucket_path(trial)
    bucket.write_bytes(b"this is not a pickle")

    assert list(storage.get_trials()) == []
    captured = capfd.readouterr()
    assert "ignoring corrupted bucket" in captured.err


def test_bucket_storage_read_missing_bucket_returns_empty(tmp_path):
    storage = make_storage(tmp_path)

    assert storage._read_bucket(tmp_path / "missing.bin") == []


def test_bucket_storage_write_empty_bucket_unlinks_file(tmp_path):
    storage = make_storage(tmp_path)
    trial = make_trial(1)
    storage.put_trial(trial)
    bucket = storage._bucket_path(trial)
    assert bucket.is_file()

    storage._write_bucket(bucket, [])

    assert not bucket.exists()


def test_bucket_storage_bucket_iter_only_yields_bin_files(tmp_path):
    binary = tmp_path / "a.bin"
    text = tmp_path / "b.txt"
    binary.write_bytes(b"")
    text.write_text("")
    storage = make_storage(tmp_path)

    assert list(storage._bucket_iter(tmp_path)) == [binary]


def test_bucket_storage_custom_lock_factory(tmp_path):
    events = []

    class RecordingLock:
        def __init__(self, path):
            self.path = path

        def acquire(self):
            events.append(("acquire", self.path))
            return True

        def release(self):
            events.append(("release", self.path))

    def lock_factory(path):
        events.append(("create", path))
        return RecordingLock(path)

    storage = make_storage(tmp_path, lock_factory=lock_factory)
    storage.put_trial(make_trial(1))
    list(storage.get_trials())

    assert any(event[0] == "create" for event in events)
    assert any(event[0] == "acquire" for event in events)
    assert any(event[0] == "release" for event in events)


def test_lock_retries_with_exponential_backoff(monkeypatch):
    import rungrid.plumbing.storage as storage_module

    attempts = []
    releases = []
    sleeps = []

    class FlakyLock:
        def acquire(self):
            attempts.append(1)
            return len(attempts) >= 3

        def release(self):
            releases.append(1)

    monkeypatch.setattr(storage_module.time, "sleep", lambda delay: sleeps.append(delay))

    with _lock(FlakyLock()):
        pass

    assert len(attempts) == 3
    assert sleeps == [1e-3, 2e-3]
    assert releases == [1]


def test_lock_releases_when_body_raises():
    releases = []

    class Lock:
        def acquire(self):
            return True

        def release(self):
            releases.append(1)

    with pytest.raises(RuntimeError, match="inside lock"):
        with _lock(Lock()):
            raise RuntimeError("inside lock")

    assert releases == [1]


def test_bucket_storage_prefetch_empty_iterable(tmp_path):
    storage = make_storage(tmp_path)

    assert list(storage._prefetching_bucket_iter([])) == []


def test_bucket_storage_prefetch_empty_directory(tmp_path):
    storage = make_storage(tmp_path)

    assert list(storage._prefetching_bucket_iter(tmp_path)) == []
