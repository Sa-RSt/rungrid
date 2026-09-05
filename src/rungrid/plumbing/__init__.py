import importlib.util
import pickle
import random
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from io import BytesIO
from typing import IO, Any

import joblib

from rungrid.experiment import StaleTrial, Trial

PredicateType = Callable[[Trial], bool]


class LoaderDumper:
    """Common wrapper for serializing and deserializing arbitrary objects.

    Encapsulates different serialization backends (e.g., pickle, torch, joblib)
    under a uniform interface.
    """

    def __init__(
        self,
        desc: str,
        loader: Callable[[IO], Any],
        dumper: Callable[[Any, IO], None],
        loader_args,
        loader_kwargs,
        dumper_args,
        dumper_kwargs,
    ) -> None:
        """Create a serializer/deserializer wrapper.

        :param desc: A description for the serializer (used for debugging/repr).
        :type desc: str
        :param loader: A callable accepting a readable file-like object that returns a deserialized object.
        :type loader: collections.abc.Callable[[typing.IO], typing.Any]
        :param dumper: A callable accepting an object and a writable file-like object to serialize into it.
        :type dumper: collections.abc.Callable[[typing.Any, typing.IO], None]
        :param loader_args: Extra positional arguments to pass to the loader.
        :type loader_args: tuple
        :param loader_kwargs: Extra keyword arguments to pass to the loader.
        :type loader_kwargs: dict
        :param dumper_args: Extra positional arguments to pass to the dumper.
        :type dumper_args: tuple
        :param dumper_kwargs: Extra keyword arguments to pass to the dumper.
        :type dumper_kwargs: dict
        """
        self._desc = desc
        self._loader = loader
        self._dumper = dumper
        self._loader_args = loader_args
        self._loader_kwargs = loader_kwargs
        self._dumper_args = dumper_args
        self._dumper_kwargs = dumper_kwargs

    def loads(self, data: bytes) -> Any:
        """Deserialize an object from bytes.

        :param data: The serialized byte data.
        :type data: bytes
        :return: The deserialized object.
        :rtype: typing.Any
        """
        return self.load(BytesIO(data))

    def dumps(self, obj: Any) -> bytes:
        """Serialize an object to bytes.

        :param obj: The object to serialize.
        :type obj: typing.Any
        :return: The serialized byte string.
        :rtype: bytes
        """
        buf = BytesIO()
        self.dump(obj, buf)
        return buf.getvalue()

    def load(self, stream: IO) -> Any:
        """Deserialize an object from a stream.

        :param stream: A readable buffered file-like object.
        :type stream: typing.IO
        :return: The deserialized object.
        :rtype: typing.Any
        """
        return self._loader(stream, *self._loader_args, **self._loader_kwargs)

    def dump(self, obj: Any, stream: IO) -> None:
        """Serialize an object to a stream.

        :param obj: The object to serialize.
        :type obj: typing.Any
        :param stream: A writable buffered file-like object.
        :type stream: typing.IO
        """
        self._dumper(obj, stream, *self._dumper_args, **self._dumper_kwargs)

    def __repr__(self) -> str:
        return f"LoaderDumper({self._desc})"

    @classmethod
    def make(
        cls,
        backend_name: str,
        loader_args,
        loader_kwargs,
        dumper_args,
        dumper_kwargs,
    ) -> "LoaderDumper":
        """Create and return a built-in LoaderDumper wrapper.

        :param backend_name: One of "default", "torch", "joblib" or "pickle".
        :type backend_name: str
        :param loader_args: Position arguments for the loader callable.
        :type loader_args: tuple
        :param loader_kwargs: Keyword arguments for the loader callable.
        :type loader_kwargs: dict
        :param dumper_args: Position arguments for the dumper callable.
        :type dumper_args: tuple
        :param dumper_kwargs: Keyword arguments for the dumper callable.
        :type dumper_kwargs: dict
        :return: A built-in configured LoaderDumper wrapper instance.
        :rtype: LoaderDumper
        """
        implementations = {
            "default": cls.make_default,
            "torch": cls.make_torch,
            "joblib": cls.make_joblib,
            "pickle": cls.make_pickle,
        }
        return implementations[backend_name](
            loader_args, loader_kwargs, dumper_args, dumper_kwargs
        )

    @classmethod
    def make_default(
        cls, loader_args, loader_kwargs, dumper_args, dumper_kwargs
    ) -> "LoaderDumper":
        """Create a default built-in LoaderDumper.

        Currently prefers PyTorch if installed, otherwise falls back to joblib.
        However, this behavior may change in future versions.

        :param loader_args: Position arguments for the loader callable.
        :param loader_kwargs: Keyword arguments for the loader callable.
        :param dumper_args: Position arguments for the dumper callable.
        :param dumper_kwargs: Keyword arguments for the dumper callable.
        :return: A default configured LoaderDumper wrapper instance.
        :rtype: LoaderDumper
        """
        if importlib.util.find_spec("torch") is not None:
            return cls.make_torch(
                loader_args, loader_kwargs, dumper_args, dumper_kwargs
            )
        return cls.make_joblib(loader_args, loader_kwargs, dumper_args, dumper_kwargs)

    @classmethod
    def make_torch(
        cls, loader_args, loader_kwargs, dumper_args, dumper_kwargs
    ) -> "LoaderDumper":
        """Create a PyTorch-backed LoaderDumper.

        :param loader_args: Position arguments for torch.load.
        :param loader_kwargs: Keyword arguments for torch.load.
        :param dumper_args: Position arguments for torch.save.
        :param dumper_kwargs: Keyword arguments for torch.save.
        :return: A PyTorch-configured LoaderDumper wrapper instance.
        :rtype: LoaderDumper
        """
        import torch

        return cls(
            "torch-default",
            torch.load,
            torch.save,
            loader_args,
            loader_kwargs,
            dumper_args,
            dumper_kwargs,
        )

    @classmethod
    def make_pickle(
        cls, loader_args, loader_kwargs, dumper_args, dumper_kwargs
    ) -> "LoaderDumper":
        """Create a pickle-backed LoaderDumper.

        :param loader_args: Position arguments for pickle.load.
        :param loader_kwargs: Keyword arguments for pickle.load.
        :param dumper_args: Position arguments for pickle.dump.
        :param dumper_kwargs: Keyword arguments for pickle.dump.
        :return: A pickle-configured LoaderDumper wrapper instance.
        :rtype: LoaderDumper
        """
        return cls(
            "pickle-default",
            pickle.load,
            pickle.dump,
            loader_args,
            loader_kwargs,
            dumper_args,
            dumper_kwargs,
        )

    @classmethod
    def make_joblib(
        cls, loader_args, loader_kwargs, dumper_args, dumper_kwargs
    ) -> "LoaderDumper":
        """Create a joblib-backed LoaderDumper.

        :param loader_args: Position arguments for joblib.load.
        :param loader_kwargs: Keyword arguments for joblib.load.
        :param dumper_args: Position arguments for joblib.dump.
        :param dumper_kwargs: Keyword arguments for joblib.dump.
        :return: A joblib-configured LoaderDumper wrapper instance.
        :rtype: LoaderDumper
        """
        if "mmap_mode" not in loader_kwargs:
            loader_kwargs["mmap_mode"] = "c"
        return cls(
            "joblib-default",
            joblib.load,
            joblib.dump,  # type: ignore
            loader_args,
            loader_kwargs,
            dumper_args,
            dumper_kwargs,
        )


class Sink(ABC):
    """Abstract base class representing a write-only sink for experimental trials."""

    def with_sink_filter(self, predicate: PredicateType) -> "FilterSink":
        return FilterSink(self, predicate)

    @abstractmethod
    def put_trial(self, trial: Trial) -> None:
        """Store or record a trial.

        :param trial: The trial instance to store.
        :type trial: Trial
        """
        pass


class Source(ABC):
    """Abstract base class representing a read-only source for experimental trials."""

    def with_source_filter(self, predicate: PredicateType) -> "FilterSource":
        return FilterSource(self, predicate)

    @abstractmethod
    def get_trials(
        self, *, finished: bool | None = None, search_tag: str | None = None
    ) -> Iterable[Trial]:
        """Retrieve trials matching the specified criteria.

        :param finished: Filter by finished state (True for completed, False for ongoing, None for all).
        :type finished: bool | None
        :param search_tag: Optional tag string to filter trials.
        :type search_tag: str | None
        :return: An iterable of stored trials.
        :rtype: collections.abc.Iterable[StaleTrial]
        """
        return NotImplemented

    @abstractmethod
    def consume_and_tag(
        self,
        applied_tag: str,
        *,
        finished: bool | None = None,
        search_tag: str | None = None,
        predicate: PredicateType = lambda _: True,
    ) -> Trial | None:
        """Atomically find, tag, and return a trial from this source.

        :param applied_tag: The tag to apply to the trial.
        :type applied_tag: str
        :param finished: Filter by finished state (True for completed, False for ongoing, None for all).
        :type finished: bool | None
        :param search_tag: Optional tag string to filter the trials list before tag selection.
        :type search_tag: str | None
        :return: The selected and tagged StaleTrial, or None if no match is found.
        :rtype: StaleTrial | None
        """
        return NotImplemented


class Empty(Sink, Source):
    """An empty sink and source that stores nothing and yields no trials."""

    def put_trial(self, trial: Trial) -> None:
        """Store or record a trial (does nothing).

        :param trial: The trial instance to store.
        :type trial: Trial
        """
        pass

    def get_trials(
        self, *, finished: bool | None = None, search_tag: str | None = None
    ) -> Iterable[StaleTrial]:
        """Retrieve trials matching specified criteria (always returns empty list).

        :param finished: Filter by finished state.
        :type finished: bool | None
        :param search_tag: Optional tag string filter.
        :type search_tag: str | None
        :return: An empty list.
        :rtype: list
        """
        return []

    def consume_and_tag(
        self,
        applied_tag: str,
        *,
        finished: bool | None = None,
        search_tag: str | None = None,
        predicate: PredicateType = lambda _: True,
    ) -> StaleTrial | None:
        """Find, tag, and return a trial (always returns None).

        :param applied_tag: The tag to apply.
        :type applied_tag: str
        :param finished: Filter by finished state.
        :type finished: bool | None
        :param search_tag: Optional tag string filter.
        :type search_tag: str | None
        :return: None.
        :rtype: None
        """
        return None


class MultiSource(Source):
    """A combined trial source delegating to multiple component sources."""

    def __init__(
        self,
        components: Iterable[Source],
        rng: random.Random | None = None,
    ) -> None:
        """Initialize the MultiSource.

        :param components: An iterable of source components to merge.
        :type components: collections.abc.Iterable[Source]
        :param rng: Optional random number generator for random delegation selection.
        :type rng: random.Random | None
        """
        super().__init__()
        self._sources = list(components)
        if rng is None:
            rng = random.Random()
        self._rng = rng

    def get_trials(
        self, *, finished: bool | None = None, search_tag: str | None = None
    ) -> Iterable[Trial]:
        """Retrieve trials across all component sources.

        :param finished: Filter by finished state.
        :type finished: bool | None
        :param search_tag: Optional tag string filter.
        :type search_tag: str | None
        :return: An iterable of stored trials.
        :rtype: collections.abc.Iterable[StaleTrial]
        """
        sources = list(self._sources)
        pool: list[Source | Iterator[Trial]] = list(sources)
        available: list[bool] = [True] * len(pool)
        while any(available):
            i = self._rng.randrange(len(pool))
            iterator = pool[i]
            if isinstance(iterator, Source):
                iterator = pool[i] = iter(
                    iterator.get_trials(finished=finished, search_tag=search_tag)
                )
            try:
                yield next(iterator)
                available[i] = True
            except StopIteration:
                pool[i] = sources[i]
                available[i] = False

    def consume_and_tag(
        self,
        applied_tag: str,
        *,
        finished: bool | None = None,
        search_tag: str | None = None,
        predicate: PredicateType = lambda _: True,
    ) -> Trial | None:
        """Select a component source at random, consume and tag a trial from it.

        :param applied_tag: The tag to apply.
        :type applied_tag: str
        :param finished: Filter by finished state.
        :type finished: bool | None
        :param search_tag: Optional tag string filter.
        :type search_tag: str | None
        :return: A tagged StaleTrial, or None.
        :rtype: StaleTrial | None
        """
        if not self._sources:
            return None
        retries = 2 * len(self._sources)
        for _ in range(retries):
            comp = self._rng.choice(self._sources)
            trial = comp.consume_and_tag(
                applied_tag=applied_tag,
                finished=finished,
                search_tag=search_tag,
                predicate=predicate,
            )
            if trial is not None:
                return trial
        return None


class CopyingMultiSink(Sink):
    """A trial sink that copies every received trial to all component sinks."""

    def __init__(self, components: Iterable[Sink]) -> None:
        """Initialize the CopyingMultiSink.

        :param components: An iterable of sink components.
        :type components: collections.abc.Iterable[Sink]
        """
        super().__init__()
        self._sinks = list(components)

    def put_trial(self, trial: Trial) -> None:
        """Store the trial into all component sinks.

        :param trial: The trial to record.
        :type trial: Trial
        """
        for comp in self._sinks:
            comp.put_trial(trial)


class RandomMultiSink(Sink):
    """A trial sink that records each trial to a single, randomly chosen component sink."""

    def __init__(
        self, components: Iterable[Sink], rng: random.Random | None = None
    ) -> None:
        """Initialize the RandomMultiSink.

        :param components: An iterable of sink components.
        :type components: collections.abc.Iterable[Sink]
        :param rng: Optional random number generator.
        :type rng: random.Random | None
        """
        super().__init__()
        self._sinks = list(components)
        if rng is None:
            rng = random.Random()
        self._rng = rng

    def put_trial(self, trial: Trial) -> None:
        """Store the trial into a randomly selected component sink.

        :param trial: The trial to record.
        :type trial: Trial
        """
        if self._sinks:
            pool = list(self._sinks)
            comp = self._rng.choice(pool)
            while isinstance(comp, HasPredicate) and not comp.get_predicate()(trial):
                pool.remove(comp)
                if not pool:
                    return None
                comp = self._rng.choice(pool)
            return comp.put_trial(trial)


class HasPredicate(ABC):
    @abstractmethod
    def get_predicate(self) -> PredicateType:
        pass


class FilterSource(Source, HasPredicate):
    def __init__(self, decorated: Source, predicate: PredicateType) -> None:
        super().__init__()
        self._decorated = decorated
        self._predicate = predicate

    def get_trials(
        self, *, finished: bool | None = None, search_tag: str | None = None
    ) -> Iterable[Trial]:
        for trial in self._decorated.get_trials(
            finished=finished, search_tag=search_tag
        ):
            if self._predicate(trial):
                yield trial

    def get_predicate(self) -> PredicateType:
        return self._predicate

    def consume_and_tag(
        self,
        applied_tag: str,
        *,
        finished: bool | None = None,
        search_tag: str | None = None,
        predicate: PredicateType = lambda _: True,
    ) -> Trial | None:
        return self._decorated.consume_and_tag(
            applied_tag,
            finished=finished,
            search_tag=search_tag,
            predicate=lambda trial: self._predicate(trial) and predicate(trial),
        )


class FilterSink(Sink, HasPredicate):
    def __init__(self, decorated: Sink, predicate: PredicateType) -> None:
        super().__init__()
        self._decorated = decorated
        self._predicate = predicate

    def get_predicate(self) -> PredicateType:
        return self._predicate

    def put_trial(self, trial: Trial) -> None:
        if self._predicate(trial):
            return self._decorated.put_trial(trial)
