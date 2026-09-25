"""Serialization wrapper and base components for trial sources and sinks."""

import importlib.util
import pickle
import random
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from io import BytesIO
from typing import IO, Any, Literal, Sequence, TypeVar

import joblib

from rungrid.experiment import StaleTrial, StepRecord, Trial

PredicateType = Callable[[Trial], bool]
T = TypeVar("T")


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

        The `make*` methods are probably more suitable for most use cases than calling
        the constructor directly.

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
        """Return a string representation of the LoaderDumper instance.

        :return: A string representation indicating the backend descriptor.
        :rtype: str
        """
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
            loader_kwargs_wo = {"weights_only": False}
            loader_kwargs_wo.update(loader_kwargs)
            return cls.make_torch(
                loader_args, loader_kwargs_wo, dumper_args, dumper_kwargs
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
        """Wrap this sink with a filtering predicate.

        :param predicate: A filter function returning True for trials to accept.
        :type predicate: PredicateType
        :return: A FilterSink wrapping this sink.
        :rtype: FilterSink
        """
        return FilterSink(self, predicate)

    @abstractmethod
    def put_trial(self, trial: Trial) -> None:
        """Store or record a trial.

        :param trial: The trial instance to store.
        :type trial: Trial
        """
        pass

    def put_trials(self, trials: Sequence[Trial]) -> None:
        """Store multiple trials in the sink.

        :param trials: A sequence of trials to store.
        :type trials: collections.abc.Sequence[Trial]
        """
        for trial in trials:
            self.put_trial(trial)


class Source(ABC):
    """Abstract base class representing a read-only source for experimental trials."""

    def with_source_filter(self, predicate: PredicateType) -> "FilterSource":
        """Wrap this source with a filtering predicate.

        :param predicate: A filter function returning True for trials to yield.
        :type predicate: PredicateType
        :return: A FilterSource wrapping this source.
        :rtype: FilterSource
        """
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

    def __repr__(self) -> str:
        return "Empty()"


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
        :rtype: collections.abc.Iterable[Trial]
        """

        # The iterable returned by get_trials can be infinite,
        # so you can't just pick a component at random and
        # yield from it.
        sources = list(self._sources)
        pool: list[Source | Iterator[Trial]] = list(sources)
        while pool:
            i = self._rng.randrange(len(pool))
            iterator = pool[i]
            if isinstance(iterator, Source):
                iterator = pool[i] = iter(
                    iterator.get_trials(finished=finished, search_tag=search_tag)
                )
            try:
                yield next(iterator)
            except StopIteration:
                pool = [x for x in pool if x is not iterator]

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
        :rtype: Trial | None
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

        # Before returning None, actually make sure no components
        # have available trials.
        for comp in self._sources:
            trial = comp.consume_and_tag(
                applied_tag=applied_tag,
                finished=finished,
                search_tag=search_tag,
                predicate=predicate,
            )
            if trial is not None:
                return trial
        return None

    def __repr__(self) -> str:
        return f"MultiSource({repr(self._sources)})"


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

    def put_trials(self, trials: Sequence[Trial]) -> None:
        """Store multiple trials in all component sinks.

        :param trials: A sequence of trials to record.
        :type trials: collections.abc.Sequence[Trial]
        """
        for comp in self._sinks:
            comp.put_trials(trials)

    def __repr__(self) -> str:
        return f"CopyingMultiSink({repr(self._sinks)})"


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
                pool = [x for x in pool if x is not comp]
                if not pool:
                    return None
                comp = self._rng.choice(pool)
            return comp.put_trial(trial)

    def __repr__(self) -> str:
        return f"RandomMultiSink({repr(self._sinks)})"


class HasPredicate(ABC):
    """Abstract base class for components that filter trials using a predicate function."""

    @abstractmethod
    def get_predicate(self) -> PredicateType:
        """Get the filtering predicate function.

        :return: The predicate function.
        :rtype: PredicateType
        """
        pass


class FilterSource(Source, HasPredicate):
    """A trial source that filters trials from an underlying source using a predicate."""

    def __init__(self, decorated: Source, predicate: PredicateType) -> None:
        """Initialize the FilterSource.

        :param decorated: The underlying source to retrieve trials from.
        :type decorated: Source
        :param predicate: The predicate function to filter trials.
        :type predicate: PredicateType
        """
        super().__init__()
        self._decorated = decorated
        self._predicate = predicate

    def get_trials(
        self, *, finished: bool | None = None, search_tag: str | None = None
    ) -> Iterable[Trial]:
        """Retrieve filtered trials matching criteria from the decorated source.

        :param finished: Filter by finished state.
        :type finished: bool | None
        :param search_tag: Optional tag string filter.
        :type search_tag: str | None
        :return: An iterable of filtered Trial instances.
        :rtype: collections.abc.Iterable[Trial]
        """
        for trial in self._decorated.get_trials(
            finished=finished, search_tag=search_tag
        ):
            if self._predicate(trial):
                yield trial

    def get_predicate(self) -> PredicateType:
        """Get the filtering predicate function.

        :return: The predicate function.
        :rtype: PredicateType
        """
        return self._predicate

    def consume_and_tag(
        self,
        applied_tag: str,
        *,
        finished: bool | None = None,
        search_tag: str | None = None,
        predicate: PredicateType = lambda _: True,
    ) -> Trial | None:
        """Find, tag, and return a trial that satisfies both the source predicate and the input predicate.

        :param applied_tag: The tag to append to the found trial.
        :type applied_tag: str
        :param finished: Filter by finished state.
        :type finished: bool | None
        :param search_tag: Optional tag filter to apply.
        :type search_tag: str | None
        :param predicate: An additional filter function.
        :type predicate: PredicateType
        :return: The selected, tagged trial copy, or None if no match is found.
        :rtype: Trial | None
        """
        return self._decorated.consume_and_tag(
            applied_tag,
            finished=finished,
            search_tag=search_tag,
            predicate=lambda trial: self._predicate(trial) and predicate(trial),
        )

    def __repr__(self) -> str:
        return f"FilterSource({self._decorated}, {self._predicate})"


class FilterSink(Sink, HasPredicate):
    """A trial sink that filters trials using a predicate before storing them in an underlying sink."""

    def __init__(self, decorated: Sink, predicate: PredicateType) -> None:
        """Initialize the FilterSink.

        :param decorated: The underlying sink to write accepted trials to.
        :type decorated: Sink
        :param predicate: The predicate function to filter trials.
        :type predicate: PredicateType
        """
        super().__init__()
        self._decorated = decorated
        self._predicate = predicate

    def get_predicate(self) -> PredicateType:
        """Get the filtering predicate function.

        :return: The predicate function.
        :rtype: PredicateType
        """
        return self._predicate

    def put_trial(self, trial: Trial) -> None:
        """Store the trial if it satisfies the predicate.

        :param trial: The trial instance to store.
        :type trial: Trial
        """
        if self._predicate(trial):
            return self._decorated.put_trial(trial)

    def put_trials(self, trials: Sequence[Trial]) -> None:
        """Store trials that satisfy the predicate.

        :param trials: A sequence of Trial instances to store.
        :type trials: collections.abc.Sequence[Trial]
        """
        return self._decorated.put_trials([x for x in trials if self._predicate(x)])

    def __repr__(self) -> str:
        return f"FilterSink({self._decorated}, {self._predicate})"


class ResultPodiumSink(Sink):
    """A trial sink that keeps track of the top-k best trials based on their results.

    Maintains a "podium" (leaderboard) of trials, sorted by their result value.
    Supports minimizing or maximizing the objective result. When the podium
    changes (i.e., a new trial enters the podium or the order of trials on the
    podium changes), an optional callback is triggered.
    """

    def __init__(
        self,
        *,
        direction: Literal["minimize", "maximize"] = "minimize",
        top_k: int = 1,
        on_winners_update: Callable[[list[Trial]], Any] = lambda _: None,
    ) -> None:
        """Initialize the ResultPodiumSink.

        :param direction: Whether to minimize or maximize the trial result. Must be "minimize" or "maximize".
        :type direction: str
        :param top_k: The maximum number of winning trials to keep on the podium.
        :type top_k: int
        :param on_winners_update: Callback called with the updated list of winning trials whenever the podium changes.
        :type on_winners_update: collections.abc.Callable[[list[Trial]], typing.Any]
        """
        super().__init__()
        choices = ["maximize", "minimize"]
        if direction not in choices:
            raise ValueError(f"{direction} not in {choices}")
        self._reverse = direction == "maximize"
        self._top_k = top_k
        self._podium: list[tuple[Any, Trial]] = []
        self._on_winners_update = on_winners_update

    def put_trial(self, trial: Trial) -> None:
        """Store a trial and update the podium if its result is eligible.

        Ignores trials without a result, and trials that failed with an error
        or were pruned. If the trial's result qualifies it for the top-k
        podium, the podium is updated, sorted, and the `on_winners_update`
        callback is triggered if the podium content or order changed.

        :param trial: The trial instance to store and potentially place on the podium.
        :type trial: Trial
        """
        if trial.result is None:
            return
        if trial.result.is_error or trial.result.is_pruned:
            return
        old_podium = self._podium.copy()
        self._podium.append((trial.result.result, trial))
        self._podium.sort(reverse=self._reverse)
        self._podium = self._podium[: min(len(self._podium), self._top_k)]
        if len(old_podium) != len(self._podium) or any(
            x is not y for x, y in zip(old_podium, self._podium)
        ):
            print(self.get_winners())
            self._on_winners_update(self.get_winners())

    def get_winner(self) -> Trial | None:
        """Retrieve the single best trial currently on the podium.

        :return: The overall best trial, or None if the podium is empty.
        :rtype: Trial | None
        """
        if self._podium:
            return self._podium[0][1]

    def get_winners(self) -> list[Trial]:
        """Retrieve all trials currently on the podium, ordered from best to worst.

        :return: A list of the winning Trial instances currently on the podium.
        :rtype: list[Trial]
        """
        return [x[1] for x in self._podium]

    def __repr__(self) -> str:
        direction = {True: "maximize", False: "minimize"}[self._reverse]
        return f"ResultPodiumSink(direction={direction}, top_k={self._top_k})"


class StateSink(Sink):
    """A trial sink that extracts specific arguments from the trial's step records and passes them to a callback.

    Useful for tracking, checkpointing, or saving specific states (such as model
    parameters or configuration dicts) that were scheduled or passed as keyword
    arguments to future experiment steps.
    """

    def __init__(
        self,
        step_name: str,
        arg_name: str,
        dest: Callable[[Trial, Any | list[Any]], Any],
        last_only: bool = True,
    ) -> None:
        """Initialize the StateSink.

        :param step_name: The name (or suffix) of the next step to look for in step records.
        :type step_name: str
        :param arg_name: The name of the argument inside the step's next_step_kwargs to extract.
        :type arg_name: str
        :param dest: Callback function to receive the extracted argument value(s). Signature: `dest(trial, value)`.
        :type dest: collections.abc.Callable[[Trial, typing.Any], typing.Any]
        :param last_only: If True, only extracts from the last relevant step record. If False, extracts all matching ones as a list.
        :type last_only: bool
        """
        super().__init__()
        self._step_name = step_name
        self._arg_name = arg_name
        self._dest = dest
        self._last_only = last_only

    def _record_is_relevant(self, rec: StepRecord) -> bool:
        return (
            rec.next_step_kwargs is not None
            and self._arg_name in rec.next_step_kwargs.keys()
            and rec.next_step_name is not None
            and rec.next_step_name.split(".")[-1] == self._step_name
        )

    def put_trial(self, trial: Trial) -> None:
        """Process the trial's step records and send relevant argument values to the destination callback.

        Searches for step records whose next step name matches `step_name` and contains `arg_name`
        in its keyword arguments. If `last_only` is True, only the last matching record's argument
        value is passed. Otherwise, all matching argument values are passed as a list.

        :param trial: The trial instance to process.
        :type trial: Trial
        """
        recs = trial.step_records
        if self._last_only:
            for i in range(len(recs) - 1, -1, -1):
                rec = recs[i]
                if self._record_is_relevant(rec):
                    return self._dest(
                        trial,
                        rec.next_step_kwargs[self._arg_name],  # type: ignore
                    )
        else:
            self._dest(
                trial,
                [
                    rec.next_step_kwargs[self._arg_name]  # type: ignore
                    for rec in recs
                    if self._record_is_relevant(rec)
                ],
            )

    def __repr__(self) -> str:
        return f"StateSink(step_name={self._step_name}, arg_name={self._arg_name}, last_only={self._last_only})"
