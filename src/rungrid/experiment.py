"""Experiment, Trial, and Sampler declarations and related framework signals."""

import random
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from types import MethodType
from typing import Any, Generic, Iterable, NoReturn, Sequence, Type, TypeVar, final
from uuid import UUID

import optuna
from optuna.distributions import CategoricalChoiceType
from typing_extensions import Self

from rungrid.error import StructureError

T = TypeVar("T")


def _check_range(minimum, maximum) -> None:
    if minimum > maximum:
        raise ValueError(f"minimum ({minimum}) > maximum ({maximum})")

    if minimum == maximum:
        raise ValueError(
            f"minimum ({minimum}) == maximum ({maximum}), use Sampler.precomputed instead"
        )


def _check_type(name: str, val: Any, t: Type) -> None:
    if not isinstance(val, t):
        raise TypeError(f"{name}: expected type {t.__name__}, got {type(val).__name__}")


def _check_length(length: int) -> None:
    _check_type("length", length, int)
    if length < 0:
        raise ValueError("length can't be negative")


def _check_not_sampling_strategy(name: str, v: Any) -> None:
    if isinstance(v, SamplingStrategy):
        raise TypeError(
            f"{name} can't be of type SamplingStrategy; please store it in a VarNamespace"
            + "first or sample manually with SamplingStrategy.sample(variable_name)"
        )


class SamplingStrategy:
    """Represents a strategy for sampling experiment variables.

    A strategy is identified by a name and wraps a function that performs
    the actual value sampling.
    """

    def __init__(
        self, name: str, sample_fn: Callable[[str], Any], taken_variable_names: set[str]
    ) -> None:
        """Initialize the SamplingStrategy.

        :param name: The name of the sampling strategy.
        :type name: str
        :param sample_fn: A callable that generates a value given a variable name.
        :type sample_fn: collections.abc.Callable[[str], typing.Any]
        :param taken_variable_names: A set tracking already declared variable names to ensure uniqueness.
        :type taken_variable_names: set[str]
        """
        self._name = name
        self._sample_fn = sample_fn
        self._variable_names = taken_variable_names

    @property
    def name(self) -> str:
        """Get the name of the sampling strategy.

        :return: The strategy name.
        :rtype: str
        """
        return self._name

    @classmethod
    def make_precomputed(cls, value: Any, taken_variable_names: set[str]) -> Self:
        """Create a SamplingStrategy that always yields a precomputed value.

        :param value: The constant value to return upon sampling.
        :type value: typing.Any
        :param taken_variable_names: A set tracking already declared variable names to ensure uniqueness.
        :type taken_variable_names: set[str]
        :return: A SamplingStrategy representing a precomputed constant.
        :rtype: SamplingStrategy
        """
        return cls(f"precomputed({value})", lambda _: value, taken_variable_names)

    def sample(self, variable_name: str) -> Any:
        """Sample a value for a specific variable name.

        :param variable_name: The name of the variable being sampled.
        :type variable_name: str
        :return: The sampled value.
        :rtype: typing.Any
        :raises StructureError: If the variable name is not unique within the tracker or if this sampler
        has been called more than once.
        """
        if variable_name in self._variable_names:
            raise StructureError(f'variable name "{variable_name}" is not unique')
        self._variable_names.add(variable_name)
        if self._sample_fn is None:
            raise StructureError(
                "this sampler has already been disposed; "
                + "make sure all sampling is done inside the variables() method."
            )
        res = self._sample_fn(variable_name)
        return res

    def dispose(self) -> None:
        """Make it so that this sampler can no longer be called.

        This must be called before pickling.
        """
        self._sample_fn = None


class Sampler:
    """A wrapper around an Optuna trial to suggest and construct sampling strategies.

    Provides high-level APIs to define various parameter distributions (integers, floats,
    categorical options, combinations, and sequences) while maintaining parameter trackers.
    """

    def __init__(self, optuna_trial: optuna.Trial) -> None:
        """Initialize the Sampler.

        :param optuna_trial: The underlying Optuna trial used for suggesting values.
        :type optuna_trial: optuna.Trial
        """
        self._optuna_trial = optuna_trial
        self._variable_names: set[str] = set()

    def _maybe_wrap_precomputed(self, s) -> SamplingStrategy:
        if not isinstance(s, SamplingStrategy):
            return self.precomputed(s)
        return s

    def precomputed(self, k: Any) -> SamplingStrategy:
        """Create a precomputed sampling strategy returning a constant value.

        :param k: The constant value to return.
        :type k: typing.Any
        :return: A SamplingStrategy yielding the constant.
        :rtype: SamplingStrategy
        """
        return SamplingStrategy.make_precomputed(k, self._variable_names)

    def integer(self, lower: int, upper: int) -> SamplingStrategy:
        """Define an integer sampling strategy within a range.

        :param lower: The lower bound of the range (inclusive).
        :type lower: int
        :param upper: The upper bound of the range (inclusive).
        :type upper: int
        :return: An integer SamplingStrategy.
        :rtype: SamplingStrategy
        :raises ValueError: If lower is greater than or equal to upper.
        :raises TypeError: If lower or upper is not an integer.
        """
        _check_not_sampling_strategy("lower", lower)
        _check_not_sampling_strategy("upper", upper)
        _check_type("lower", lower, int)
        _check_type("upper", upper, int)
        _check_range(lower, upper)
        return SamplingStrategy(
            f"integer({lower},{upper})",
            lambda name: self._optuna_trial.suggest_int(name, lower, upper),
            self._variable_names,
        )

    def uniform(self, lower: float, upper: float) -> SamplingStrategy:
        """Define a uniform float sampling strategy within a range.

        :param lower: The lower bound of the range (inclusive).
        :type lower: float
        :param upper: The upper bound of the range (exclusive/inclusive depending on backend).
        :type upper: float
        :return: A uniform float SamplingStrategy.
        :rtype: SamplingStrategy
        :raises ValueError: If lower is greater than or equal to upper.
        :raises TypeError: If lower or upper is not a float.
        """
        _check_not_sampling_strategy("lower", lower)
        _check_not_sampling_strategy("upper", upper)
        _check_type("lower", lower, float)
        _check_type("upper", upper, float)
        _check_range(lower, upper)
        return SamplingStrategy(
            f"uniform({lower},{upper})",
            lambda name: self._optuna_trial.suggest_float(name, lower, upper),
            self._variable_names,
        )

    def log_uniform(self, lower: float, upper: float) -> SamplingStrategy:
        """Define a log-uniform float sampling strategy within a range.

        :param lower: The lower bound of the range (inclusive).
        :type lower: float
        :param upper: The upper bound of the range (exclusive/inclusive depending on backend).
        :type upper: float
        :return: A log-uniform float SamplingStrategy.
        :rtype: SamplingStrategy
        :raises ValueError: If lower is greater than or equal to upper.
        :raises TypeError: If lower or upper is not a float.
        """
        _check_not_sampling_strategy("lower", lower)
        _check_not_sampling_strategy("upper", upper)
        _check_type("lower", lower, float)
        _check_type("upper", upper, float)
        _check_range(lower, upper)
        return SamplingStrategy(
            f"log_uniform({lower},{upper})",
            lambda name: self._optuna_trial.suggest_float(name, lower, upper, log=True),
            self._variable_names,
        )

    def discrete_uniform(
        self, lower: float, upper: float, q: float
    ) -> SamplingStrategy:
        """Define a discrete uniform float sampling strategy with a step size.

        :param lower: The lower bound of the range (inclusive).
        :type lower: float
        :param upper: The upper bound of the range (inclusive).
        :type upper: float
        :param q: The quantization step size.
        :type q: float
        :return: A discrete uniform float SamplingStrategy.
        :rtype: SamplingStrategy
        :raises ValueError: If lower is greater than or equal to upper, or if step size q is too large.
        :raises TypeError: If lower or upper is not a float.
        """
        _check_not_sampling_strategy("lower", lower)
        _check_not_sampling_strategy("upper", upper)
        _check_type("lower", lower, float)
        _check_type("upper", upper, float)
        _check_range(lower, upper)
        if q >= upper - lower:
            raise ValueError("quantization interval too large (q >= upper - lower)")
        return SamplingStrategy(
            f"discrete_uniform({lower},{upper})",
            lambda name: self._optuna_trial.suggest_discrete_uniform(
                name, lower, upper, q
            ),
            self._variable_names,
        )

    def categorical(
        self, categories: Sequence[CategoricalChoiceType]
    ) -> SamplingStrategy:
        """Define a categorical choice sampling strategy.

        :param categories: A sequence of categories to sample from.
        :type categories: collections.abc.Sequence[optuna.distributions.CategoricalChoiceType]
        :return: A categorical SamplingStrategy.
        :rtype: SamplingStrategy
        :raises TypeError: If any option in categories is itself a SamplingStrategy.
        """
        _check_not_sampling_strategy("categories", categories)
        for i in range(len(categories)):
            _check_not_sampling_strategy(f"categories[{i}]", categories[i])
        join = ",".join(str(x) for x in categories)
        return SamplingStrategy(
            f"categorical({join})",
            lambda name: self._optuna_trial.suggest_categorical(name, categories),
            self._variable_names,
        )

    def random(
        self,
        strategies: Sequence[SamplingStrategy | Any],
        rng: random.Random | None = None,
    ) -> SamplingStrategy:
        """Define a sampling strategy that randomly delegates to one of multiple strategies.

        :param strategies: A sequence of strategies or raw values to choose from randomly.
        :type strategies: collections.abc.Sequence[SamplingStrategy | typing.Any]
        :param rng: An optional random number generator instance.
        :type rng: random.Random | None
        :return: A randomly selecting SamplingStrategy.
        :rtype: SamplingStrategy
        :raises TypeError: If strategies itself is a SamplingStrategy or rng is not a Random instance.
        """
        _check_not_sampling_strategy("strategies", strategies)
        if rng is None:
            rng = random.Random()
        else:
            _check_type("rng", rng, random.Random)
        strategies_wrapped = [self._maybe_wrap_precomputed(x) for x in strategies]
        join = "|".join(x.name for x in strategies_wrapped)

        def _random_fn(name):
            index = rng.randrange(len(strategies_wrapped))
            return strategies_wrapped[index].sample(f"{name}_R{index}")

        return SamplingStrategy(f"random({join})", _random_fn, self._variable_names)

    def transformed(
        self,
        ss: SamplingStrategy | Any,
        trans: Callable[[Any], Any],
        alt_name: str | None = None,
    ) -> SamplingStrategy:
        """Define a sampling strategy that applies a function transformation to another strategy's output.

        :param ss: The base sampling strategy or constant value to transform.
        :type ss: SamplingStrategy | typing.Any
        :param trans: A callable that transforms the sampled value.
        :type trans: collections.abc.Callable[[typing.Any], typing.Any]
        :param alt_name: An optional custom name for the transformation; defaults to trans.__name__.
        :type alt_name: str | None
        :return: A transformed SamplingStrategy.
        :rtype: SamplingStrategy
        :raises TypeError: If trans or alt_name are themselves sampling strategies.
        """
        ss = self._maybe_wrap_precomputed(ss)
        _check_not_sampling_strategy("trans", trans)
        _check_not_sampling_strategy("alt_name", alt_name)
        if alt_name is None:
            alt_name = trans.__name__
        else:
            _check_type("alt_name", alt_name, str)
        return SamplingStrategy(
            ss.name + f"/transformed({alt_name})",
            lambda name: trans(ss.sample(name)),
            self._variable_names,
        )

    def list(self, ss: SamplingStrategy | Any, length: int) -> SamplingStrategy:
        """Define a sampling strategy that generates a list of sampled values.

        :param ss: The base sampling strategy or constant value.
        :type ss: SamplingStrategy | typing.Any
        :param length: The exact length of the list to generate.
        :type length: int
        :return: A list-generating SamplingStrategy.
        :rtype: SamplingStrategy
        :raises ValueError: If length is negative.
        :raises TypeError: If length is not an integer or is a SamplingStrategy.
        """
        _check_not_sampling_strategy("length", length)
        _check_length(length)
        ss = self._maybe_wrap_precomputed(ss)
        return SamplingStrategy(
            ss.name + f"/list({length})",
            lambda name: [ss.sample(name + f"_L{i}") for i in range(length)],
            self._variable_names,
        )

    def list_transformed(
        self,
        ss: SamplingStrategy | Any,
        length: int,
        trans: Callable[[int, Any], Any],
        alt_name: str | None = None,
    ) -> SamplingStrategy:
        """Define a sampling strategy that generates a list of transformed values.

        Each element in the list is the result of applying trans(index, sampled_value).

        :param ss: The base sampling strategy or constant value.
        :type ss: SamplingStrategy | typing.Any
        :param length: The exact length of the list to generate.
        :type length: int
        :param trans: A transformation callable accepting (index, value) and returning the transformed element.
        :type trans: collections.abc.Callable[[int, typing.Any], typing.Any]
        :param alt_name: An optional custom name for the transformation; defaults to trans.__name__.
        :type alt_name: str | None
        :return: A list-transformed SamplingStrategy.
        :rtype: SamplingStrategy
        :raises ValueError: If length is negative.
        :raises TypeError: If length is not an integer.
        """
        _check_length(length)
        ss = self._maybe_wrap_precomputed(ss)
        if alt_name is None:
            alt_name = trans.__name__
        return SamplingStrategy(
            ss.name + f"/list_transformed({alt_name}:{length})",
            lambda name: [trans(i, ss.sample(name)) for i in range(length)],
            self._variable_names,
        )


class VarNamespace:
    """A namespace for tracking and accessing trial variables and their values.

    Variables can be set either as constant values (using standard attribute or item
    assignment) or as dynamically sampled values (using the `sampled` method).
    """

    def __init__(self, _variable_names: set[str]) -> None:
        """Initialize the VarNamespace.

        :param _variable_names: A set tracking registered variable names.
        :type _variable_names: set[str]
        """
        self._variables_dict: dict[str, Any] = {}
        self._sampling_strategies: dict[str, SamplingStrategy] = {}
        self._variable_names = _variable_names
        self._stale = False

    def _as_stale(self) -> "VarNamespace":
        vn = VarNamespace(self._variable_names.copy())
        vn._variables_dict = self._variables_dict.copy()
        vn._sampling_strategies = self._sampling_strategies.copy()
        vn._stale = True
        return vn

    def __setattr__(self, name: str, value: Any, /) -> None:
        """Assign a constant value to a variable in this namespace.

        This automatically creates a precomputed SamplingStrategy under the hood.

        :param name: The name of the variable to set.
        :type name: str
        :param value: The constant value to assign.
        :type value: typing.Any
        """
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            ss = SamplingStrategy.make_precomputed(value, self._variable_names)
            self.sampled(name, ss)

    def _set_variable(self, name: str, value: Any, /) -> None:
        if name in [
            "_variables_dict",
            "_sampling_strategies",
            "_variable_names",
            "_stale",
            "_as_stale",
        ]:
            super().__setattr__(name, value)
        elif isinstance(value, SamplingStrategy):
            raise ValueError(
                "must sample before setting the variable; "
                + "try using VarNamespace.sampled('name', x) instead of VarNamespace.name = x, "
                + "as the latter should only be used to set constant values"
            )
        else:
            self._variables_dict[name] = value

    def __getattr__(self, name: str, /) -> Any:
        """Retrieve a variable's value as an attribute.

        :param name: The name of the variable.
        :type name: str
        :return: The value associated with the variable.
        :rtype: typing.Any
        :raises AttributeError: If the variable does not exist.
        """
        if name.startswith("_") or "_variables_dict" not in self.__dict__:
            raise AttributeError(f"'VarNamespace' object has no attribute {name!r}")
        try:
            return self._variables_dict[name]
        except KeyError:
            raise AttributeError(f"'VarNamespace' object has no attribute {name!r}")

    def __getitem__(self, name: str, /) -> Any:
        """Retrieve a variable's value using bracket notation.

        :param name: The name of the variable.
        :type name: str
        :return: The value associated with the variable.
        :rtype: typing.Any
        :raises KeyError: If the variable does not exist.
        """
        return self._variables_dict[name]

    def __setitem__(self, name: str, val: Any, /) -> None:
        """Assign a constant value to a variable using bracket notation.

        :param name: The name of the variable.
        :type name: str
        :param val: The constant value to assign.
        :type val: typing.Any
        """
        setattr(self, name, val)

    def sampled(self, name: str, ss: SamplingStrategy) -> None:
        """Register and sample a variable using a sampling strategy.

        :param name: The name of the variable to sample.
        :type name: str
        :param ss: The sampling strategy used to generate the variable's value.
        :type ss: SamplingStrategy
        """
        self._set_variable(name, ss.sample(name))
        self._sampling_strategies[name] = ss


@dataclass
class ExperimentStep:
    """Represents a discrete step in an experiment.

    Tracks execution metadata such as caching options, exceptions to
    recover from, argument names to drop and experiment information.
    """

    name: str
    cacheable: bool
    recover_from: list[type[BaseException]]
    drop_args: set[str]
    experiment_ident: str
    experiment_name: str
    fn: Callable

    def record(
        self,
        start: datetime,
        end: datetime,
        next_step_name: str | None = None,
        **kwargs,
    ) -> "StepRecord":
        """Record the execution of this experiment step.

        :param start: The start timestamp of the step execution.
        :type start: datetime.datetime
        :param next_step_name: The name of the next step, if any.
        :type next_step_name: str | None
        :param kwargs: The keyword arguments containing inputs for the next step.
        :return: A record of this step execution.
        :rtype: StepRecord
        """
        return StepRecord(self, next_step_name, kwargs, start, end)


@dataclass
class _ExperimentStepBuilder:
    name: str
    cacheable: bool
    fn: Callable
    recover_from: list[type[BaseException]]
    drop_args: set[str]

    def build(self, experiment: "Experiment"):
        bound_fn = MethodType(self.fn, experiment)
        return ExperimentStep(
            name=self.name,
            cacheable=self.cacheable,
            fn=bound_fn,
            recover_from=self.recover_from,
            drop_args=self.drop_args,
            experiment_ident=experiment.identifier,
            experiment_name=experiment.name,
        )


@dataclass
class StepRecord:
    """A record of an executed experiment step.

    Captures the step definition, start time, next step name, and the keyword arguments to be fed into the next step.
    """

    step: ExperimentStep
    next_step_name: str | None
    next_step_kwargs: dict | None
    start: datetime
    end: datetime


@dataclass
class TrialResult:
    """The outcome of a trial run.

    Stores the final result value, any encountered error, success status, and completion time.
    """

    result: Any
    error: Any
    prune_reason: str | None
    is_error: bool
    is_pruned: bool
    end: datetime

    @classmethod
    def make_error(cls, error: Any, end: datetime) -> Self:
        """Create a failed TrialResult containing an error.

        :param error: The exception or error object.
        :type error: typing.Any
        :param end: The completion timestamp.
        :type end: datetime.datetime
        :return: A failed TrialResult.
        :rtype: TrialResult
        """
        return cls(None, error, None, True, False, end)

    @classmethod
    def make_pruned(cls, reason: str, end: datetime) -> Self:
        """Create a pruned TrialResult containing a prune reason.

        :param reason: The reason why the trial was pruned.
        :type reason: str
        :param end: The completion timestamp.
        :type end: datetime.datetime
        :return: A pruned TrialResult.
        :rtype: TrialResult
        """
        return cls(None, None, reason, False, True, end)

    @classmethod
    def make_ok(cls, result: Any, end: datetime) -> Self:
        """Create a successful TrialResult containing the output value.

        :param result: The successful return value.
        :type result: typing.Any
        :param end: The completion timestamp.
        :type end: datetime.datetime
        :return: A successful TrialResult.
        :rtype: TrialResult
        """
        return cls(result, None, None, False, False, end)

    def has_result(self) -> bool:
        """Check if the trial completed successfully without errors.

        :return: True if the trial was successful, False otherwise.
        :rtype: bool
        """
        return not self.is_error and not self.is_pruned


_fqn_to_step_builder: dict[str, _ExperimentStepBuilder] = dict()


def _get_fqn(fn: Callable) -> str:
    return f"{fn.__module__}.{fn.__qualname__}"


def step_method(
    *,
    cache: bool = False,
    recover_from: Iterable[type[BaseException]] = [KeyboardInterrupt],
    drop_args: Iterable[str] = [],
):
    """Decorate a method of an Experiment subclass as an experiment step.

    :param cache: Whether the step result is cacheable.
    :type cache: bool
    :param recover_from: Exceptions to ignore/recover from during this step.
    :type recover_from: collections.abc.Iterable[type[BaseException]]
    :param drop_args: Argument names to drop before passing to the next step.
    :type drop_args: collections.abc.Iterable[str]
    :return: A decorator function that wraps the step method.
    :rtype: collections.abc.Callable
    """

    def decorator(fn: Callable):
        fqn = _get_fqn(fn)
        _fqn_to_step_builder[fqn] = _ExperimentStepBuilder(
            fqn, cache, fn, list(recover_from), set(drop_args)
        )
        return fn

    return decorator


class ExperimentSignal(BaseException):
    """Base exception class used for signaling experimental control flow.

    Signals are used to transition between steps, end trials, or prune trials.
    """

    pass


class SignalDone(ExperimentSignal):
    """Signal indicating that the experiment has successfully completed.

    Carries the final result of the experiment.
    """

    def __init__(self, result: Any) -> None:
        """Initialize the SignalDone.

        :param result: The final output of the experiment.
        :type result: typing.Any
        """
        super().__init__()
        self.result = result


class SignalNextStep(ExperimentSignal):
    """Signal indicating that the experiment should transition to a next step.

    Carries the next step to execute and its input keyword arguments.
    """

    def __init__(self, next_step: ExperimentStep, kwargs: dict[str, Any]) -> None:
        """Initialize the SignalNextStep.

        :param next_step: The metadata of the next step to run.
        :type next_step: ExperimentStep
        :param kwargs: The keyword arguments to pass as inputs to that next step.
        :type kwargs: dict[str, typing.Any]
        """
        super().__init__()
        self.next_step = next_step
        self.kwargs = kwargs


class SignalPrune(ExperimentSignal):
    """Signal indicating that the current trial should be pruned (aborted early).

    Carries a reason explanation for why the trial was pruned.
    """

    def __init__(self, reason: str) -> None:
        """Initialize the SignalPrune.

        :param reason: The reason for pruning the trial.
        :type reason: str
        """
        super().__init__()
        self.reason = reason


class ExperimentRegistry:
    """Registry to keep track of loaded Experiment instances by name and identifier."""

    def __init__(self) -> None:
        """Initialize the ExperimentRegistry."""
        self._experiments_by_name: dict[str, Experiment] = {}
        self._experiments_by_ident: dict[str, Experiment] = {}

    @classmethod
    @lru_cache
    def get_instance(cls):
        """Get the global single instance of the registry.

        :return: The global ExperimentRegistry.
        :rtype: ExperimentRegistry
        """
        return cls()

    def register(self, exp: "Experiment") -> None:
        """Register an experiment instance.

        :param exp: The experiment to register.
        :type exp: Experiment
        :raises StructureError: If an experiment with the same name is already registered.
        """
        if exp.name in self._experiments_by_name.keys():
            raise StructureError(
                f'non-unique experiment name "{exp.name}". Do you have '
                + "multiple subclasses of Experiment with the same name?"
            )
        assert exp.identifier not in self._experiments_by_ident.keys()
        self._experiments_by_name[exp.name] = exp
        self._experiments_by_ident[exp.identifier] = exp

    def get_by_name(self, name: str) -> "Experiment | None":
        """Retrieve a registered experiment by its class name.

        :param name: The name of the experiment class.
        :type name: str
        :return: The registered Experiment instance, or None if not found.
        :rtype: Experiment | None
        """
        return self._experiments_by_name.get(name, None)

    def get_by_ident(self, ident: str) -> "Experiment | None":
        """Retrieve a registered experiment by its unique identifier.

        :param ident: The unique identifier of the experiment.
        :type ident: str
        :return: The registered Experiment instance, or None if not found.
        :rtype: Experiment | None
        """
        return self._experiments_by_ident.get(ident, None)


class Experiment(ABC, Generic[T]):
    """Abstract base class representing an experimental workflow.

    Defines the layout of steps and parameters to be executed and optimized.
    An experiment contains multiple methods prefixed with ``step_`` which are
    registered as experiment steps during initialization.

    All methods and properties defined by this class are either
    abstract (must be overridden) or final (must NOT be overridden).
    """

    @final
    def __init__(self) -> None:
        """Initialize the Experiment and register its step methods.

        Automatically inspects class attributes for methods decorated with `@step_method`,
        registering them as ExperimentStep metadata.
        """
        self._steps: dict[str, ExperimentStep] = {}

        for val in vars(type(self)).values():
            if callable(val):
                fqn = _get_fqn(val)
                try:
                    sb = _fqn_to_step_builder[fqn]
                    self._steps[fqn] = sb.build(self)
                except KeyError:
                    pass
        reg = ExperimentRegistry.get_instance()
        reg.register(self)

    @property
    @final
    def name(self) -> str:
        """Get the name of the experiment. Current behavior is to return the name
        of the class, but that may change in the future.

        :return: The name of the experiment.
        :rtype: str
        """
        return type(self).__name__

    @abstractmethod
    def version(self) -> str:
        """Get a string that uniquely identifies the current version of the experiment.

        Must be overridden by subclasses. See the `rungrid.version` module for utilities
        for automatically generating version strings.

        :return: The experiment's version string.
        :rtype: str
        """
        pass

    @property
    @lru_cache
    @final
    def identifier(self) -> str:
        """Get a string that uniquely identifies this experiment in its current version.
        It is formed by appending the name of the experiment to the hash of its source
        code.

        :return: A unique identifier for the experiment.
        :rtype: str
        """
        return self.name + self.version()

    @abstractmethod
    def variables(self, v: VarNamespace, s: Sampler) -> None:
        """Define and sample variables for the experiment.

        Must be implemented by subclasses to register trial parameters.

        :param v: The variable namespace where sampled parameters are stored.
        :type v: VarNamespace
        :param s: The sampler used to suggest parameter distributions.
        :type s: Sampler
        """
        pass

    @abstractmethod
    def first_step(self) -> Callable:
        """Specify the initial step function of the experiment.

        Must be implemented by subclasses.

        :return: A callable corresponding to the first step of the experiment.
        :rtype: collections.abc.Callable
        """
        pass

    @final
    def _step_from_fn(self, fn: Callable) -> ExperimentStep:
        for _, step in self._steps.items():
            if step.fn is fn or step.fn == fn:
                return step
        raise LookupError(
            f"{fn!r} was not recognized as an experiment step. "
            + "Make sure it is a method of the same experiment and its name is "
            + 'prefixed with "step_".'
        )

    @final
    def _get_first_step(self) -> ExperimentStep:
        return self._step_from_fn(self.first_step())

    @property
    @final
    def steps(self) -> dict[str, ExperimentStep]:
        """Get all registered steps of this experiment.

        :return: A dictionary mapping step names to their ExperimentStep metadata.
        :rtype: dict[str, ExperimentStep]
        """
        return self._steps

    @final
    def exit_done(self, result: T) -> NoReturn:
        """Raise a SignalDone exception to terminate the experiment successfully.

        :param result: The final output value of the experiment.
        :type result: T
        :raises SignalDone: Always, to signal completion.
        """
        raise SignalDone(result)

    @final
    def exit_next_step(self, next_step: Callable, **kwargs) -> NoReturn:
        """Raise a SignalNextStep exception to transition the trial to the next step.

        :param next_step: The callable of the next step to execute.
        :type next_step: collections.abc.Callable
        :param kwargs: The keyword arguments to pass to the next step.
        :raises SignalNextStep: Always, to signal a transition.
        """
        raise SignalNextStep(self._step_from_fn(next_step), kwargs)

    @final
    def exit_prune(self, reason: str) -> NoReturn:
        """Raise a SignalPrune exception to prune the current trial.

        :param reason: Explanatory text for why the trial is being pruned.
        :type reason: str
        :raises SignalPrune: Always, to signal pruning.
        """
        raise SignalPrune(reason)


class Trial(ABC):
    """Abstract base class representing a single trial execution in an experiment.

    Tracks trial metadata, tagged annotations, variables, and execution step records.
    """

    def __init__(
        self,
        optuna_trial_number: int,
        uuid: UUID,
        v: VarNamespace,
        step_records: list[StepRecord] = [],
        tags: set[str] = set(),
    ) -> None:
        """Initialize the Trial.

        :param optuna_trial_number: The trial number assigned by Optuna.
        :type optuna_trial_number: int
        :param uuid: A unique identifier for this trial.
        :type uuid: uuid.UUID
        :param v: The namespace of variables used or sampled by this trial.
        :type v: VarNamespace
        :param step_records: A list of step execution records for this trial.
        :type step_records: list[StepRecord]
        """
        self._optuna_trial_number = optuna_trial_number
        self._uuid = uuid
        self._v = v
        self._tags: set[str] = set(tags)
        self._result: TrialResult | None = None
        self._step_records = list(step_records)

    @property
    def experiment_idents(self) -> set[str]:
        """Get the set of experiment identifiers for all steps executed in this trial.

        :return: A set of unique experiment identifier strings.
        :rtype: set[str]
        """
        return {x.step.experiment_ident for x in self.step_records}

    @property
    def tags(self) -> set[str]:
        """Get the tags associated with this trial.

        :return: A copy of the tag set.
        :rtype: set[str]
        """
        return set(self._tags)

    @property
    def uuid(self) -> UUID:
        """Get the unique identifier of the trial.

        :return: The trial UUID.
        :rtype: uuid.UUID
        """
        return self._uuid

    @property
    def v(self) -> VarNamespace:
        """Get the variables namespace associated with this trial.

        :return: The VarNamespace instance.
        :rtype: VarNamespace
        """
        return self._v

    @property
    def result(self) -> TrialResult | None:
        """Get the outcome of this trial if finished.

        :return: The TrialResult, or None if the trial is not finished.
        :rtype: TrialResult | None
        """
        return self._result

    @property
    def optuna_trial_number(self) -> int:
        """Get the trial index assigned by Optuna.

        :return: The trial number.
        :rtype: int
        """
        return self._optuna_trial_number

    def add_tag(self, tag: str) -> None:
        """Add a tag to this trial.

        :param tag: The tag string to add.
        :type tag: str
        """
        self._tags.add(tag)

    def add_tags(self, tags: Iterable[str]) -> None:
        """Add multiple tags to this trial.

        :param tags: An iterable of tag strings.
        :type tags: collections.abc.Iterable[str]
        """
        self._tags |= set(tags)

    @property
    def step_records(self) -> Iterable[StepRecord]:
        """Get the step records of executed steps in this trial.

        :return: An iterable of StepRecord objects.
        :rtype: collections.abc.Iterable[StepRecord]
        """
        return list(self._step_records)

    def record_step(
        self,
        es: ExperimentStep,
        start: datetime,
        end: datetime,
        next_step_name: str | None = None,
        **kwargs,
    ) -> None:
        """Record the execution of an experiment step in this trial.

        :param es: The experiment step metadata to record.
        :type es: ExperimentStep
        :param start: The start timestamp of the step.
        :type start: datetime.datetime
        :param next_step_name: The name of the next step, if any.
        :type next_step_name: str | None
        :param kwargs: The inputs representing outputs of the current step for the next step.
        """
        for rec in self._step_records:
            if rec.next_step_kwargs is not None:
                for key in list(rec.next_step_kwargs.keys()):
                    if key in es.drop_args:
                        del rec.next_step_kwargs[key]
        self._step_records.append(es.record(start, end, next_step_name, **kwargs))

    def as_stale(self) -> "StaleTrial":
        """Convert this trial to a StoredTrial snapshot.

        :return: A serialized-friendly StoredTrial instance.
        :rtype: StoredTrial
        """
        stale = StaleTrial(
            self.optuna_trial_number,
            self.uuid,
            self.v._as_stale(),
            self._step_records,
            self._tags,
        )
        if self._result is not None:
            stale._result = self._result
        return stale

    def with_result(self, result: TrialResult) -> "StaleTrial":
        """Return a copy of this trial as a StaleTrial updated with the given result.

        :param result: The TrialResult to attach to the trial copy.
        :type result: TrialResult
        :return: A StaleTrial copy with the result set.
        :rtype: StaleTrial
        """
        t = self.as_stale()
        t._result = result
        return t

    def is_finished(self) -> bool:
        """Check if the trial has completed execution.

        :return: True if the trial result is set, False otherwise.
        :rtype: bool
        """
        return self.result is not None


class LiveTrial(Trial):
    """Represents a trial with an Optuna trial associated to it."""

    def __init__(
        self,
        optuna_trial: optuna.Trial,
        uuid: UUID,
        v: VarNamespace,
    ) -> None:
        """Initialize the LiveTrial.

        :param optuna_trial: The underlying Optuna trial instance.
        :type optuna_trial: optuna.Trial
        :param uuid: The unique trial UUID.
        :type uuid: uuid.UUID
        :param v: The variables namespace for this trial.
        :type v: VarNamespace
        """
        super().__init__(optuna_trial._trial_id, uuid, v)
        self._optuna_trial = optuna_trial

    @property
    def optuna_trial(self) -> optuna.Trial:
        """Get the active Optuna trial instance.

        :return: The underlying Optuna trial.
        :rtype: optuna.Trial
        """
        return self._optuna_trial


class StaleTrial(Trial):
    """Represents a trial loaded or reconstructed from storage.

    A stored trial may be either completed or still ongoing.
    """

    pass
