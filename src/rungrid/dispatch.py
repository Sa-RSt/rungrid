import datetime
import os
import traceback
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from joblib import Memory, Parallel
from typing_extensions import Self

from rungrid.error import StructureError
from rungrid.experiment import (
    Experiment,
    ExperimentRegistry,
    ExperimentStep,
    SignalDone,
    SignalNextStep,
    SignalPrune,
    StepRecord,
    Trial,
    TrialResult,
)


def _last_record(it: Iterable[StepRecord]) -> StepRecord | None:
    last = None
    for r in it:
        last = r
    return last


class Scheduler(ABC):
    @abstractmethod
    def schedule(self, trials: Iterable[Trial]) -> Iterable[Trial]:
        pass

    @abstractmethod
    def get_experiment(self) -> Experiment:
        pass


class LocalDispatcher(Scheduler):
    """A dispatcher for running experiment trials locally.

    This class handles the execution and scheduling of experimental trials
    using local caching to avoid redundant runs.
    """

    def __init__(
        self, experiment: Experiment, cache_provider: Memory, job_provider: Parallel
    ) -> None:
        """Initialize the LocalDispatcher with a cache provider.

        :param cache_provider: The joblib Memory instance used for caching.
        :type cache_provider: joblib.Memory
        """
        self._experiment = experiment
        self._mem = cache_provider
        self._job = job_provider

    @classmethod
    def make_default(cls, experiment: Experiment, cache_dir: os.PathLike) -> Self:
        """Create a default LocalDispatcher instance using a local cache directory.

        :param cache_dir: Path to the directory where cached results will be saved.
        :type cache_dir: os.PathLike
        :return: A default-configured LocalDispatcher.
        :rtype: LocalDispatcher
        """
        return cls(
            experiment,
            Memory(location=Path(cache_dir), backend="local", mmap_mode="c", verbose=0),
            Parallel(return_as="generator_unordered"),
        )

    def schedule(self, trials: Iterable[Trial]) -> Iterable[Trial]:
        """Schedule a trial for asynchronous execution.

        :param trial: The trial instance to be scheduled.
        :type trial: Trial
        :return: A future that will resolve to the trial result.
        :rtype: concurrent.futures.Future[TrialResult]
        """
        return self._job(self.run(x) for x in trials)  # type: ignore

    def get_experiment(self) -> Experiment:
        return self._experiment

    def run(self, trial: Trial) -> Trial:
        kwargs = {}
        current_step = self._experiment._get_first_step()
        while not trial.is_finished():
            trial, current_step, kwargs = run_step(current_step, trial, **kwargs)
        assert trial.result is not None
        return trial


def _experiment_from_step(step: ExperimentStep) -> Experiment:
    reg = ExperimentRegistry.get_instance()
    experiment = reg.get_by_name(step.experiment_name)
    if experiment is None:
        raise StructureError(
            f"{step!r}: experiment not loaded! Make sure to import the module under which it is declared"
        )
    if step not in experiment.steps.values():
        raise StructureError(
            f"{step!r} is not contained in {experiment!r}! Did you declare multiple experiments under the same name?"
        )
    return experiment


def run_step(
    step: ExperimentStep, trial: Trial, **kwargs
) -> tuple[Trial, ExperimentStep | None, dict[str, Any] | None]:
    experiment = _experiment_from_step(step)
    start = datetime.datetime.now()
    try:
        res = step.fn(trial, **kwargs)
        raise SignalDone(res)
    except SignalDone as e:
        end = datetime.datetime.now()
        tr = TrialResult.make_ok(e.result, end)
        trial.record_step(step, start, end, None)
        return trial.with_result(tr), None, None
    except SignalNextStep as e:
        end = datetime.datetime.now()
        trial.record_step(step, start, end, e.next_step.name, **e.kwargs)
        return trial, e.next_step, e.kwargs
    except SignalPrune as e:
        end = datetime.datetime.now()
        tr = TrialResult.make_pruned(e.reason, end)
        return trial.with_result(tr), None, None
    except Exception as e:
        if isinstance(e, tuple(step.recover_from)):
            print(f"Exception ignored in {step!r}:")
            traceback.print_exc()
            record = _last_record(trial.step_records)
            if record is None or record.next_step_name is None:
                return trial, step, kwargs
            else:
                return (
                    trial,
                    experiment.steps[record.next_step_name],
                    record.next_step_kwargs or {},
                )
        end = datetime.datetime.now()
        tr = TrialResult.make_error(e, end)
        return trial.with_result(tr), None, None
