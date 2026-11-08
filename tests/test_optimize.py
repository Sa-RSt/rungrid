import datetime
import itertools
import pickle

import optuna

from rungrid.dispatch import LocalDispatcher
from rungrid.experiment import (
    Experiment,
    TrialResult,
    VarNamespace,
    step_method,
)
from rungrid.optimize import OptimizingSampler


class OptimizeExperimentWithoutOptunaVariables(Experiment):
    """A dummy experiment class for testing."""

    def version(self) -> str:
        return "1.0-without"

    def variables(self, v: VarNamespace, s) -> None:
        v.param1 = s.precomputed(123)

    @step_method(cache=True, first=True)
    def step_start(self, trial, **kwargs):
        return "step_start_result"


class OptimizeExperimentWithOptunaVariables(Experiment):
    """A dummy experiment class for testing."""

    def version(self) -> str:
        return "1.0-with"

    def variables(self, v: VarNamespace, s) -> None:
        v.param1 = s.integer(123, 456)

    @step_method(cache=True, first=True)
    def step_start(self, trial, **kwargs):
        return "step_start_result"


def test_optimizing_sampler_doesnt_create_optuna_trial_if_unnecessary():
    study = optuna.create_study(direction="minimize")
    exp = OptimizeExperimentWithoutOptunaVariables.get_instance()
    s = OptimizingSampler(exp, study)
    trial = next(iter(s.get_trials()))
    assert trial is not None
    assert trial.is_finished() is False
    assert trial.optuna_trial_number is None


def test_optimizing_sampler():
    """Verify that OptimizingSampler yields trials from Optuna and reports back results."""
    # Create an in-memory Optuna study
    study = optuna.create_study(direction="minimize")
    exp = OptimizeExperimentWithOptunaVariables.get_instance()

    sampler = OptimizingSampler(exp, study)

    # Generate an unfinished trial
    trials_it = sampler.get_trials(finished=False)
    trial = next(iter(trials_it))

    assert trial is not None
    assert trial.is_finished() is False
    assert trial.optuna_trial_number == 0

    # Attach a successful result and report back to study
    ok_res = TrialResult.make_ok(0.123, datetime.datetime.now())
    finished_trial = trial.with_result(ok_res)

    sampler.put_trial(finished_trial)

    # Verify the study received the trial value
    assert len(study.trials) == 1
    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
    assert study.trials[0].value == 0.123

    # Generate another trial
    trial2 = next(iter(sampler.get_trials()))
    assert trial2.optuna_trial_number == 1

    # Attach a pruned result and report back to study
    pruned_res = TrialResult.make_pruned("pruned reason", datetime.datetime.now())
    finished_trial2 = trial2.with_result(pruned_res)

    sampler.put_trial(finished_trial2)
    assert len(study.trials) == 2
    assert study.trials[1].state == optuna.trial.TrialState.PRUNED


def test_optimize_pipeline(tmp_path):
    """Verify the entire pipeline of declaring variables in variables() and accessing them in step functions during dispatch."""

    # Keep track of accessed variable values inside the step
    accessed_values = []

    class OptimizePipelineExperiment(Experiment):
        def version(self) -> str:
            return "1.0-pipeline"

        def variables(self, v: VarNamespace, s) -> None:
            # Declare variables using precomputed and Optuna suggestion
            v.val1 = s.uniform(10.0, 20.0)
            v.val2 = s.precomputed("const_val")

        @step_method(first=True)
        def step_one(self, trial, **kwargs):
            # Access variables from trial
            accessed_values.append(trial.v.val1)
            accessed_values.append(trial.v.val2)
            return "done"

    study = optuna.create_study(direction="minimize")
    exp = OptimizePipelineExperiment.get_instance()
    sampler = OptimizingSampler(exp, study)
    dispatcher = LocalDispatcher.make_default(exp, tmp_path)

    # Dispatch 1 trial from the optimizer sampler
    trials = list(itertools.islice(sampler.get_trials(), 1))
    completed = list(dispatcher.schedule(trials))

    assert len(completed) == 1
    assert completed[0].is_finished()
    assert len(accessed_values) == 2
    # Verify val1 was suggested in [10.0, 20.0]
    assert 10.0 <= accessed_values[0] <= 20.0
    # Verify val2 is "const_val"
    assert accessed_values[1] == "const_val"


def test_sampling_strategy_pickles_successfully(tmp_path):
    """Ensure SamplingStrategy objects are disposed"""

    # Keep track of accessed variable values inside the step
    vns = []

    class OptimizePicklesSuccessfullyExperiment(Experiment):
        def version(self) -> str:
            return "1.0-pickles"

        def variables(self, v: VarNamespace, s) -> None:
            # Declare variables using precomputed and Optuna suggestion
            v.val1 = s.uniform(10.0, 20.0)
            v.val2 = s.precomputed("const_val")
            v.vec = s.list(s.uniform(-1.0, 1.0), 32)
            vns.append(v)

        @step_method(first=True)
        def step_one(self, trial, **kwargs):
            return "done"

    study = optuna.create_study(direction="minimize")
    exp = OptimizePicklesSuccessfullyExperiment.get_instance()
    sampler = OptimizingSampler(exp, study)
    dispatcher = LocalDispatcher.make_default(exp, tmp_path)

    trials = list(itertools.islice(sampler.get_trials(), 32))
    list(dispatcher.schedule(trials))
    for v in vns:
        pickle.dumps(v)
