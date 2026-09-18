import datetime

import optuna

from rungrid.experiment import (
    Experiment,
    TrialResult,
    VarNamespace,
    step_method,
)
from rungrid.optimize import OptimizingSampler


class OptimizeDummyExperiment(Experiment):
    """A dummy experiment class for testing."""

    def version(self) -> str:
        return "1.0-dummy"

    def variables(self, v: VarNamespace, s) -> None:
        v.sampled("param1", s.precomputed(123))

    def first_step(self):
        return self.step_start

    @step_method(cache=True)
    def step_start(self, trial, **kwargs):
        return "step_start_result"


def test_optimizing_sampler():
    """Verify that OptimizingSampler yields trials from Optuna and reports back results."""
    # Create an in-memory Optuna study
    study = optuna.create_study(direction="minimize")
    exp = OptimizeDummyExperiment()

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
    import itertools
    import optuna
    from rungrid.optimize import OptimizingSampler
    from rungrid.dispatch import LocalDispatcher

    # Keep track of accessed variable values inside the step
    accessed_values = []

    class OptimizePipelineExperiment(Experiment):
        def version(self) -> str:
            return "1.0-pipeline"

        def variables(self, v: VarNamespace, s) -> None:
            # Declare variables using precomputed and Optuna suggestion
            v.sampled("val1", s.uniform(10.0, 20.0))
            v.sampled("val2", s.precomputed("const_val"))

        def first_step(self):
            return self.step_one

        @step_method()
        def step_one(self, trial, **kwargs):
            # Access variables from trial
            accessed_values.append(trial.v.val1)
            accessed_values.append(trial.v.val2)
            return "done"

    study = optuna.create_study(direction="minimize")
    exp = OptimizePipelineExperiment()
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
