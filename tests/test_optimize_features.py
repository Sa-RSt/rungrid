"""Additional tests for ``OptimizingSampler``."""

import datetime
import itertools
from uuid import uuid4

import optuna

from rungrid.experiment import (
    Experiment,
    StaleTrial,
    TrialResult,
    VarNamespace,
    step_method,
)
from rungrid.optimize import OptimizingSampler


class OptimizeFeatureExperiment(Experiment):
    """Experiment with one Optuna-suggested variable."""

    def version(self) -> str:
        return "optimize-features-1"

    def variables(self, v: VarNamespace, s) -> None:
        v.x = s.integer(0, 10)

    @step_method(first=True)
    def step_one(self, trial, **kwargs):
        return "done"


class OptimizeFeatureConstantExperiment(Experiment):
    """Experiment with only constant variables."""

    def version(self) -> str:
        return "optimize-features-2"

    def variables(self, v: VarNamespace, s) -> None:
        v.x = s.precomputed(1)

    @step_method(first=True)
    def step_one(self, trial, **kwargs):
        return "done"


def test_optimizing_sampler_accepts_custom_sampler_callable():
    study = optuna.create_study(direction="minimize")
    exp = OptimizeFeatureConstantExperiment.get_instance()
    seen = []

    def custom_sampler(v, s):
        seen.append((v, s))
        v.custom = s.precomputed(123)

    sampler = OptimizingSampler(exp, study, sampler=custom_sampler)
    trial = next(iter(sampler.get_trials()))

    assert trial.v.custom == 123
    assert len(seen) == 1


def test_optimizing_sampler_get_trials_finished_returns_nothing():
    study = optuna.create_study(direction="minimize")
    exp = OptimizeFeatureConstantExperiment.get_instance()
    sampler = OptimizingSampler(exp, study)

    assert list(sampler.get_trials(finished=True)) == []
    assert list(sampler.get_trials(search_tag="tag")) == []
    assert list(sampler.get_trials(finished=False, search_tag="tag")) == []


def test_optimizing_sampler_get_trials_unfinished_works():
    study = optuna.create_study(direction="minimize")
    exp = OptimizeFeatureConstantExperiment.get_instance()
    sampler = OptimizingSampler(exp, study)

    trial = next(iter(sampler.get_trials(finished=False)))

    assert trial.v.x == 1


def test_optimizing_sampler_consume_and_tag_selects_matching_trial():
    study = optuna.create_study(direction="minimize")
    exp = OptimizeFeatureConstantExperiment.get_instance()
    sampler = OptimizingSampler(exp, study)

    trial = sampler.consume_and_tag("chosen", predicate=lambda trial: trial.v.x == 1)

    assert trial is not None
    assert "chosen" in trial.tags


def test_optimizing_sampler_consume_and_tag_with_finished_returns_none():
    study = optuna.create_study(direction="minimize")
    exp = OptimizeFeatureConstantExperiment.get_instance()
    sampler = OptimizingSampler(exp, study)

    assert sampler.consume_and_tag("chosen", finished=True) is None
    assert sampler.consume_and_tag("chosen", search_tag="tag") is None


def test_optimizing_sampler_put_trial_ignores_unfinished_trial():
    study = optuna.create_study(direction="minimize")
    exp = OptimizeFeatureConstantExperiment.get_instance()
    sampler = OptimizingSampler(exp, study)
    trial = StaleTrial(1, uuid4(), VarNamespace(set()))

    sampler.put_trial(trial)

    assert study.trials == []


def test_optimizing_sampler_put_trial_ignores_trial_without_optuna_number():
    study = optuna.create_study(direction="minimize")
    exp = OptimizeFeatureConstantExperiment.get_instance()
    sampler = OptimizingSampler(exp, study)
    trial = StaleTrial(None, uuid4(), VarNamespace(set())).with_result(
        TrialResult.make_ok(1, datetime.datetime.now())
    )

    sampler.put_trial(trial)

    assert study.trials == []


def test_optimizing_sampler_put_trial_reports_ok_and_error():
    study = optuna.create_study(direction="minimize")
    exp = OptimizeFeatureExperiment.get_instance()
    sampler = OptimizingSampler(exp, study)

    trial = next(iter(sampler.get_trials()))
    result = TrialResult.make_ok(3.5, datetime.datetime.now())
    sampler.put_trial(trial.with_result(result))

    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
    assert study.trials[0].value == 3.5

    trial2 = next(iter(sampler.get_trials()))
    error_result = TrialResult.make_error(ValueError("bad"), datetime.datetime.now())
    sampler.put_trial(trial2.with_result(error_result))

    assert study.trials[1].state == optuna.trial.TrialState.FAIL


def test_optimizing_sampler_generates_unique_uuids_per_trial():
    study = optuna.create_study(direction="minimize")
    exp = OptimizeFeatureConstantExperiment.get_instance()
    sampler = OptimizingSampler(exp, study)

    trials = list(itertools.islice(sampler.get_trials(), 5))

    assert len({t.uuid for t in trials}) == 5
    assert all(t.optuna_trial_number is None for t in trials)
