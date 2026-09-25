from uuid import uuid4

import pytest

from rungrid.experiment import (
    Experiment,
    ExperimentRegistry,
    StaleTrial,
    Trial,
    VarNamespace,
    step_method,
)


class DummyExperiment(Experiment):
    """A dummy experiment class for testing."""

    def version(self) -> str:
        return "1.0-dummy"

    def variables(self, v: VarNamespace, s) -> None:
        v.param1 = s.precomputed(123)

    def first_step(self):
        return self.step_start

    @step_method(cache=True)
    def step_start(self, trial, **kwargs):
        return "step_start_result"


def test_experiment_registry_and_initialization():
    """Verify that custom Experiments register themselves and populate steps correctly."""
    registry = ExperimentRegistry.get_instance()

    # Instantiate DummyExperiment
    exp = DummyExperiment()

    # Should be registered under class name and identifier
    assert registry.get_by_name("DummyExperiment") is exp
    assert registry.get_by_ident(exp.identifier) is exp

    # Verify steps are correctly populated
    steps = exp.steps
    assert len(steps) == 1
    step_fqn = next(iter(steps.keys()))
    assert "DummyExperiment.step_start" in step_fqn

    step = steps[step_fqn]
    assert step.name == step_fqn
    assert step.cacheable is True


def test_trial_properties():
    """Verify StaleTrial and LiveTrial properties work correctly."""
    v = VarNamespace(set())
    trial = StaleTrial(10, uuid4(), v, tags={"test-tag"})

    assert trial.optuna_trial_number == 10
    assert "test-tag" in trial.tags
    assert trial.is_finished() is False
    assert trial.result is None

    # Add tag
    trial.add_tag("new-tag")
    assert "new-tag" in trial.tags

    # Add multiple tags
    trial.add_tags(["tag-a", "tag-b"])
    assert "tag-a" in trial.tags
    assert "tag-b" in trial.tags


def test_varnamespace_assign_sampling_strategy_raises_structure_error():
    """Verify assigning a SamplingStrategy directly to VarNamespace attribute does NOT raise ValueError (old behavior)."""
    import pytest
    from rungrid.experiment import SamplingStrategy

    v = VarNamespace(set())
    ss = SamplingStrategy("dummy", lambda name: 42, set())
    v.x = ss
    assert v.x == 42


def test_varnamespace_duplicate_sampling_raises_structure_error():
    """Verify sampling a variable name that is already in use in tracker raises StructureError."""
    import pytest
    from rungrid.error import StructureError
    from rungrid.experiment import SamplingStrategy

    names = {"x"}
    ss = SamplingStrategy("dummy", lambda name: 42, names)
    with pytest.raises(StructureError, match='variable name "x" is not unique'):
        ss.sample("x")


def test_varnamespace_set_variable_sampling_strategy_raises_value_error():
    """Verify direct _set_variable call with a SamplingStrategy does NOT raise ValueError (old behavior)."""
    import pytest
    from rungrid.experiment import SamplingStrategy

    v = VarNamespace(set())
    ss = SamplingStrategy("dummy", lambda name: 42, set())
    v._set_variable("x", ss)


def test_experiment_step_from_fn_unregistered_raises_lookup_error():
    """Verify resolving an unregistered/undecorated method in _step_from_fn raises LookupError."""
    import pytest

    class LookupErrorExperiment(Experiment):
        def version(self) -> str:
            return "lookup-err"

        def variables(self, v, s) -> None:
            pass

        def first_step(self):
            return self.step_none

        @step_method()
        def step_none(self, trial, **kwargs):
            return 42

    exp = LookupErrorExperiment()
    with pytest.raises(LookupError, match="was not recognized as an experiment step"):
        exp._step_from_fn(lambda: None)


def test_unallowed_step_method_call_raises_runtime_error():
    class WaywardCallExperiment(Experiment):
        def version(self) -> str:
            return "wayward-call"

        def variables(self, v, s) -> None:
            pass

        def first_step(self):
            return self.my_step

        @step_method()
        def my_step(self, trial, **kwargs):
            self.other_step(trial)
            assert isinstance(self, Experiment)
            assert isinstance(trial, Trial)
            return 43

        @step_method()
        def other_step(self, trial, **kwargs):
            assert isinstance(self, Experiment)
            assert isinstance(trial, Trial)
            return 44

    exp = WaywardCallExperiment()
    t = StaleTrial(1, uuid4(), VarNamespace(set()))
    m = "direct call"
    with pytest.raises(RuntimeError, match=m):
        exp.my_step(t)
    with pytest.raises(RuntimeError, match=m):
        exp.other_step(t)
    exp.allow_step_call()
    with pytest.raises(RuntimeError, match=m):
        exp.my_step(t)
    exp.allow_step_call()
    assert exp.other_step(t) == 44
