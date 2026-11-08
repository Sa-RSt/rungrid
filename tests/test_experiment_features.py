"""Additional tests for Experiment, StepRecord, Trial, VarNamespace and fixtures."""

import datetime
from typing import Annotated
from uuid import uuid4

import optuna
import pytest

from rungrid.error import StructureError
from rungrid.experiment import (
    STATE_ARG_RECORD,
    Experiment,
    ExperimentRegistry,
    ExperimentStep,
    LiveTrial,
    SignalDone,
    SignalNextStep,
    SignalPrune,
    StaleTrial,
    StepRecord,
    TrialResult,
    VarNamespace,
    fixture,
    step_method,
)


def make_step(
    name: str = "step",
    *,
    drop_args: set[str] | None = None,
    arg_names: set[str] | None = None,
    experiment_ident: str = "ident",
    experiment_name: str = "name",
) -> ExperimentStep:
    """Create a standalone ExperimentStep for Trial tests."""
    return ExperimentStep(
        name=name,
        cacheable=False,
        first=False,
        recover_from=[],
        drop_args=drop_args if drop_args is not None else set(),
        experiment_ident=experiment_ident,
        experiment_name=experiment_name,
        fn=lambda trial, **kwargs: None,
        arg_names=arg_names if arg_names is not None else set(),
    )


def make_variables(**values) -> VarNamespace:
    """Create a VarNamespace with preset values."""
    v = VarNamespace(set())
    v._variables_dict = dict(values)
    return v


class BasicExperiment(Experiment):
    """Experiment with two steps used by several tests."""

    def version(self) -> str:
        return "1.0-basic"

    def variables(self, v: VarNamespace, s) -> None:
        v.x = s.precomputed(1)

    @step_method(first=True)
    def step_one(self, trial):
        return "one"

    @step_method()
    def step_two(self, trial, keep):
        return "two"


class AnnotatedStepExperiment(Experiment):
    """Experiment with an Annotated state argument."""

    def version(self) -> str:
        return "1.0-annotated"

    def variables(self, v: VarNamespace, s) -> None:
        pass

    @step_method(first=True)
    def step_start(self, trial):
        return None

    @step_method()
    def step_annotated(
        self,
        trial,
        state: Annotated[dict, STATE_ARG_RECORD],
        normal: int,
    ):
        return state, normal


def test_experiment_metadata_and_registry():
    exp = BasicExperiment.get_instance()
    registry = ExperimentRegistry.get_instance()

    assert exp.name == "BasicExperiment"
    assert exp.identifier == "BasicExperiment1.0-basic"
    assert registry.get_by_name("BasicExperiment") is exp
    assert registry.get_by_ident(exp.identifier) is exp
    assert registry.get_by_name("does-not-exist") is None
    assert registry.get_by_ident("does-not-exist") is None


def test_get_instance_is_a_singleton():
    assert BasicExperiment.get_instance() is BasicExperiment.get_instance()


def test_experiment_requires_singleton_key():
    with pytest.raises(TypeError, match="Experiment subclasses are singleton"):
        BasicExperiment()


def test_experiment_registers_steps():
    exp = BasicExperiment.get_instance()

    assert len(exp.steps) == 2
    step_one = next(s for s in exp.steps.values() if s.name.endswith("step_one"))
    step_two = next(s for s in exp.steps.values() if s.name.endswith("step_two"))

    assert step_one.first is True
    assert step_two.first is False
    assert exp.get_first_step() is step_one
    assert step_one.experiment_name == "BasicExperiment"
    assert step_one.experiment_ident == exp.identifier


def test_experiment_with_zero_first_steps_raises():
    class ZeroFirstExperiment(Experiment):
        def version(self) -> str:
            return "zero"

        def variables(self, v, s) -> None:
            pass

        @step_method()
        def step_only(self, trial, **kwargs):
            return None

    with pytest.raises(StructureError, match="zero first steps"):
        ZeroFirstExperiment.get_instance()


def test_experiment_with_multiple_first_steps_raises():
    class TwoFirstExperiment(Experiment):
        def version(self) -> str:
            return "two"

        def variables(self, v, s) -> None:
            pass

        @step_method(first=True)
        def step_a(self, trial, **kwargs):
            return None

        @step_method(first=True)
        def step_b(self, trial, **kwargs):
            return None

    with pytest.raises(StructureError, match="more than one first step"):
        TwoFirstExperiment.get_instance()


def test_experiment_duplicate_name_raises():
    BasicExperiment.get_instance()

    with pytest.raises(
        StructureError, match='non-unique experiment name "BasicExperiment"'
    ):
        BasicExperiment(Experiment._Experiment__singleton)


def test_registry_auxiliary_data_and_thread_locals():
    exp = BasicExperiment.get_instance()
    registry = ExperimentRegistry.get_instance()

    aux = registry.get_auxiliary_data(exp)
    assert aux is registry.get_auxiliary_data(exp)

    tp = registry.get_thread_locals(exp)
    assert tp is registry.get_thread_locals(exp)
    tp.value = 123
    assert registry.get_thread_locals(exp).value == 123


def test_step_method_annotated_state_is_not_dropped():
    exp = AnnotatedStepExperiment.get_instance()
    step = next(s for s in exp.steps.values() if s.name.endswith("step_annotated"))

    assert "state" not in step.drop_args
    assert "normal" in step.drop_args
    assert "self" not in step.arg_names
    assert "trial" not in step.arg_names
    assert {"state", "normal"} <= step.arg_names


def test_step_trim_kwargs():
    step = make_step(arg_names={"a", "b"})

    trimmed = step.trim_kwargs({"a": 1, "b": 2, "c": 3, "self": 4, "trial": 5})

    assert trimmed == {"a": 1, "b": 2}


def test_step_record_creates_step_record():
    step = make_step(name="recorded")
    start = datetime.datetime.now()
    end = start + datetime.timedelta(seconds=1)

    record = step.record(start, end, next_step_name="next", a=1, b=2)

    assert isinstance(record, StepRecord)
    assert record.step is step
    assert record.next_step_name == "next"
    assert record.next_step_kwargs == {"a": 1, "b": 2}
    assert record.start == start
    assert record.end == end


def test_step_method_requires_self_and_trial():
    with pytest.raises(StructureError, match='argument called "self"'):

        @step_method()
        def missing_self(trial):
            return None

    with pytest.raises(StructureError, match='argument called "trial"'):

        @step_method()
        def missing_trial(self):
            return None


def test_step_method_warns_about_default_arguments():
    with pytest.warns(UserWarning, match="state argument foo has a default value"):

        class DefaultArgumentExperiment(Experiment):
            def version(self) -> str:
                return "default-arg"

            def variables(self, v, s) -> None:
                pass

            @step_method()
            def step_default(self, trial, foo=1):
                return foo


def test_experiment_exit_done_and_prune():
    exp = BasicExperiment.get_instance()

    with pytest.raises(SignalDone) as done:
        exp.exit_done(42)
    assert done.value.result == 42

    with pytest.raises(SignalPrune) as pruned:
        exp.exit_prune("not promising")
    assert pruned.value.reason == "not promising"


def test_exit_next_step_filters_kwargs():
    exp = BasicExperiment.get_instance()

    with pytest.raises(SignalNextStep) as next_step:
        exp.exit_next_step(exp.step_two, keep=1, drop=2, unknown=3)

    signal = next_step.value
    assert signal.next_step.name.endswith("step_two")
    assert signal.kwargs == {"keep": 1}


def test_exit_next_step_rejects_unregistered_callable():
    exp = BasicExperiment.get_instance()

    def not_a_step(trial):
        return None

    with pytest.raises(LookupError, match="was not recognized as an experiment step"):
        exp.exit_next_step(not_a_step)


def test_exit_next_step_fill_locals():
    exp = BasicExperiment.get_instance()

    class Caller:
        def call(self, trial, keep, drop):
            exp.exit_next_step(exp.step_two, fill_locals=True)

    with pytest.raises(SignalNextStep) as next_step:
        Caller().call("trial-value", keep="kept", drop="dropped")

    assert next_step.value.kwargs == {"keep": "kept"}


def test_exit_next_step_explicit_kwargs_override_locals():
    exp = BasicExperiment.get_instance()

    class Caller:
        def call(self, trial, keep):
            exp.exit_next_step(exp.step_two, fill_locals=True, keep="override")

    with pytest.raises(SignalNextStep) as next_step:
        Caller().call(None, keep="local")

    assert next_step.value.kwargs == {"keep": "override"}


def test_step_from_fn_accepts_known_step():
    exp = BasicExperiment.get_instance()

    assert (
        exp._step_from_fn(exp.step_two)
        is exp.steps[next(k for k in exp.steps if k.endswith("step_two"))]
    )


def test_fixture_caches_by_arguments():
    calls = []

    class FixtureExperiment(Experiment):
        def version(self) -> str:
            return "fixture-1"

        def variables(self, v, s) -> None:
            pass

        @step_method(first=True)
        def step_start(self, trial, **kwargs):
            return None

        @fixture()
        def expensive(self, value):
            calls.append(value)
            return value * 2

    exp = FixtureExperiment.get_instance()

    assert exp.expensive(3) == 6
    assert exp.expensive(3) == 6
    assert calls == [3]

    assert exp.expensive(4) == 8
    assert calls == [3, 4]


def test_fixture_respects_max_size():
    calls = []

    class SmallFixtureExperiment(Experiment):
        def version(self) -> str:
            return "fixture-2"

        def variables(self, v, s) -> None:
            pass

        @step_method(first=True)
        def step_start(self, trial, **kwargs):
            return None

        @fixture(max_size=1)
        def expensive(self, value):
            calls.append(value)
            return value

    exp = SmallFixtureExperiment.get_instance()

    exp.expensive(1)
    exp.expensive(2)
    exp.expensive(1)

    assert calls == [1, 2, 1]


def test_fixture_requires_self_argument():
    with pytest.raises(
        StructureError, match='fixtures must have an argument called "self"'
    ):

        @fixture()
        def no_self(value):
            return value


# ---------------------------------------------------------------------------
# VarNamespace
# ---------------------------------------------------------------------------


def test_varnamespace_attribute_and_item_access():
    v = VarNamespace(set())

    v.x = 1
    v["y"] = 2

    assert v.x == 1
    assert v["x"] == 1
    assert v.y == 2
    assert v["y"] == 2


def test_varnamespace_missing_attribute_and_key():
    v = VarNamespace(set())

    with pytest.raises(AttributeError, match="has no attribute 'missing'"):
        v.missing
    with pytest.raises(AttributeError, match="has no attribute '_missing'"):
        v._missing
    with pytest.raises(KeyError):
        v["missing"]


def test_varnamespace_private_assignment_uses_normal_attribute():
    v = VarNamespace(set())

    v._private = 123

    assert v._private == 123
    assert "_private" not in v._variables_dict


def test_varnamespace_raw_assignment_creates_precomputed_strategy():
    v = VarNamespace(set())

    v.x = 123

    assert v._variables_dict["x"] == 123
    assert v._sampling_strategies["x"].name == "precomputed(123)"


def test_varnamespace_strategy_assignment_samples_and_tracks():
    names: set[str] = set()
    from rungrid.experiment import SamplingStrategy

    v = VarNamespace(names)
    ss = SamplingStrategy("dummy", lambda name: 42, names, lambda name: name)

    v.x = ss

    assert v.x == 42
    assert v._sampling_strategies["x"] is ss
    assert "x" in names


def test_varnamespace_as_stale_copies_state():
    names = {"original"}
    v = VarNamespace(names)
    v.x = 1

    stale = v._as_stale()

    assert stale is not v
    assert stale._stale is True
    assert v._stale is False
    assert stale._variables_dict == v._variables_dict
    assert stale._variables_dict is not v._variables_dict
    assert stale._variable_names == v._variable_names
    assert stale._variable_names is not v._variable_names
    assert stale._sampling_strategies == v._sampling_strategies
    assert stale._sampling_strategies is not v._sampling_strategies


def test_varnamespace_uninitialized_attribute_access_raises():
    v = VarNamespace.__new__(VarNamespace)

    with pytest.raises(AttributeError, match="has no attribute 'x'"):
        v.x


# ---------------------------------------------------------------------------
# Trial
# ---------------------------------------------------------------------------


def test_trial_properties_and_tag_copies():
    trial = StaleTrial(7, uuid4(), make_variables(x=1), tags={"one"})

    assert trial.optuna_trial_number == 7
    assert trial.v.x == 1
    assert trial.tags == {"one"}

    tags = trial.tags
    tags.add("two")
    assert trial.tags == {"one"}

    trial.add_tag("two")
    trial.add_tags(["three", "four"])
    assert trial.tags == {"one", "two", "three", "four"}
    assert trial.is_finished() is False
    assert trial.result is None


def test_trial_step_records_returns_a_copy():
    trial = StaleTrial(1, uuid4(), make_variables())
    step = make_step()

    trial.record_step(step, datetime.datetime.now(), datetime.datetime.now())

    records = trial.step_records
    records.clear()
    assert len(trial.step_records) == 1


def test_trial_record_step_trims_previous_kwargs():
    trial = StaleTrial(1, uuid4(), make_variables())
    first = make_step(name="first", arg_names={"x", "y"})
    second = make_step(name="second", drop_args={"x"}, arg_names={"y"})
    now = datetime.datetime.now()

    trial.record_step(first, now, now, next_step_name="second", x=1, y=2)
    assert trial.step_records[0].next_step_kwargs == {"x": 1, "y": 2}

    trial.record_step(second, now, now)

    assert trial.step_records[0].next_step_kwargs == {"y": 2}


def test_trial_experiment_idents():
    trial = StaleTrial(1, uuid4(), make_variables())
    now = datetime.datetime.now()
    first = make_step(name="first", experiment_ident="ident-a")
    second = make_step(name="second", experiment_ident="ident-b")

    trial.record_step(first, now, now)
    trial.record_step(second, now, now)

    assert trial.experiment_idents == {"ident-a", "ident-b"}


def test_trial_as_stale_and_with_result():
    trial = StaleTrial(1, uuid4(), make_variables(x=1), tags={"tag"})
    trial.append_trace("loss", 0.5, epoch=1)
    result = TrialResult.make_ok(123, datetime.datetime.now())

    stale = trial.with_result(result)

    assert isinstance(stale, StaleTrial)
    assert stale is not trial
    assert stale.result is result
    assert stale.tags == {"tag"}
    assert stale.v.x == 1
    assert stale.v._stale is True
    assert list(stale.trace_values("loss")) == [0.5]
    assert trial.result is None


def test_trial_as_stale_preserves_existing_result():
    now = datetime.datetime.now()
    trial = StaleTrial(1, uuid4(), make_variables())
    result = TrialResult.make_ok(1, now)
    trial._result = result

    stale = trial.as_stale()

    assert stale.result is result


def test_trial_traces():
    trial = StaleTrial(1, uuid4(), make_variables())
    trial.append_trace("loss", 10, epoch=1)
    trial.append_trace("loss", 5, epoch=2)
    trial.append_trace("accuracy", 0.9)

    assert list(trial.trace_values("loss")) == [10, 5]
    assert list(trial.trace_all("loss")) == [
        (10, {"epoch": 1}),
        (5, {"epoch": 2}),
    ]
    assert list(trial.trace_items("loss", "epoch")) == [[10, 1], [5, 2]]
    assert list(trial.trace_items("loss", "epoch", "missing")) == [
        [10, 1, None],
        [5, 2, None],
    ]
    assert list(trial.trace_keys()) == ["loss", "accuracy"]


def test_trial_trace_missing_key_raises():
    trial = StaleTrial(1, uuid4(), make_variables())

    with pytest.raises(KeyError):
        list(trial.trace_values("missing"))


def test_trial_with_result_and_finished():
    trial = StaleTrial(1, uuid4(), make_variables())
    result = TrialResult.make_pruned("nope", datetime.datetime.now())

    finished = trial.with_result(result)

    assert finished.is_finished()
    assert finished.result is result
    assert not trial.is_finished()


def test_live_trial_wraps_optuna_trial():
    study = optuna.create_study(direction="minimize")
    optuna_trial = study.ask()
    uuid = uuid4()

    trial = LiveTrial(optuna_trial, uuid, make_variables(x=1))

    assert trial.optuna_trial is optuna_trial
    assert trial.uuid == uuid
    assert trial.optuna_trial_number == optuna_trial.number
    assert trial.v.x == 1


# ---------------------------------------------------------------------------
# First-step signature validation
# ---------------------------------------------------------------------------


def test_first_step_with_only_self_and_trial_is_accepted():
    exp = BasicExperiment.get_instance()
    step = exp.get_first_step()

    assert step.arg_names == set()


def test_first_step_allows_var_keyword_arguments():
    class KwargsFirstExperiment(Experiment):
        def version(self) -> str:
            return "first-kwargs"

        def variables(self, v, s) -> None:
            pass

        @step_method(first=True)
        def step_start(self, trial, **kwargs):
            return None

    assert KwargsFirstExperiment.get_instance().get_first_step() is not None


def test_first_step_rejects_extra_positional_argument():
    with pytest.raises(
        StructureError, match="first step must only accept .*self.* and .*trial"
    ):

        class ExtraArgumentExperiment(Experiment):
            def version(self) -> str:
                return "first-extra"

            def variables(self, v, s) -> None:
                pass

            @step_method(first=True)
            def step_start(self, trial, extra):
                return None


def test_first_step_rejects_extra_argument_with_default():
    with (
        pytest.warns(UserWarning),
        pytest.raises(StructureError, match="first step must only accept"),
    ):

        class DefaultArgumentFirstExperiment(Experiment):
            def version(self) -> str:
                return "first-default"

            def variables(self, v, s) -> None:
                pass

            @step_method(first=True)
            def step_start(self, trial, extra=1):
                return None


def test_first_step_rejects_keyword_only_argument():
    with (
        pytest.warns(UserWarning),
        pytest.raises(StructureError, match="first step must only accept"),
    ):

        class KeywordOnlyFirstExperiment(Experiment):
            def version(self) -> str:
                return "first-keyword-only"

            def variables(self, v, s) -> None:
                pass

            @step_method(first=True)
            def step_start(self, trial, *, extra=1):
                return None


def test_first_step_rejects_var_positional_argument():
    with pytest.raises(StructureError, match="first step must only accept"):

        class VarPositionalFirstExperiment(Experiment):
            def version(self) -> str:
                return "first-var-positional"

            def variables(self, v, s) -> None:
                pass

            @step_method(first=True)
            def step_start(self, trial, *args):
                return None


def test_non_first_step_may_still_declare_extra_arguments():
    exp = BasicExperiment.get_instance()
    step = next(s for s in exp.steps.values() if s.name.endswith("step_two"))

    assert step.arg_names == {"keep"}
