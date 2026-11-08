"""Additional tests for dispatch, step execution, caching and scheduling."""

import datetime
from uuid import uuid4

import pytest
from joblib import Memory

from rungrid.dispatch import (
    LocalDispatcher,
    _cached_step_execution,
    _experiment_from_step,
    _last_record,
    run_step,
)
from rungrid.error import StructureError
from rungrid.experiment import (
    Experiment,
    ExperimentStep,
    StaleTrial,
    StepRecord,
    VarNamespace,
    step_method,
)


def make_trial(**variables) -> StaleTrial:
    """Create a trial with preset variables."""
    v = VarNamespace(set())
    v._variables_dict = dict(variables)
    return StaleTrial(1, uuid4(), v)


def step_by_suffix(exp: Experiment, suffix: str) -> ExperimentStep:
    """Find a step by the suffix of its fully-qualified name."""
    return next(s for s in exp.steps.values() if s.name.endswith(suffix))


class DoneExperiment(Experiment):
    """Single-step experiment returning a computed value."""

    def version(self) -> str:
        return "dispatch-done-1"

    def variables(self, v: VarNamespace, s) -> None:
        pass

    @step_method(first=True)
    def step_done(self, trial, value):
        return value * 2


class ChainExperiment(Experiment):
    """Experiment with a transition from one step to another."""

    def version(self) -> str:
        return "dispatch-chain-1"

    def variables(self, v: VarNamespace, s) -> None:
        pass

    @step_method(first=True)
    def step_first(self, trial, value):
        self.exit_next_step(self.step_second, doubled=value * 2)

    @step_method()
    def step_second(self, trial, doubled):
        return doubled + 1


class AutoChainExperiment(Experiment):
    """Chain experiment whose first step needs no external arguments."""

    def version(self) -> str:
        return "dispatch-auto-chain-1"

    def variables(self, v: VarNamespace, s) -> None:
        pass

    @step_method(first=True)
    def step_auto_first(self, trial):
        self.exit_next_step(self.step_auto_second, doubled=10)

    @step_method()
    def step_auto_second(self, trial, doubled):
        return doubled + 1


class PruneExperiment(Experiment):
    """Experiment that prunes immediately."""

    def version(self) -> str:
        return "dispatch-prune-1"

    def variables(self, v: VarNamespace, s) -> None:
        pass

    @step_method(first=True)
    def step_prune(self, trial):
        self.exit_prune("not worth it")


class ErrorExperiment(Experiment):
    """Experiment that raises an exception."""

    def version(self) -> str:
        return "dispatch-error-1"

    def variables(self, v: VarNamespace, s) -> None:
        pass

    @step_method(first=True)
    def step_error(self, trial):
        raise ValueError("boom")


class KeyboardInterruptExperiment(Experiment):
    """Experiment that raises KeyboardInterrupt (recovered by default)."""

    def version(self) -> str:
        return "dispatch-keyboard-interrupt-1"

    def variables(self, v: VarNamespace, s) -> None:
        pass

    @step_method(first=True)
    def step_interrupt(self, trial):
        raise KeyboardInterrupt()


class SystemExitExperiment(Experiment):
    """Experiment that raises a non-recoverable BaseException."""

    def version(self) -> str:
        return "dispatch-system-exit-1"

    def variables(self, v: VarNamespace, s) -> None:
        pass

    @step_method(first=True)
    def step_system_exit(self, trial):
        raise SystemExit(3)


class RecoverExperiment(Experiment):
    """Experiment that can recover from ValueError."""

    def version(self) -> str:
        return "dispatch-recover-1"

    def variables(self, v: VarNamespace, s) -> None:
        pass

    @step_method(first=True, recover_from=[ValueError])
    def step_recover(self, trial, flag):
        if flag:
            raise ValueError("recover me")
        return "recovered"

    @step_method()
    def step_after_recovery(self, trial, value):
        return value


class CacheChainExperiment(Experiment):
    """Cacheable experiment that transitions to a cached-step result."""

    def version(self) -> str:
        return "dispatch-cache-chain-1"

    def variables(self, v: VarNamespace, s) -> None:
        pass

    @step_method(cache=True, first=True)
    def step_cached_first(self, trial, value):
        self.exit_next_step(self.step_cached_second, result=value + 10)

    @step_method()
    def step_cached_second(self, trial, result):
        return result


class CacheErrorExperiment(Experiment):
    """Cacheable experiment that raises."""

    def version(self) -> str:
        return "dispatch-cache-error-1"

    def variables(self, v: VarNamespace, s) -> None:
        pass

    @step_method(cache=True, first=True)
    def step_cached_error(self, trial):
        raise ValueError("cached boom")


class CachePruneExperiment(Experiment):
    """Cacheable experiment that prunes."""

    def version(self) -> str:
        return "dispatch-cache-prune-1"

    def variables(self, v: VarNamespace, s) -> None:
        pass

    @step_method(cache=True, first=True)
    def step_cached_prune(self, trial):
        self.exit_prune("cached prune reason")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_last_record():
    step = step_by_suffix(DoneExperiment.get_instance(), "step_done")
    now = datetime.datetime.now()
    first = StepRecord(step, None, None, now, now)
    second = StepRecord(step, None, None, now, now)

    assert _last_record([]) is None
    assert _last_record([first]) is first
    assert _last_record([first, second]) is second


def test_experiment_from_step_returns_registered_experiment():
    exp = DoneExperiment.get_instance()
    step = step_by_suffix(exp, "step_done")

    assert _experiment_from_step(step) is exp


def test_experiment_from_step_rejects_foreign_step():
    exp = DoneExperiment.get_instance()
    fake_step = ExperimentStep(
        name="fake.step",
        cacheable=False,
        first=False,
        recover_from=[],
        drop_args=set(),
        experiment_ident=exp.identifier,
        experiment_name=exp.name,
        fn=lambda trial, **kwargs: None,
        arg_names=set(),
    )

    with pytest.raises(StructureError, match="is not contained in"):
        _experiment_from_step(fake_step)


# ---------------------------------------------------------------------------
# _cached_step_execution
# ---------------------------------------------------------------------------


def test_cached_step_execution_done():
    exp = DoneExperiment.get_instance()
    step = step_by_suffix(exp, "step_done")

    outcome = _cached_step_execution(step.name, (), {"value": 7})

    assert outcome == ("done", 14)


def test_cached_step_execution_next_step():
    exp = ChainExperiment.get_instance()
    step = step_by_suffix(exp, "step_first")
    next_step = step_by_suffix(exp, "step_second")

    outcome = _cached_step_execution(step.name, (), {"value": 3})

    assert outcome == ("next_step", (next_step.name, {"doubled": 6}))


def test_cached_step_execution_prune():
    exp = PruneExperiment.get_instance()
    step = step_by_suffix(exp, "step_prune")

    outcome = _cached_step_execution(step.name, (), {})

    assert outcome == ("prune", "not worth it")


def test_cached_step_execution_error():
    exp = ErrorExperiment.get_instance()
    step = step_by_suffix(exp, "step_error")

    outcome = _cached_step_execution(step.name, (), {})

    assert outcome[0] == "error"
    assert isinstance(outcome[1], ValueError)
    assert str(outcome[1]) == "boom"


def test_cached_step_execution_reconstructs_trial_variables():
    """The cached function must expose variables to the step as a VarNamespace."""

    received = []

    class VariableReadingExperiment(Experiment):
        def version(self) -> str:
            return "dispatch-cached-vars-1"

        def variables(self, v: VarNamespace, s) -> None:
            pass

        @step_method(first=True)
        def step_read(self, trial, **kwargs):
            received.append(trial.v.x)
            return trial.v.x

    exp = VariableReadingExperiment.get_instance()
    step = step_by_suffix(exp, "step_read")

    outcome = _cached_step_execution(step.name, (("x", 99),), {})

    assert outcome == ("done", 99)
    assert received == [99]


# ---------------------------------------------------------------------------
# run_step
# ---------------------------------------------------------------------------


def test_run_step_done_records_result():
    exp = DoneExperiment.get_instance()
    step = step_by_suffix(exp, "step_done")
    trial = make_trial()

    returned, next_step, kwargs = run_step(step, trial, value=4)

    assert next_step is None
    assert kwargs is None
    assert returned.is_finished()
    assert returned.result is not None
    assert returned.result.has_result()
    assert returned.result.result == 8
    assert len(returned.step_records) == 1
    assert returned.step_records[0].step is step
    assert returned.step_records[0].next_step_name is None


def test_run_step_transitions_to_next_step():
    exp = ChainExperiment.get_instance()
    first = step_by_suffix(exp, "step_first")
    second = step_by_suffix(exp, "step_second")
    trial = make_trial()

    returned, next_step, kwargs = run_step(first, trial, value=5)

    assert returned is trial
    assert not returned.is_finished()
    assert next_step is second
    assert kwargs == {"doubled": 10}
    assert len(returned.step_records) == 1
    assert returned.step_records[0].next_step_name == second.name
    assert returned.step_records[0].next_step_kwargs == {"doubled": 10}


def test_run_step_chain_completes():
    exp = ChainExperiment.get_instance()
    first = step_by_suffix(exp, "step_first")
    second = step_by_suffix(exp, "step_second")
    trial = make_trial()

    trial, next_step, kwargs = run_step(first, trial, value=5)
    assert next_step is second
    trial, next_step, kwargs = run_step(second, trial, **kwargs)

    assert next_step is None
    assert kwargs is None
    assert trial.result is not None
    assert trial.result.result == 11
    assert len(trial.step_records) == 2


def test_run_step_prune():
    exp = PruneExperiment.get_instance()
    step = step_by_suffix(exp, "step_prune")
    trial = make_trial()

    returned, next_step, kwargs = run_step(step, trial)

    assert next_step is None
    assert kwargs is None
    assert returned.is_finished()
    assert returned.result is not None
    assert returned.result.is_pruned
    assert returned.result.prune_reason == "not worth it"


def test_run_step_error():
    exp = ErrorExperiment.get_instance()
    step = step_by_suffix(exp, "step_error")
    trial = make_trial()

    returned, next_step, kwargs = run_step(step, trial)

    assert next_step is None
    assert kwargs is None
    assert returned.is_finished()
    assert returned.result is not None
    assert returned.result.is_error
    assert isinstance(returned.result.error, ValueError)


def test_run_step_recovers_without_previous_record():
    exp = RecoverExperiment.get_instance()
    step = step_by_suffix(exp, "step_recover")
    trial = make_trial()

    returned, next_step, kwargs = run_step(step, trial, flag=True)

    assert returned is trial
    assert next_step is step
    assert kwargs == {"flag": True}
    assert not returned.is_finished()
    assert returned.step_records == []


def test_run_step_recovers_from_keyboard_interrupt_by_default():
    exp = KeyboardInterruptExperiment.get_instance()
    step = step_by_suffix(exp, "step_interrupt")
    trial = make_trial()

    returned, next_step, kwargs = run_step(step, trial)

    assert returned is trial
    assert next_step is step
    assert not returned.is_finished()
    assert returned.step_records == []


def test_run_step_reraises_non_recoverable_base_exception():
    exp = SystemExitExperiment.get_instance()
    step = step_by_suffix(exp, "step_system_exit")
    trial = make_trial()

    with pytest.raises(SystemExit):
        run_step(step, trial)


def test_run_step_recovers_to_recorded_next_step():
    exp = RecoverExperiment.get_instance()
    step = step_by_suffix(exp, "step_recover")
    next_step = step_by_suffix(exp, "step_after_recovery")
    now = datetime.datetime.now()
    record = StepRecord(
        step=next_step,
        next_step_name=next_step.name,
        next_step_kwargs={"value": "from-record"},
        start=now,
        end=now,
    )
    trial = StaleTrial(1, uuid4(), VarNamespace(set()), step_records=[record])

    returned, returned_next, kwargs = run_step(step, trial, flag=True)

    assert returned is trial
    assert returned_next is next_step
    assert kwargs == {"value": "from-record"}


def test_run_step_cacheable_done(tmp_path):
    exp = DoneExperiment.get_instance()
    step = step_by_suffix(exp, "step_done")
    mem = Memory(location=tmp_path, verbose=0)
    trial = make_trial()

    # A cacheable step without cache_provider still executes normally.
    returned, next_step, kwargs = run_step(step, trial, value=2)

    assert returned.result is not None
    assert returned.result.result == 4


def test_run_step_cacheable_next_step(tmp_path):
    exp = CacheChainExperiment.get_instance()
    first = step_by_suffix(exp, "step_cached_first")
    second = step_by_suffix(exp, "step_cached_second")
    mem = Memory(location=tmp_path, verbose=0)
    trial = make_trial()

    returned, next_step, kwargs = run_step(first, trial, mem, value=1)

    assert next_step is second
    assert kwargs == {"result": 11}
    assert not returned.is_finished()


def test_run_step_cacheable_error(tmp_path):
    exp = CacheErrorExperiment.get_instance()
    step = step_by_suffix(exp, "step_cached_error")
    mem = Memory(location=tmp_path, verbose=0)
    trial = make_trial()

    returned, next_step, kwargs = run_step(step, trial, mem)

    assert next_step is None
    assert kwargs is None
    assert returned.result is not None
    assert returned.result.is_error
    assert isinstance(returned.result.error, ValueError)
    assert str(returned.result.error) == "cached boom"


def test_run_step_cacheable_prune(tmp_path):
    exp = CachePruneExperiment.get_instance()
    step = step_by_suffix(exp, "step_cached_prune")
    mem = Memory(location=tmp_path, verbose=0)
    trial = make_trial()

    returned, next_step, kwargs = run_step(step, trial, mem)

    assert next_step is None
    assert kwargs is None
    assert returned.result is not None
    assert returned.result.is_pruned
    assert returned.result.prune_reason == "cached prune reason"


# ---------------------------------------------------------------------------
# LocalDispatcher
# ---------------------------------------------------------------------------


def test_local_dispatcher_get_experiment():
    exp = DoneExperiment.get_instance()
    dispatcher = LocalDispatcher.make_default(exp, "/tmp/rungrid-dispatch-test")

    assert dispatcher.get_experiment() is exp


def test_recover_last_step_kwargs_empty_and_result_records():
    exp = ChainExperiment.get_instance()
    dispatcher = LocalDispatcher.make_default(exp, "/tmp/rungrid-dispatch-test")

    assert dispatcher._recover_last_step_kwargs(make_trial()) is None

    now = datetime.datetime.now()
    step = step_by_suffix(exp, "step_first")
    finished_record = StepRecord(step, None, None, now, now)
    assert (
        dispatcher._recover_last_step_kwargs(
            StaleTrial(1, uuid4(), VarNamespace(set()), [finished_record])
        )
        is None
    )

    no_kwargs_record = StepRecord(step, step.name, None, now, now)
    assert (
        dispatcher._recover_last_step_kwargs(
            StaleTrial(1, uuid4(), VarNamespace(set()), [no_kwargs_record])
        )
        is None
    )


def test_recover_last_step_kwargs_returns_step_and_copy():
    exp = ChainExperiment.get_instance()
    dispatcher = LocalDispatcher.make_default(exp, "/tmp/rungrid-dispatch-test")
    step = step_by_suffix(exp, "step_second")
    now = datetime.datetime.now()
    record = StepRecord(step, step.name, {"value": 1}, now, now)
    trial = StaleTrial(1, uuid4(), VarNamespace(set()), [record])

    recovered = dispatcher._recover_last_step_kwargs(trial)

    assert recovered is not None
    recovered_step, kwargs = recovered
    assert recovered_step is step
    assert kwargs == {"value": 1}
    kwargs["value"] = 999
    assert record.next_step_kwargs == {"value": 1}


def test_dispatcher_run_executes_chain(tmp_path):
    exp = AutoChainExperiment.get_instance()
    dispatcher = LocalDispatcher.make_default(exp, tmp_path)
    trial = make_trial()

    results = list(dispatcher.run(trial))

    assert len(results) == 2
    assert results[0] is trial
    assert results[-1].is_finished()
    assert results[-1].result is not None
    assert results[-1].result.result == 11
    assert len(results[-1].step_records) == 2


def test_dispatcher_run_resumes_from_recovered_record(tmp_path):
    exp = AutoChainExperiment.get_instance()
    dispatcher = LocalDispatcher.make_default(exp, tmp_path)
    second = step_by_suffix(exp, "step_auto_second")
    now = datetime.datetime.now()
    record = StepRecord(second, second.name, {"doubled": 20}, now, now)
    trial = StaleTrial(1, uuid4(), VarNamespace(set()), [record])

    results = list(dispatcher.run(trial))

    assert len(results) == 1
    assert results[0].is_finished()
    assert results[0].result is not None
    assert results[0].result.result == 21
    # The pre-existing record plus the newly executed step.
    assert len(results[0].step_records) == 2


def test_dispatcher_run_skips_finished_trial(tmp_path):
    exp = DoneExperiment.get_instance()
    dispatcher = LocalDispatcher.make_default(exp, tmp_path)
    from rungrid.experiment import TrialResult

    trial = make_trial().with_result(
        TrialResult.make_ok("already done", datetime.datetime.now())
    )

    assert list(dispatcher.run(trial)) == []


def test_dispatcher_schedule_runs_all_trials(tmp_path):
    exp = AutoChainExperiment.get_instance()
    dispatcher = LocalDispatcher.make_default(exp, tmp_path)
    trials = [make_trial(), make_trial()]

    completed = list(dispatcher.schedule(trials))

    # The chain has two steps, so each trial is yielded after each step.
    assert len(completed) == 4
    finished = [t for t in completed if t.is_finished()]
    assert len(finished) == 2
    assert all(t.result is not None for t in finished)
    assert all(t.result.result == 11 for t in finished)


def test_dispatcher_schedule_skips_finished_trials(tmp_path):
    from rungrid.experiment import TrialResult

    exp = DoneExperiment.get_instance()
    dispatcher = LocalDispatcher.make_default(exp, tmp_path)
    finished = make_trial().with_result(
        TrialResult.make_ok("done", datetime.datetime.now())
    )

    assert list(dispatcher.schedule([finished])) == []
