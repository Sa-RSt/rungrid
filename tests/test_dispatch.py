from uuid import uuid4

from rungrid.dispatch import LocalDispatcher, run_step
from rungrid.experiment import (
    Experiment,
    StaleTrial,
    VarNamespace,
    step_method,
)

# Keep track of execution counts globally/statically for the test
EXECUTION_COUNTS = {
    "cacheable": 0,
    "non_cacheable": 0,
}


class CachingExperiment(Experiment):
    """An experiment designed to verify step execution caching behavior."""

    def version(self) -> str:
        return "1.0-cache-test"

    def variables(self, v: VarNamespace, s) -> None:
        v.p = 42

    def first_step(self):
        return self.step_cacheable

    @step_method(cache=True)
    def step_cacheable(self, trial, **kwargs):
        EXECUTION_COUNTS["cacheable"] += 1
        return f"cacheable-result-{kwargs.get('arg', '')}"

    @step_method(cache=False)
    def step_non_cacheable(self, trial, **kwargs):
        EXECUTION_COUNTS["non_cacheable"] += 1
        return f"non-cacheable-result-{kwargs.get('arg', '')}"


def test_local_dispatcher_caching(tmp_path):
    """Verify that LocalDispatcher caches cacheable steps but not non-cacheable steps.

    Ensures that a cacheable step is not called twice with the same arguments,
    while a non-cacheable step is.
    """
    exp = CachingExperiment()
    dispatcher = LocalDispatcher.make_default(exp, tmp_path)

    v = VarNamespace(set())
    # Set the same variable value to represent identical trial configurations
    v.p = 42

    trial1 = StaleTrial(1, uuid4(), v)
    trial2 = StaleTrial(2, uuid4(), v)

    # 1. Test Cacheable Step
    step_cacheable = exp.steps[next(k for k in exp.steps if "step_cacheable" in k)]

    # Run once
    run_step(step_cacheable, trial1, dispatcher._mem, arg="same-val")
    assert EXECUTION_COUNTS["cacheable"] == 1

    # Run again with same arguments
    run_step(step_cacheable, trial2, dispatcher._mem, arg="same-val")
    # Call count should STILL be 1 because it loaded from cache!
    assert EXECUTION_COUNTS["cacheable"] == 1

    # Run with different argument
    trial3 = StaleTrial(3, uuid4(), v)
    run_step(step_cacheable, trial3, dispatcher._mem, arg="diff-val")
    assert EXECUTION_COUNTS["cacheable"] == 2

    # 2. Test Non-Cacheable Step
    step_non_cacheable = exp.steps[
        next(k for k in exp.steps if "step_non_cacheable" in k)
    ]

    # Run once
    run_step(step_non_cacheable, trial1, dispatcher._mem, arg="same-val")
    assert EXECUTION_COUNTS["non_cacheable"] == 1

    # Run again with same arguments
    run_step(step_non_cacheable, trial2, dispatcher._mem, arg="same-val")
    # Call count should be 2 because caching is disabled!
    assert EXECUTION_COUNTS["non_cacheable"] == 2


def test_dispatch_structure_errors():
    """Verify dispatch structure errors are raised when step or experiment is unregistered."""
    import pytest
    from rungrid.error import StructureError
    from rungrid.experiment import ExperimentStep
    from rungrid.dispatch import _experiment_from_step, _cached_step_execution

    dummy_step = ExperimentStep(
        name="dummy_nonexistent",
        cacheable=True,
        recover_from=[],
        drop_args=set(),
        experiment_ident="nonexistent-ident",
        experiment_name="NonExistentExperiment",
        fn=lambda trial, **kwargs: 42,
    )

    with pytest.raises(StructureError, match="experiment not loaded"):
        _experiment_from_step(dummy_step)

    with pytest.raises(StructureError, match="not found in any registered experiments"):
        _cached_step_execution("NonExistentExperiment.nonexistent_step", (), {})

