"""Unit tests for `Sampler.grid`."""

import itertools
import random
from uuid import uuid4

import optuna
import pytest

from rungrid.error import StructureError
from rungrid.experiment import (
    Experiment,
    Sampler,
    SamplingStrategy,
    VarNamespace,
    step_method,
)
from rungrid.optimize import OptimizingSampler


def make_study(**kwargs) -> optuna.Study:
    """Create a fresh in-memory Optuna study."""
    return optuna.create_study(direction="minimize", **kwargs)


def as_declaration(variables):
    """Turn a dict of ``{name: choices}`` into a ``variables()``-like callable."""
    if callable(variables):
        return variables

    def declare(v: VarNamespace, s: Sampler) -> None:
        for name, strategies in variables.items():
            setattr(v, name, s.grid(strategies))

    return declare


def sample_trial(study: optuna.Study, variables, *, trial_id=None) -> VarNamespace:
    """Run one full sampling round using fresh grid strategies."""
    s = Sampler(trial_id if trial_id is not None else uuid4(), study)
    v = s._make_var_namespace()
    as_declaration(variables)(v, s)
    s.dispose()
    return v


def sample_trials(study: optuna.Study, variables, n: int) -> list[VarNamespace]:
    """Run ``n`` sampling rounds."""
    return [sample_trial(study, variables) for _ in range(n)]


def variable_values(trials: list[VarNamespace], name: str) -> list:
    """Extract the value of one variable from each trial."""
    return [getattr(v, name) for v in trials]


def pairs(trials: list[VarNamespace], *names: str) -> list[tuple]:
    """Extract a tuple of variables from each trial."""
    return [tuple(getattr(v, name) for name in names) for v in trials]


def grid_state(study: optuna.Study) -> dict:
    """Return the persisted grid state."""
    return study.user_attrs["rungrid_grid"]


def test_grid_returns_sampling_strategy_with_expected_name():
    s = Sampler(uuid4(), make_study())

    ss = s.grid([1, 2, 3])

    assert isinstance(ss, SamplingStrategy)
    assert ss.name == "grid(precomputed(1)|precomputed(2)|precomputed(3))"


def test_grid_name_uses_sampling_strategy_names():
    s = Sampler(uuid4(), make_study())

    ss = s.grid([s.precomputed("a"), s.integer(0, 5)])

    assert ss.name == "grid(precomputed(a)|integer(0,5))"


def test_grid_accepts_a_string_as_a_sequence_of_characters():
    s = Sampler(uuid4(), make_study())

    assert s.grid("abc").name == "grid(precomputed(a)|precomputed(b)|precomputed(c))"


def test_grid_accepts_sampling_strategy_values():
    s = Sampler(uuid4(), make_study())

    # SamplingStrategy objects are fine as *choices*; only the outer sequence
    # itself must not be a SamplingStrategy.
    ss = s.grid([s.precomputed(1), 2])

    assert isinstance(ss, SamplingStrategy)


def test_grid_cycles_through_constant_choices():
    study = make_study()

    trials = sample_trials(study, {"x": [10, 20, 30]}, 7)

    assert variable_values(trials, "x") == [10, 20, 30, 10, 20, 30, 10]


def test_grid_iterates_strings_character_by_character():
    study = make_study()

    trials = sample_trials(study, {"x": "abc"}, 4)

    assert variable_values(trials, "x") == ["a", "b", "c", "a"]


def test_grid_single_choice_always_returns_that_choice():
    study = make_study()

    trials = sample_trials(study, {"x": [42]}, 4)

    assert variable_values(trials, "x") == [42, 42, 42, 42]
    assert grid_state(study)["variables"] == [
        {"name": "x", "choices": 1, "next_choice": 0}
    ]


def test_grid_preserves_arbitrary_value_types():
    choices = [None, True, (1, 2), {"a": 1}, 3.5]
    study = make_study()

    trials = sample_trials(study, {"x": choices}, len(choices) + 1)

    assert variable_values(trials, "x") == choices + [choices[0]]


def test_grid_enumerates_two_variable_cartesian_product():
    study = make_study()

    trials = sample_trials(study, {"a": [0, 1], "b": ["x", "y"]}, 5)

    # The first variable is the fastest-changing one.
    expected = [(a, b) for b in ["x", "y"] for a in [0, 1]]
    assert pairs(trials, "a", "b") == expected + [expected[0]]


def test_grid_enumerates_three_variable_cartesian_product():
    study = make_study()

    trials = sample_trials(
        study,
        {"a": [0, 1], "b": ["x", "y"], "c": [True, False]},
        8,
    )

    expected = [(a, b, c) for c in [True, False] for b in ["x", "y"] for a in [0, 1]]
    assert pairs(trials, "a", "b", "c") == expected


def test_grid_enumerates_mixed_radices():
    study = make_study()

    trials = sample_trials(study, {"a": [0, 1], "b": [0, 1, 2]}, 6)

    expected = [(a, b) for b in range(3) for a in range(2)]
    assert pairs(trials, "a", "b") == expected


def test_grid_large_product_is_a_permutation_and_then_repeats():
    study = make_study()

    trials = sample_trials(
        study,
        {"a": [0, 1, 2], "b": ["x", "y"], "c": [10, 20, 30, 40]},
        48,
    )

    combinations = pairs(trials, "a", "b", "c")
    first_cycle = combinations[:24]
    second_cycle = combinations[24:]

    assert len(set(first_cycle)) == 24
    assert set(first_cycle) == set(
        itertools.product([0, 1, 2], ["x", "y"], [10, 20, 30, 40])
    )
    assert second_cycle == first_cycle


def test_grid_wraps_all_counters_after_a_full_product():
    study = make_study()

    sample_trials(study, {"a": [0, 1], "b": [0, 1]}, 5)

    assert grid_state(study)["variables"] == [
        {"name": "a", "choices": 2, "next_choice": 0},
        {"name": "b", "choices": 2, "next_choice": 0},
    ]


def test_grid_with_precomputed_strategies():
    study = make_study()

    def declare(v, s):
        v.x = s.grid([s.precomputed(5), 6])

    trials = [sample_trial(study, declare) for _ in range(4)]

    assert variable_values(trials, "x") == [5, 6, 5, 6]


def test_grid_with_transformed_strategy():

    def plus_ten(x):
        return x + 10

    study = make_study()

    def declare(v, s):
        v.x = s.grid([s.transformed(1, plus_ten, "plus_ten"), 99])

    trials = [sample_trial(study, declare) for _ in range(4)]

    assert variable_values(trials, "x") == [11, 99, 11, 99]


def test_grid_with_random_strategy():
    rng = random.Random(1234)
    reference_rng = random.Random(1234)
    study = make_study()

    def declare(v, s):
        v.x = s.grid([s.random([1, 2, 3], rng=rng), 99])

    trials = [sample_trial(study, declare) for _ in range(6)]

    expected = []
    for i in range(6):
        if i % 2 == 0:
            expected.append([1, 2, 3][reference_rng.randrange(3)])
        else:
            expected.append(99)
    assert variable_values(trials, "x") == expected


def test_grid_does_not_ask_optuna_for_precomputed_choices():
    study = make_study()
    s = Sampler(uuid4(), study)
    v = s._make_var_namespace()

    v.x = s.grid([1, 2, 3])

    assert not s.has_asked_optuna()
    assert study.trials == []
    s.dispose()


def test_grid_only_asks_optuna_for_the_selected_choice():
    study = make_study()

    def declare(v, s):
        v.x = s.grid([s.integer(0, 5), 9])

    s1 = Sampler(uuid4(), study)
    v1 = s1._make_var_namespace()
    declare(v1, s1)
    assert s1.has_asked_optuna()
    assert 0 <= v1.x <= 5
    assert "x_grid0" in study.trials[0].params
    s1.dispose()

    s2 = Sampler(uuid4(), study)
    v2 = s2._make_var_namespace()
    declare(v2, s2)
    assert not s2.has_asked_optuna()
    assert v2.x == 9
    # The second choice is a precomputed value, so no new Optuna trial is asked.
    assert len(study.trials) == 1
    s2.dispose()


def test_grid_selects_optuna_strategy_in_later_cycle():
    study = make_study()

    def declare(v, s):
        v.x = s.grid([9, s.integer(10, 20)])

    s1 = Sampler(uuid4(), study)
    v1 = s1._make_var_namespace()
    declare(v1, s1)
    assert not s1.has_asked_optuna()
    assert v1.x == 9
    s1.dispose()

    s2 = Sampler(uuid4(), study)
    v2 = s2._make_var_namespace()
    declare(v2, s2)
    assert s2.has_asked_optuna()
    assert 10 <= v2.x <= 20
    assert "x_grid1" in study.trials[0].params
    s2.dispose()


def test_grid_is_unaffected_by_branch_for_variable_names():
    study = make_study()

    def declare(v, s):
        with s.branch("architecture"):
            v.x = s.grid([1, 2])

    trials = sample_trials(study, declare, 5)

    assert variable_values(trials, "x") == [1, 2, 1, 2, 1]
    assert grid_state(study)["variables"][0]["name"] == "x"


def test_grid_is_unaffected_by_nested_branches():
    study = make_study()

    def declare(v, s):
        with s.branch("outer"):
            with s.branch("inner"):
                v.x = s.grid([1, 2])

    trials = sample_trials(study, declare, 5)

    assert variable_values(trials, "x") == [1, 2, 1, 2, 1]
    assert grid_state(study)["variables"][0]["name"] == "x"


def test_grid_ignores_branch_for_its_own_variable_names():
    study = make_study()

    def declare(v, s):
        with s.branch("architecture"):
            v.x = s.grid([s.integer(0, 5)])

    s = Sampler(uuid4(), study)
    v = s._make_var_namespace()
    declare(v, s)
    s.dispose()

    # The grid itself is not branched, so the variable recorded in the grid
    # state is plain "x". Strategies *inside* the grid still apply their own
    # branch transform (that is the branch's job), hence the parameter name.
    assert grid_state(study)["variables"][0]["name"] == "x"
    assert any(name.endswith("x_grid0") for name in study.trials[0].params)


def test_grid_state_is_persisted_across_samplers():
    study = make_study()
    first_id = uuid4()
    second_id = uuid4()

    sample_trial(study, {"x": [10, 20, 30]}, trial_id=first_id)

    state = grid_state(study)
    assert state["variables"] == [{"name": "x", "choices": 3, "next_choice": 0}]
    assert state["last_updated_by"] == str(first_id)
    assert state["updated_names"] == ["x"]

    sample_trial(study, {"x": [10, 20, 30]}, trial_id=second_id)

    state = grid_state(study)
    assert state["variables"] == [{"name": "x", "choices": 3, "next_choice": 1}]
    assert state["last_updated_by"] == str(second_id)
    assert state["updated_names"] == ["x"]


def test_grid_state_contains_variables_in_declaration_order():
    study = make_study()

    sample_trial(study, {"b": [1, 2], "a": ["x", "y"]})

    assert [v["name"] for v in grid_state(study)["variables"]] == ["b", "a"]


def test_grid_counter_advances_once_per_trial_not_per_variable():
    study = make_study()

    sample_trial(study, {"a": [0, 1], "b": [0, 1], "c": [0, 1]})

    # All variables are sampled at the initial all-zero combination.
    assert [v["next_choice"] for v in grid_state(study)["variables"]] == [0, 0, 0]


def test_grid_state_roundtrips_through_persistent_storage(tmp_path):
    study = make_study(
        storage=f"sqlite:///{tmp_path / 'study.db'}",
        study_name="grid-storage",
    )

    trials = sample_trials(study, {"a": [0, 1], "b": ["x", "y"]}, 5)

    expected = [(a, b) for b in ["x", "y"] for a in [0, 1]]
    assert pairs(trials, "a", "b") == expected + [expected[0]]


def test_grid_declaration_order_does_not_change_enumeration():
    study = make_study()

    def declare_a_then_b(v, s):
        v.a = s.grid([0, 1])
        v.b = s.grid([0, 1])

    def declare_b_then_a(v, s):
        v.b = s.grid([0, 1])
        v.a = s.grid([0, 1])

    v1 = sample_trial(study, declare_a_then_b)
    v2 = sample_trial(study, declare_b_then_a)
    v3 = sample_trial(study, declare_a_then_b)
    v4 = sample_trial(study, declare_b_then_a)

    assert [(v1.a, v1.b), (v2.a, v2.b), (v3.a, v3.b), (v4.a, v4.b)] == [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
    ]


def test_grid_rejects_a_sampling_strategy_as_the_choice_sequence():
    s = Sampler(uuid4(), make_study())

    with pytest.raises(TypeError, match="can't be of type SamplingStrategy"):
        s.grid(s.precomputed(1))


@pytest.mark.parametrize("empty", [[], ""])
def test_grid_rejects_empty_choice_sequences(empty):
    s = Sampler(uuid4(), make_study())

    with pytest.raises(ValueError, match="empty collection"):
        s.grid(empty)


def test_grid_rejects_changing_the_number_of_choices_for_a_variable():
    study = make_study()
    sample_trial(study, {"x": [1, 2]})

    with pytest.raises(StructureError, match="must be a fixed list"):
        sample_trial(study, {"x": [1, 2, 3]})


def test_grid_rejects_declaring_the_same_variable_twice():
    study = make_study()

    def declare(v, s):
        v.x = s.grid([1, 2])
        v.x = s.grid([1, 2])

    # The duplicate is normally caught by SamplingStrategy's name tracker.
    with pytest.raises(StructureError, match='variable name "x" is not unique'):
        sample_trial(study, declare)


def test_grid_rejects_sampling_the_same_variable_twice_in_a_trial():
    study = make_study()
    trial_id = uuid4()
    sample_trial(study, {"x": [1, 2]}, trial_id=trial_id)

    # A different Sampler with the same trial identity is still the same grid
    # round, so the internal duplicate guard must fire.
    s = Sampler(trial_id, study)
    v = s._make_var_namespace()
    with pytest.raises(StructureError, match="called grid twice"):
        v.x = s.grid([1, 2])


def test_grid_dispose_rejects_missing_variable():
    study = make_study()
    sample_trial(study, {"a": [1, 2], "b": [3, 4]})

    s = Sampler(uuid4(), study)
    v = s._make_var_namespace()
    v.a = s.grid([1, 2])

    with pytest.raises(StructureError, match="same variable names"):
        s.dispose()


def test_grid_dispose_rejects_forgetting_a_previously_seen_variable():
    study = make_study()
    sample_trial(study, {"a": [1, 2]})
    # Introducing a new grid variable is accepted on its first appearance...
    sample_trial(study, {"a": [1, 2], "b": [3, 4]})

    # ...but once it is part of the grid, it must be sampled on every round.
    s = Sampler(uuid4(), study)
    v = s._make_var_namespace()
    v.a = s.grid([1, 2])

    with pytest.raises(StructureError, match="same variable names"):
        s.dispose()


def test_grid_dispose_rejects_skipping_all_grid_calls():
    study = make_study()
    sample_trial(study, {"a": [1, 2]})

    s = Sampler(uuid4(), study)
    s._make_var_namespace()

    with pytest.raises(StructureError, match="same variable names"):
        s.dispose()


def test_grid_dispose_is_idempotent():
    study = make_study()
    s = Sampler(uuid4(), study)
    v = s._make_var_namespace()
    v.x = s.grid([1, 2])

    s.dispose()
    s.dispose()  # Must not raise a second time.


def test_grid_dispose_without_grid_does_not_raise():
    study = make_study()
    s = Sampler(uuid4(), study)
    v = s._make_var_namespace()
    v.x = s.precomputed(123)

    s.dispose()

    assert "rungrid_grid" not in study.user_attrs


class GridIntegrationExperiment(Experiment):
    """Experiment used to check `Sampler.grid` through `OptimizingSampler`."""

    def version(self) -> str:
        return "1.0-grid-integration"

    def variables(self, v: VarNamespace, s: Sampler) -> None:
        v.x = s.grid([1, 2, 3])

    @step_method(first=True)
    def step_start(self, trial, **kwargs):
        return "done"


def test_grid_enumerates_over_optimizing_sampler_trials():
    study = make_study()
    exp = GridIntegrationExperiment.get_instance()
    sampler = OptimizingSampler(exp, study)

    trials = list(itertools.islice(sampler.get_trials(), 7))

    assert [t.v.x for t in trials] == [1, 2, 3, 1, 2, 3, 1]


def test_optimizing_sampler_gives_each_grid_trial_a_unique_uuid():
    study = make_study()
    exp = GridIntegrationExperiment.get_instance()
    sampler = OptimizingSampler(exp, study)

    trials = list(itertools.islice(sampler.get_trials(), 5))

    assert len({t.uuid for t in trials}) == 5
