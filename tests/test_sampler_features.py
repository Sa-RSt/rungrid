"""Tests for ``Sampler`` strategies other than ``Sampler.grid``."""

import random
from uuid import uuid4

import optuna
import pytest

from rungrid.error import StructureError
from rungrid.experiment import Sampler, SamplingStrategy


def make_study() -> optuna.Study:
    """Create a fresh Optuna study."""
    return optuna.create_study(direction="minimize")


def make_sampler(study: optuna.Study | None = None) -> Sampler:
    """Create a Sampler backed by a fresh or given study."""
    return Sampler(uuid4(), study if study is not None else make_study())


def sample_one(declare) -> tuple[object, Sampler, optuna.Study]:
    """Run one sampling round and return the value, sampler and study."""
    study = make_study()
    s = make_sampler(study)
    v = s._make_var_namespace()
    declare(v, s)
    s.dispose()
    return v, s, study


# ---------------------------------------------------------------------------
# Optuna trial management
# ---------------------------------------------------------------------------


def test_sampler_does_not_ask_optuna_eagerly():
    study = make_study()
    s = make_sampler(study)

    assert not s.has_asked_optuna()
    assert study.trials == []


def test_get_optuna_trial_is_lazy_and_cached():
    study = make_study()
    s = make_sampler(study)

    trial = s.get_optuna_trial()
    assert s.has_asked_optuna()
    assert s.get_optuna_trial() is trial
    assert len(study.trials) == 1


def test_get_optuna_trial_returns_an_optuna_trial():
    s = make_sampler()

    assert isinstance(s.get_optuna_trial(), optuna.Trial)


def test_sampler_uses_string_trial_id():
    study = make_study()
    trial_id = uuid4()

    s = Sampler(trial_id, study)

    assert s._trial_id == str(trial_id)


# ---------------------------------------------------------------------------
# Precomputed values
# ---------------------------------------------------------------------------


def test_precomputed_returns_constant_and_name():
    s = make_sampler()
    value = object()
    ss = s.precomputed(value)

    assert ss.name == f"precomputed({value})"
    assert ss.sample("x") is value


def test_precomputed_works_with_unusual_values():
    s = make_sampler()

    for value in [None, True, (1, 2), {"a": 1}, [1, 2]]:
        assert s.precomputed(value).sample(f"x{id(value)}") is value


# ---------------------------------------------------------------------------
# Integer distributions
# ---------------------------------------------------------------------------


def test_integer_samples_within_inclusive_bounds():
    v, _, study = sample_one(lambda v, s: setattr(v, "x", s.integer(3, 7)))

    assert isinstance(v.x, int)
    assert 3 <= v.x <= 7
    assert "x" in study.trials[0].params


@pytest.mark.parametrize(
    "lower,upper,message",
    [
        (1.0, 2, "lower: expected type int, got float"),
        (1, 2.0, "upper: expected type int, got float"),
        (5, 5, "minimum \\(5\\) == maximum \\(5\\)"),
        (6, 5, "minimum \\(6\\) > maximum \\(5\\)"),
    ],
)
def test_integer_validates_bounds_and_types(lower, upper, message):
    s = make_sampler()

    with pytest.raises((TypeError, ValueError), match=message):
        s.integer(lower, upper)


def test_integer_rejects_sampling_strategy_bounds():
    s = make_sampler()
    ss = s.precomputed(1)

    with pytest.raises(TypeError, match="lower can't be of type SamplingStrategy"):
        s.integer(ss, 2)

    with pytest.raises(TypeError, match="upper can't be of type SamplingStrategy"):
        s.integer(1, ss)


def test_integer_power_samples_powers_of_base():
    v, _, _ = sample_one(lambda v, s: setattr(v, "x", s.integer_power(3, 0, 3)))

    assert v.x in {1, 3, 9, 27}


def test_power_of_two_samples_powers_of_two():
    v, _, _ = sample_one(lambda v, s: setattr(v, "x", s.power_of_two(0, 4)))

    assert v.x in {1, 2, 4, 8, 16}


def test_integer_power_name_contains_transformation():
    s = make_sampler()

    ss = s.integer_power(2, 1, 4)

    assert ss.name == "integer(1,4)/transformed(pow_2)"


# ---------------------------------------------------------------------------
# Float distributions
# ---------------------------------------------------------------------------


def test_uniform_samples_within_bounds():
    v, _, study = sample_one(lambda v, s: setattr(v, "x", s.uniform(1.5, 2.5)))

    assert isinstance(v.x, float)
    assert 1.5 <= v.x <= 2.5
    assert "x" in study.trials[0].params


@pytest.mark.parametrize(
    "lower,upper,message",
    [
        (1, 2.0, "lower: expected type float, got int"),
        (1.0, 2, "upper: expected type float, got int"),
        (2.0, 1.0, "minimum \\(2.0\\) > maximum \\(1.0\\)"),
        (2.0, 2.0, "minimum \\(2.0\\) == maximum \\(2.0\\)"),
    ],
)
def test_uniform_validates_bounds_and_types(lower, upper, message):
    s = make_sampler()

    with pytest.raises((TypeError, ValueError), match=message):
        s.uniform(lower, upper)


def test_uniform_rejects_sampling_strategy_bounds():
    s = make_sampler()
    ss = s.precomputed(1.0)

    with pytest.raises(TypeError, match="lower can't be of type SamplingStrategy"):
        s.uniform(ss, 2.0)
    with pytest.raises(TypeError, match="upper can't be of type SamplingStrategy"):
        s.uniform(1.0, ss)


def test_log_uniform_samples_within_positive_bounds():
    v, _, _ = sample_one(lambda v, s: setattr(v, "x", s.log_uniform(0.01, 1.0)))

    assert 0.01 <= v.x <= 1.0


@pytest.mark.parametrize(
    "lower,upper,message",
    [
        (1, 2.0, "lower: expected type float, got int"),
        (2.0, 1.0, "minimum \\(2.0\\) > maximum \\(1.0\\)"),
        (2.0, 2.0, "minimum \\(2.0\\) == maximum \\(2.0\\)"),
    ],
)
def test_log_uniform_validates_bounds_and_types(lower, upper, message):
    s = make_sampler()

    with pytest.raises((TypeError, ValueError), match=message):
        s.log_uniform(lower, upper)


def test_discrete_uniform_samples_within_bounds():
    v, _, _ = sample_one(
        lambda v, s: setattr(v, "x", s.discrete_uniform(0.0, 1.0, 0.1))
    )

    assert 0.0 <= v.x <= 1.0


def test_discrete_uniform_rejects_too_large_quantization():
    s = make_sampler()

    with pytest.raises(ValueError, match="quantization interval too large"):
        s.discrete_uniform(0.0, 1.0, 1.0)

    with pytest.raises(ValueError, match="quantization interval too large"):
        s.discrete_uniform(0.0, 1.0, 2.0)


@pytest.mark.parametrize(
    "lower,upper,message",
    [
        (0, 1.0, "lower: expected type float, got int"),
        (0.0, 1, "upper: expected type float, got int"),
        (2.0, 1.0, "minimum \\(2.0\\) > maximum \\(1.0\\)"),
    ],
)
def test_discrete_uniform_validates_bounds_and_types(lower, upper, message):
    s = make_sampler()

    with pytest.raises((TypeError, ValueError), match=message):
        s.discrete_uniform(lower, upper, 0.1)


# ---------------------------------------------------------------------------
# Categorical distributions
# ---------------------------------------------------------------------------


def test_categorical_samples_a_category():
    v, _, study = sample_one(
        lambda v, s: setattr(v, "x", s.categorical(["a", "b", "c"]))
    )

    assert v.x in {"a", "b", "c"}
    assert "x" in study.trials[0].params


def test_categorical_name_lists_all_choices():
    s = make_sampler()

    assert s.categorical([1, 2, 3]).name == "categorical(1,2,3)"


def test_categorical_rejects_empty_collection():
    s = make_sampler()

    with pytest.raises(ValueError, match="empty collection"):
        s.categorical([])


def test_categorical_rejects_strategy_as_outer_sequence():
    s = make_sampler()

    with pytest.raises(TypeError, match="can't be of type SamplingStrategy"):
        s.categorical(s.precomputed(1))


def test_categorical_rejects_strategy_choices():
    s = make_sampler()
    ss = s.precomputed(1)

    with pytest.raises(TypeError, match="categories\\[0\\] can't be of type"):
        s.categorical([ss])
    with pytest.raises(TypeError, match="categories\\[1\\] can't be of type"):
        s.categorical([1, ss])


def test_ordinal_samples_an_ordered_category():
    categories = ["small", "medium", "large"]
    v, _, _ = sample_one(lambda v, s: setattr(v, "x", s.ordinal(categories)))

    assert v.x in categories


def test_ordinal_name_lists_all_choices():
    s = make_sampler()

    assert s.ordinal([1, 2, 3]).name == "ordinal(1,2,3)"


def test_ordinal_rejects_empty_collection():
    s = make_sampler()

    with pytest.raises(ValueError, match="empty collection"):
        s.ordinal([])


def test_ordinal_rejects_strategy_choices():
    s = make_sampler()
    ss = s.precomputed(1)

    with pytest.raises(TypeError, match="categories\\[0\\] can't be of type"):
        s.ordinal([ss])


def test_boolean_samples_boolean():
    v, _, _ = sample_one(lambda v, s: setattr(v, "x", s.boolean()))

    assert v.x in {True, False}


def test_boolean_name_is_categorical():
    s = make_sampler()

    assert s.boolean().name == "categorical(True,False)"


# ---------------------------------------------------------------------------
# Random delegation
# ---------------------------------------------------------------------------


def test_random_selects_among_values_and_strategies():
    rng = random.Random(123)
    v, _, _ = sample_one(
        lambda v, s: setattr(v, "x", s.random([1, "two", s.precomputed(3)], rng=rng))
    )

    assert v.x in {1, "two", 3}


def test_random_name_uses_wrapped_strategy_names():
    s = make_sampler()

    assert (
        s.random([1, "a"], rng=random.Random(0)).name
        == "random(precomputed(1)|precomputed(a))"
    )


def test_random_rejects_strategy_as_outer_sequence():
    s = make_sampler()

    with pytest.raises(TypeError, match="strategies can't be of type SamplingStrategy"):
        s.random(s.precomputed(1))


def test_random_rejects_invalid_rng_type():
    s = make_sampler()

    with pytest.raises(TypeError, match="rng: expected type Random, got int"):
        s.random([1, 2], rng=42)


def test_random_draws_follow_the_supplied_rng():
    rng = random.Random(7)
    reference = random.Random(7)
    study = make_study()
    outcomes = []

    for _ in range(10):
        s = make_sampler(study)
        v = s._make_var_namespace()
        v.x = s.random([1, 2, 3], rng=rng)
        s.dispose()
        outcomes.append(v.x)

    expected = [[1, 2, 3][reference.randrange(3)] for _ in range(10)]
    assert outcomes == expected


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------


def test_transformed_applies_transformation():
    v, _, _ = sample_one(
        lambda v, s: setattr(v, "x", s.transformed(3, lambda x: x * 2, "double"))
    )

    assert v.x == 6


def test_transformed_defaults_name_to_function_name():
    def increment(x):
        return x + 1

    s = make_sampler()

    assert s.transformed(1, increment).name == "precomputed(1)/transformed(increment)"


def test_transformed_accepts_custom_name():
    s = make_sampler()

    assert (
        s.transformed(1, lambda x: x, "identity").name
        == "precomputed(1)/transformed(identity)"
    )


def test_transformed_accepts_sampling_strategy_base():
    v, _, _ = sample_one(
        lambda v, s: setattr(
            v, "x", s.transformed(s.integer(1, 3), lambda x: x * 10, "times_ten")
        )
    )

    assert v.x in {10, 20, 30}


def test_transformed_rejects_strategy_transformation():
    s = make_sampler()

    with pytest.raises(TypeError, match="trans can't be of type SamplingStrategy"):
        s.transformed(1, s.precomputed(2))


def test_transformed_rejects_strategy_alt_name():
    s = make_sampler()

    with pytest.raises(TypeError, match="alt_name can't be of type SamplingStrategy"):
        s.transformed(1, lambda x: x, s.precomputed("name"))


def test_transformed_rejects_non_string_alt_name():
    s = make_sampler()

    with pytest.raises(TypeError, match="alt_name: expected type str, got int"):
        s.transformed(1, lambda x: x, 42)


# ---------------------------------------------------------------------------
# Lists
# ---------------------------------------------------------------------------


def test_list_samples_multiple_values():
    v, _, _ = sample_one(lambda v, s: setattr(v, "x", s.list(s.integer(0, 10), 3)))

    assert isinstance(v.x, list)
    assert len(v.x) == 3
    assert all(0 <= x <= 10 for x in v.x)


def test_list_of_zero_length_is_empty():
    v, s, study = sample_one(lambda v, s: setattr(v, "x", s.list(s.integer(0, 10), 0)))

    assert v.x == []
    assert not s.has_asked_optuna()


def test_list_name_contains_length():
    s = make_sampler()

    assert s.list(s.integer(0, 1), 4).name == "integer(0,1)/list(4)"


def test_list_rejects_negative_length():
    s = make_sampler()

    with pytest.raises(ValueError, match="length can't be negative"):
        s.list(s.precomputed(1), -1)


def test_list_rejects_non_integer_length():
    s = make_sampler()

    with pytest.raises(TypeError, match="length: expected type int, got float"):
        s.list(s.precomputed(1), 1.5)


def test_list_rejects_strategy_length():
    s = make_sampler()

    with pytest.raises(TypeError, match="length can't be of type SamplingStrategy"):
        s.list(s.precomputed(1), s.precomputed(2))


def test_list_transformed_passes_index_and_value():
    v, _, _ = sample_one(
        lambda v, s: setattr(
            v,
            "x",
            s.list_transformed(s.precomputed(10), 3, lambda i, x: i + x, "add_index"),
        )
    )

    assert v.x == [10, 11, 12]


def test_list_transformed_default_name_uses_function_name():
    def add_index(i, x):
        return i + x

    s = make_sampler()

    assert (
        s.list_transformed(s.precomputed(0), 2, add_index).name
        == "precomputed(0)/list_transformed(add_index:2)"
    )


def test_list_transformed_rejects_negative_length():
    s = make_sampler()

    with pytest.raises(ValueError, match="length can't be negative"):
        s.list_transformed(s.precomputed(1), -1, lambda i, x: x)


def test_list_transformed_rejects_non_integer_length():
    s = make_sampler()

    with pytest.raises(TypeError, match="length: expected type int, got str"):
        s.list_transformed(s.precomputed(1), "2", lambda i, x: x)


# ---------------------------------------------------------------------------
# Branches
# ---------------------------------------------------------------------------


def test_branch_prefixes_parameter_names_without_changing_values():
    study = make_study()

    s = make_sampler(study)
    v = s._make_var_namespace()
    with s.branch("LSTM"):
        v.x = s.integer(0, 5)
    s.dispose()

    assert 0 <= v.x <= 5
    assert "LSTM--x" in study.trials[0].params


def test_nested_branches_join_names():
    study = make_study()

    s = make_sampler(study)
    v = s._make_var_namespace()
    with s.branch("outer"):
        with s.branch("inner"):
            v.x = s.integer(0, 5)
    s.dispose()

    assert "outer--inner--x" in study.trials[0].params


def test_branch_scope_ends_when_context_exits():
    study = make_study()

    s = make_sampler(study)
    v = s._make_var_namespace()
    with s.branch("branched"):
        v.x = s.integer(0, 5)
    v.y = s.integer(0, 5)
    s.dispose()

    params = study.trials[0].params
    assert "branched--x" in params
    assert "y" in params


def test_same_variable_name_can_be_used_in_different_branches_across_trials():
    study = make_study()

    s1 = make_sampler(study)
    v1 = s1._make_var_namespace()
    with s1.branch("LSTM"):
        v1.hidden = s1.categorical([4, 8])
    s1.dispose()

    s2 = make_sampler(study)
    v2 = s2._make_var_namespace()
    with s2.branch("GRU"):
        v2.hidden = s2.categorical([16, 32])
    s2.dispose()

    assert "LSTM--hidden" in study.trials[0].params
    assert "GRU--hidden" in study.trials[1].params


# ---------------------------------------------------------------------------
# Disposal and sampling strategy invariants
# ---------------------------------------------------------------------------


def test_disposed_sampling_strategy_cannot_sample():
    study = make_study()
    s = make_sampler(study)
    v = s._make_var_namespace()
    v.x = s.precomputed(1)
    ss = v._sampling_strategies["x"]

    s.dispose()

    with pytest.raises(StructureError, match="already been disposed"):
        ss.sample("y")


def test_dispose_is_idempotent_for_plain_samplers():
    s = make_sampler()
    v = s._make_var_namespace()
    v.x = s.precomputed(1)

    s.dispose()
    s.dispose()


def test_sampling_strategy_duplicate_name_uses_shared_tracker():
    names = {"x"}
    ss = SamplingStrategy("dummy", lambda name: 1, names, lambda name: name)

    with pytest.raises(StructureError, match='variable name "x" is not unique'):
        ss.sample("x")


def test_sampling_strategy_dispose_makes_sample_fail():
    names: set[str] = set()
    ss = SamplingStrategy("dummy", lambda name: 1, names, lambda name: name)
    ss.dispose()

    with pytest.raises(StructureError, match="already been disposed"):
        ss.sample("x")


def test_sampling_strategy_name_property():
    ss = SamplingStrategy("my-name", lambda name: 1, set(), lambda name: name)

    assert ss.name == "my-name"
