import random
from uuid import uuid4
from warnings import warn

from rungrid.experiment import StaleTrial, VarNamespace
from rungrid.plumbing import (
    CopyingMultiSink,
    Empty,
    FilterSink,
    FilterSource,
    LoaderDumper,
    MultiSource,
    RandomMultiSink,
    Source,
    StateSink,
    ResultPodiumSink,
)
from rungrid.plumbing.storage import InMemoryStorage


def test_loader_dumpers():
    """Verify different LoaderDumper backends serialize and deserialize correctly."""
    data = {"hello": "world", "number": 123}

    # Pickle
    ld_pickle = LoaderDumper.make_pickle((), {}, (), {})
    assert ld_pickle.loads(ld_pickle.dumps(data)) == data

    # Joblib
    ld_joblib = LoaderDumper.make_joblib((), {}, (), {})
    assert ld_joblib.loads(ld_joblib.dumps(data)) == data

    # Torch (if available)
    try:
        import torch

        ld_torch = LoaderDumper.make_torch((), {}, (), {})
        assert ld_torch.loads(ld_torch.dumps(data)) == data
    except ImportError:
        warn("failed to import torch; not running torch-related test")

    # Default
    ld_default = LoaderDumper.make_default((), {}, (), {})
    assert ld_default.loads(ld_default.dumps(data)) == data


def test_empty_source_sink():
    """Verify that Empty behaves as a silent sink and empty source."""
    empty = Empty()
    v = VarNamespace(set())
    trial = StaleTrial(1, uuid4(), v)

    # Should not raise any exception
    empty.put_trial(trial)

    assert list(empty.get_trials()) == []
    assert empty.consume_and_tag("my-tag") is None


def test_multi_source():
    """Test MultiSource merges multiple sources and can be randomized."""
    v = VarNamespace(set())
    trial1 = StaleTrial(1, uuid4(), v)
    trial2 = StaleTrial(2, uuid4(), v)

    source1 = InMemoryStorage([trial1])
    source2 = InMemoryStorage([trial2])

    # Using deterministic seed for RNG
    rng = random.Random(42)
    multi = MultiSource([source1, source2], rng=rng)

    results = list(multi.get_trials())
    assert len(results) == 2
    assert trial1 in results
    assert trial2 in results


def test_copying_multi_sink():
    """Verify CopyingMultiSink copies to all underlying sinks."""
    v = VarNamespace(set())
    trial = StaleTrial(1, uuid4(), v)

    sink1 = InMemoryStorage()
    sink2 = InMemoryStorage()
    multi = CopyingMultiSink([sink1, sink2])

    multi.put_trial(trial)
    assert len(sink1.trials) == 1
    assert len(sink2.trials) == 1
    assert sink1.trials[0] is trial
    assert sink2.trials[0] is trial


def test_random_multi_sink_deterministic_fallback():
    """Verify RandomMultiSink never discards a trial even if RNG picks a filtering sink first."""
    v = VarNamespace(set())
    trial = StaleTrial(1, uuid4(), v, tags={"test-tag"})

    # sink1 rejects all trials containing 'test-tag'
    sub_sink1 = InMemoryStorage()
    sink1 = FilterSink(sub_sink1, lambda t: "test-tag" not in t.tags)

    # sink2 accepts all trials
    sub_sink2 = InMemoryStorage()
    sink2 = sub_sink2

    # Regardless of RNG choices or initial selections, since sink1 rejects, the trial MUST end up in sink2.
    # We run multiple times with random RNG/seeds to ensure it is 100% deterministic and non-flaky.
    for seed in range(50):
        sub_sink1.trials.clear()
        sub_sink2.trials.clear()

        rng = random.Random(seed)
        multi = RandomMultiSink([sink1, sink2], rng=rng)
        multi.put_trial(trial)

        assert len(sub_sink1.trials) == 0, (
            f"Trial was incorrectly written to filtered sink (seed {seed})"
        )
        assert len(sub_sink2.trials) == 1, (
            f"Trial was discarded or not written to the only accepting sink (seed {seed})"
        )


def test_filters():
    """Verify FilterSource and FilterSink filter trials using predicates."""
    v = VarNamespace(set())
    trial1 = StaleTrial(1, uuid4(), v, tags={"target"})
    trial2 = StaleTrial(2, uuid4(), v, tags={"other"})

    source = InMemoryStorage([trial1, trial2])

    # FilterSource
    f_source = FilterSource(source, lambda t: "target" in t.tags)
    trials = list(f_source.get_trials())
    assert len(trials) == 1
    assert trials[0] is trial1

    # FilterSink
    sink = InMemoryStorage()
    f_sink = FilterSink(sink, lambda t: "target" in t.tags)

    f_sink.put_trial(trial1)
    f_sink.put_trial(trial2)
    assert len(sink.trials) == 1
    assert sink.trials[0] is trial1


def test_multi_source_infinite_interleave():
    """Verify MultiSource interleaves elements and does not block on infinite component sources."""
    import itertools

    class InfiniteSource(Source):
        def get_trials(self, *, finished=None, search_tag=None):
            v = VarNamespace(set())
            counter = 0
            while True:
                yield StaleTrial(counter, uuid4(), v)
                counter += 1

        def consume_and_tag(self, applied_tag, *, finished=None, search_tag=None):
            pass

    finite_trial = StaleTrial(100, uuid4(), VarNamespace(set()))
    finite_source = InMemoryStorage([finite_trial])
    infinite_source = InfiniteSource()

    # Create MultiSource and get trials (deterministically seeded)
    rng = random.Random(42)
    multi = MultiSource([finite_source, infinite_source], rng=rng)

    # Since one source is infinite, we slice the iterator to avoid infinite loop
    sliced_trials = list(itertools.islice(multi.get_trials(), 100))

    # Assert we successfully yielded multiple trials
    assert len(sliced_trials) == 100
    # Assert that the finite trial was successfully interleaved and retrieved,
    # proving the infinite source did not block/monopolize the execution.
    assert finite_trial in sliced_trials


def test_loader_dumper_invalid_backend_raises_key_error():
    """Verify LoaderDumper.make raises KeyError when called with an invalid backend name."""
    import pytest

    with pytest.raises(KeyError):
        LoaderDumper.make("invalid_backend_name", (), {}, (), {})


def test_result_podium_sink():
    """Verify ResultPodiumSink tracks top-k best trials and triggers callback correctly."""
    from datetime import datetime
    from rungrid.experiment import TrialResult, StaleTrial, VarNamespace

    v = VarNamespace(set())
    now = datetime.now()

    # Create a few trials
    t1 = StaleTrial(1, uuid4(), v).with_result(TrialResult.make_ok(10.0, now))
    t2 = StaleTrial(2, uuid4(), v).with_result(TrialResult.make_ok(20.0, now))
    t3 = StaleTrial(3, uuid4(), v).with_result(TrialResult.make_ok(5.0, now))
    # Error trial (should be ignored)
    t_err = StaleTrial(4, uuid4(), v).with_result(TrialResult.make_error("oops", now))
    # Pruned trial (should be ignored)
    t_pruned = StaleTrial(5, uuid4(), v).with_result(
        TrialResult.make_pruned("pruned", now)
    )
    # Trial without result (should be ignored)
    t_no_res = StaleTrial(6, uuid4(), v)

    # Test minimize (default)
    updates_min = []
    sink_min = ResultPodiumSink(
        direction="minimize",
        top_k=2,
        on_winners_update=lambda winners: updates_min.append(list(winners)),
    )

    # Put a trial without result
    sink_min.put_trial(t_no_res)
    assert len(sink_min.get_winners()) == 0
    assert len(updates_min) == 0

    # Put error and pruned trials
    sink_min.put_trial(t_err)
    sink_min.put_trial(t_pruned)
    assert len(sink_min.get_winners()) == 0
    assert len(updates_min) == 0

    # Put t1 (result = 10.0) -> podium updated to [t1]
    sink_min.put_trial(t1)
    assert sink_min.get_winner() is t1
    assert sink_min.get_winners() == [t1]
    assert len(updates_min) == 1
    assert updates_min[-1] == [t1]

    # Put t2 (result = 20.0) -> t2 is worse, but since top_k=2, it enters podium: [t1, t2]
    sink_min.put_trial(t2)
    assert sink_min.get_winner() is t1
    assert sink_min.get_winners() == [t1, t2]
    assert len(updates_min) == 2
    assert updates_min[-1] == [t1, t2]

    # Put t3 (result = 5.0) -> t3 is best: [t3, t1] (t2 is pushed out)
    sink_min.put_trial(t3)
    assert sink_min.get_winner() is t3
    assert sink_min.get_winners() == [t3, t1]
    assert len(updates_min) == 3
    assert updates_min[-1] == [t3, t1]

    # Put t1 again -> same result, no podium change, callback should not run
    sink_min.put_trial(t1)
    assert len(updates_min) == 3

    # Test maximize
    updates_max = []
    sink_max = ResultPodiumSink(
        direction="maximize",
        top_k=2,
        on_winners_update=lambda winners: updates_max.append(list(winners)),
    )

    sink_max.put_trial(t1)  # [t1]
    assert sink_max.get_winner() is t1
    assert updates_max[-1] == [t1]

    sink_max.put_trial(t2)  # [t2, t1]
    assert sink_max.get_winner() is t2
    assert sink_max.get_winners() == [t2, t1]
    assert updates_max[-1] == [t2, t1]

    sink_max.put_trial(t3)  # t3 is worse than t1 and t2, podium remains [t2, t1]
    assert sink_max.get_winner() is t2
    assert len(updates_max) == 2  # No update triggered since podium didn't change

    # Invalid direction raises ValueError
    import pytest

    with pytest.raises(ValueError):
        ResultPodiumSink(direction="invalid_direction")

    # Repr verification
    assert "ResultPodiumSink" in repr(sink_min)
    assert "minimize" in repr(sink_min)
    assert "maximize" in repr(sink_max)


def test_state_sink():
    """Verify StateSink processes step records and extracts arguments correctly."""
    from datetime import datetime
    from rungrid.experiment import StepRecord, StaleTrial, VarNamespace

    v = VarNamespace(set())
    now = datetime.now()

    rec1 = StepRecord(
        step=None,  # type: ignore
        next_step_name="aaa.bb.step_score",
        next_step_kwargs={"model": "model_v1", "other": 123},
        start=now,
        end=now,
    )
    rec2 = StepRecord(
        step=None,  # type: ignore
        next_step_name="aaa.bb.step_score",
        next_step_kwargs={"model": "model_v2"},
        start=now,
        end=now,
    )
    # Irrelevant record (wrong step name)
    rec_irrelevant_step = StepRecord(
        step=None,  # type: ignore
        next_step_name="ccc.dd.other_step",
        next_step_kwargs={"model": "model_v3"},
        start=now,
        end=now,
    )
    # Irrelevant record (missing arg)
    rec_missing_arg = StepRecord(
        step=None,  # type: ignore
        next_step_name="aaa.bb.step_score",
        next_step_kwargs={"different_arg": "some_val"},
        start=now,
        end=now,
    )

    # Trial with all records
    trial = StaleTrial(
        1, uuid4(), v, step_records=[rec1, rec_irrelevant_step, rec_missing_arg, rec2]
    )

    # Test last_only=True
    received_last = []
    sink_last = StateSink(
        step_name="step_score",
        arg_name="model",
        dest=lambda t, val: received_last.append((t, val)),
        last_only=True,
    )
    sink_last.put_trial(trial)
    assert len(received_last) == 1
    assert received_last[0][0] is trial
    assert received_last[0][1] == "model_v2"

    # Test last_only=False
    received_all = []
    sink_all = StateSink(
        step_name="step_score",
        arg_name="model",
        dest=lambda t, val: received_all.append((t, val)),
        last_only=False,
    )
    sink_all.put_trial(trial)
    assert len(received_all) == 1
    assert received_all[0][0] is trial
    assert received_all[0][1] == ["model_v1", "model_v2"]

    # Test representation
    assert "StateSink" in repr(sink_last)
    assert "step_name=step_score" in repr(sink_last)
    assert "arg_name=model" in repr(sink_last)
