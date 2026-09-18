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
    Sink,
    Source,
)


class SimpleSink(Sink):
    """A basic in-memory sink for testing."""

    def __init__(self):
        self.trials = []

    def put_trial(self, trial):
        self.trials.append(trial)


class SimpleSource(Source):
    """A basic in-memory source for testing."""

    def __init__(self, trials):
        self._trials = list(trials)

    def get_trials(self, *, finished=None, search_tag=None):
        for t in self._trials:
            if finished is not None and t.is_finished() != finished:
                continue
            if search_tag is not None and search_tag not in t.tags:
                continue
            yield t

    def consume_and_tag(
        self, applied_tag, *, finished=None, search_tag=None, predicate=lambda _: True
    ):
        for t in self._trials:
            if finished is not None and t.is_finished() != finished:
                continue
            if search_tag is not None and search_tag not in t.tags:
                continue
            if applied_tag not in t.tags and predicate(t):
                t.add_tag(applied_tag)
                return t
        return None


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

    source1 = SimpleSource([trial1])
    source2 = SimpleSource([trial2])

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

    sink1 = SimpleSink()
    sink2 = SimpleSink()
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
    sub_sink1 = SimpleSink()
    sink1 = FilterSink(sub_sink1, lambda t: "test-tag" not in t.tags)

    # sink2 accepts all trials
    sub_sink2 = SimpleSink()
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

    source = SimpleSource([trial1, trial2])

    # FilterSource
    f_source = FilterSource(source, lambda t: "target" in t.tags)
    trials = list(f_source.get_trials())
    assert len(trials) == 1
    assert trials[0] is trial1

    # FilterSink
    sink = SimpleSink()
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
    finite_source = SimpleSource([finite_trial])
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
