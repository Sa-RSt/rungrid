"""Additional tests for plumbing sources, sinks, filters, serializers and podium/state sinks."""

import datetime
import io
import random
from uuid import uuid4

from rungrid import plumbing
from rungrid.experiment import StaleTrial, StepRecord, TrialResult, VarNamespace
from rungrid.plumbing import (
    CopyingMultiSink,
    Empty,
    FilterSink,
    FilterSource,
    LoaderDumper,
    MultiSource,
    RandomMultiSink,
    ResultPodiumSink,
    Sink,
    Source,
    StateSink,
)
from rungrid.plumbing.storage import InMemoryStorage


def make_trial(number: int = 1, *, tags=(), result=None) -> StaleTrial:
    """Create a simple StaleTrial."""
    trial = StaleTrial(number, uuid4(), VarNamespace(set()), tags=set(tags))
    if result is not None:
        trial._result = result
    return trial


class RecordingSink(Sink):
    """Sink that records calls."""

    def __init__(self) -> None:
        self.single: list = []
        self.batches: list[list] = []

    def put_trial(self, trial):
        self.single.append(trial)

    def put_trials(self, trials):
        self.batches.append(list(trials))
        self.single.extend(trials)


class RecordingSource(Source):
    """Source that records query arguments and yields a fixed list."""

    def __init__(self, trials) -> None:
        self.trials = list(trials)
        self.get_calls: list[tuple] = []
        self.consume_calls: list[tuple] = []

    def get_trials(self, *, finished=None, search_tag=None):
        self.get_calls.append((finished, search_tag))
        yield from self.trials

    def consume_and_tag(
        self,
        applied_tag,
        *,
        finished=None,
        search_tag=None,
        predicate=lambda _: True,
    ):
        self.consume_calls.append((applied_tag, finished, search_tag, predicate))
        for trial in self.trials:
            if predicate(trial):
                trial.add_tag(applied_tag)
                return trial
        return None


class FailingSource(Source):
    """Source whose consume_and_tag fails a configurable number of times."""

    def __init__(self, fail_times: int, trial: StaleTrial) -> None:
        self.fail_times = fail_times
        self.trial = trial
        self.calls = 0
        self.args: list[tuple] = []
        self.kwargs: list[dict] = []

    def get_trials(self, *, finished=None, search_tag=None):
        return []

    def consume_and_tag(
        self,
        applied_tag,
        *,
        finished=None,
        search_tag=None,
        predicate=lambda _: True,
    ):
        self.calls += 1
        self.args.append((applied_tag, finished, search_tag))
        self.kwargs.append({"predicate": predicate})
        if self.calls > self.fail_times:
            self.trial.add_tag(applied_tag)
            return self.trial
        return None


# ---------------------------------------------------------------------------
# LoaderDumper
# ---------------------------------------------------------------------------


def test_loader_dumper_custom_loader_and_dumper_arguments():
    def loader(stream, prefix, *, suffix):
        return prefix + stream.read().decode() + suffix

    def dumper(obj, stream, prefix, *, suffix):
        stream.write((prefix + obj + suffix).encode())

    ld = LoaderDumper(
        "custom",
        loader,
        dumper,
        ("P",),
        {"suffix": "S"},
        ("D",),
        {"suffix": "E"},
    )

    assert ld.dumps("x") == b"DxE"
    assert ld.loads(b"y") == "PyS"
    assert repr(ld) == "LoaderDumper(custom)"


def test_loader_dumper_load_and_dump_streams():
    ld = LoaderDumper.make_pickle((), {}, (), {})
    stream = io.BytesIO()

    ld.dump({"a": 1}, stream)
    stream.seek(0)
    assert ld.load(stream) == {"a": 1}


def test_loader_dumper_make_dispatches_to_named_backend():
    assert LoaderDumper.make("pickle", (), {}, (), {})._desc == "pickle-default"
    assert LoaderDumper.make("joblib", (), {}, (), {})._desc == "joblib-default"

    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    else:
        assert LoaderDumper.make("torch", (), {}, (), {})._desc == "torch-default"


def test_joblib_loader_dumper_sets_default_mmap_mode():
    ld = LoaderDumper.make_joblib((), {}, (), {})

    assert ld._loader_kwargs["mmap_mode"] == "c"


def test_joblib_loader_dumper_keeps_explicit_mmap_mode():
    ld = LoaderDumper.make_joblib((), {"mmap_mode": None}, (), {})

    assert ld._loader_kwargs["mmap_mode"] is None


def test_default_loader_dumper_falls_back_to_joblib(monkeypatch):
    monkeypatch.setattr(plumbing.importlib.util, "find_spec", lambda name: None)

    ld = LoaderDumper.make_default((), {}, (), {})

    assert ld._desc == "joblib-default"


def test_default_loader_dumper_prefers_torch_and_passes_weights_only(monkeypatch):
    captured = {}

    def fake_make_torch(cls, loader_args, loader_kwargs, dumper_args, dumper_kwargs):
        captured["loader_kwargs"] = loader_kwargs
        return "torch-sentinel"

    monkeypatch.setattr(plumbing.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(LoaderDumper, "make_torch", classmethod(fake_make_torch))

    result = LoaderDumper.make_default((), {"extra": 1}, (), {})

    assert result == "torch-sentinel"
    assert captured["loader_kwargs"] == {"weights_only": False, "extra": 1}


def test_default_loader_dumper_caller_can_override_weights_only(monkeypatch):
    captured = {}

    def fake_make_torch(cls, loader_args, loader_kwargs, dumper_args, dumper_kwargs):
        captured["loader_kwargs"] = loader_kwargs
        return "torch-sentinel"

    monkeypatch.setattr(plumbing.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(LoaderDumper, "make_torch", classmethod(fake_make_torch))

    LoaderDumper.make_default((), {"weights_only": True}, (), {})

    assert captured["loader_kwargs"]["weights_only"] is True


# ---------------------------------------------------------------------------
# Base Sink / Source helpers
# ---------------------------------------------------------------------------


def test_sink_put_trials_default_implementation():
    sink = RecordingSink()
    trials = [make_trial(1), make_trial(2)]

    Sink.put_trials(sink, trials)

    assert sink.single == trials


def test_with_sink_filter_wraps_sink():
    inner = RecordingSink()
    outer = inner.with_sink_filter(lambda trial: trial.optuna_trial_number == 1)

    assert isinstance(outer, FilterSink)
    assert outer.get_predicate()(make_trial(1))
    assert not outer.get_predicate()(make_trial(2))


def test_with_source_filter_wraps_source():
    inner = RecordingSource([make_trial(1)])
    outer = inner.with_source_filter(lambda trial: trial.optuna_trial_number == 1)

    assert isinstance(outer, FilterSource)
    assert outer.get_predicate()(make_trial(1))
    assert not outer.get_predicate()(make_trial(2))


def test_empty_repr():
    assert repr(Empty()) == "Empty()"


# ---------------------------------------------------------------------------
# MultiSource
# ---------------------------------------------------------------------------


def test_multi_source_forwards_filters_to_components():
    source_a = RecordingSource([make_trial(1)])
    source_b = RecordingSource([make_trial(2)])
    multi = MultiSource([source_a, source_b], rng=random.Random(0))

    trials = list(multi.get_trials(finished=True, search_tag="tag"))

    assert {t.optuna_trial_number for t in trials} == {1, 2}
    assert source_a.get_calls == [(True, "tag")]
    assert source_b.get_calls == [(True, "tag")]


def test_multi_source_yields_all_trials_from_all_components():
    trial_a = make_trial(1)
    trial_b = make_trial(2)
    multi = MultiSource(
        [InMemoryStorage([trial_a]), InMemoryStorage([trial_b])],
        rng=random.Random(42),
    )

    trials = list(multi.get_trials())

    assert len(trials) == 2
    assert set(trials) == {trial_a, trial_b}


def test_multi_source_without_components_is_empty():
    multi = MultiSource([], rng=random.Random(0))

    assert list(multi.get_trials()) == []
    assert multi.consume_and_tag("tag") is None


def test_multi_source_consume_and_tag_returns_matching_trial():
    trial = make_trial(1)
    multi = MultiSource([InMemoryStorage([trial])], rng=random.Random(0))

    consumed = multi.consume_and_tag(
        "applied", finished=False, search_tag=None, predicate=lambda t: t is trial
    )

    assert consumed is trial
    assert "applied" in consumed.tags


def test_multi_source_consume_and_tag_falls_back_after_retries():
    trial = make_trial(1)
    source = FailingSource(fail_times=2, trial=trial)
    multi = MultiSource([source], rng=random.Random(0))

    consumed = multi.consume_and_tag("applied")

    assert consumed is trial
    assert source.calls == 3
    assert source.args[0] == ("applied", None, None)


def test_multi_source_consume_and_tag_returns_none_when_all_fail():
    source_a = FailingSource(fail_times=10_000, trial=make_trial(1))
    source_b = FailingSource(fail_times=10_000, trial=make_trial(2))
    multi = MultiSource([source_a, source_b], rng=random.Random(0))

    assert multi.consume_and_tag("applied") is None
    # 2 retries per source, plus the exhaustive fallback over both sources.
    assert source_a.calls + source_b.calls == 6


def test_multi_source_repr():
    source = RecordingSource([])

    assert "MultiSource" in repr(MultiSource([source]))


# ---------------------------------------------------------------------------
# Multi sinks
# ---------------------------------------------------------------------------


def test_copying_multi_sink_put_trial():
    sink_a = RecordingSink()
    sink_b = RecordingSink()
    multi = CopyingMultiSink([sink_a, sink_b])
    trial = make_trial(1)

    multi.put_trial(trial)

    assert sink_a.single == [trial]
    assert sink_b.single == [trial]


def test_copying_multi_sink_put_trials():
    sink_a = RecordingSink()
    sink_b = RecordingSink()
    multi = CopyingMultiSink([sink_a, sink_b])
    trials = [make_trial(1), make_trial(2)]

    multi.put_trials(trials)

    assert sink_a.batches == [trials]
    assert sink_b.batches == [trials]


def test_copying_multi_sink_repr():
    assert "CopyingMultiSink" in repr(CopyingMultiSink([RecordingSink()]))


def test_random_multi_sink_without_components_is_noop():
    sink = RandomMultiSink([], rng=random.Random(0))

    assert sink.put_trial(make_trial(1)) is None


def test_random_multi_sink_uses_an_accepting_component():
    inner = RecordingSink()
    sink = RandomMultiSink([inner], rng=random.Random(0))
    trial = make_trial(1)

    sink.put_trial(trial)

    assert inner.single == [trial]


def test_random_multi_sink_returns_none_when_every_component_rejects():
    inner_a = RecordingSink()
    inner_b = RecordingSink()
    sink = RandomMultiSink(
        [
            FilterSink(inner_a, lambda trial: False),
            FilterSink(inner_b, lambda trial: False),
        ],
        rng=random.Random(0),
    )

    assert sink.put_trial(make_trial(1)) is None
    assert inner_a.single == []
    assert inner_b.single == []


def test_random_multi_sink_repr():
    assert "RandomMultiSink" in repr(RandomMultiSink([RecordingSink()]))


# ---------------------------------------------------------------------------
# FilterSource / FilterSink
# ---------------------------------------------------------------------------


def test_filter_source_filters_trials():
    trial_keep = make_trial(1, tags={"keep"})
    trial_drop = make_trial(2, tags={"drop"})
    source = FilterSource(
        InMemoryStorage([trial_keep, trial_drop]),
        lambda trial: "keep" in trial.tags,
    )

    assert list(source.get_trials()) == [trial_keep]


def test_filter_source_forwards_query_arguments():
    inner = RecordingSource([make_trial(1, tags={"keep"})])
    source = FilterSource(inner, lambda trial: True)

    list(source.get_trials(finished=False, search_tag="tag"))

    assert inner.get_calls == [(False, "tag")]


def test_filter_source_consume_and_tag_combines_predicates():
    trial_keep = make_trial(2, tags={"keep"})
    trial_other = make_trial(1, tags={"keep"})
    inner = InMemoryStorage([trial_other, trial_keep])
    source = FilterSource(inner, lambda trial: "keep" in trial.tags)

    consumed = source.consume_and_tag(
        "chosen", predicate=lambda trial: trial.optuna_trial_number == 2
    )

    assert consumed is trial_keep
    assert "chosen" in consumed.tags


def test_filter_source_get_predicate_and_repr():
    predicate = lambda trial: True
    source = FilterSource(RecordingSource([]), predicate)

    assert source.get_predicate() is predicate
    assert "FilterSource" in repr(source)


def test_filter_sink_put_trial_filters():
    inner = RecordingSink()
    sink = FilterSink(inner, lambda trial: trial.optuna_trial_number == 1)

    sink.put_trial(make_trial(1))
    sink.put_trial(make_trial(2))

    assert [t.optuna_trial_number for t in inner.single] == [1]


def test_filter_sink_put_trials_filters():
    inner = RecordingSink()
    sink = FilterSink(inner, lambda trial: trial.optuna_trial_number % 2 == 0)

    sink.put_trials([make_trial(1), make_trial(2), make_trial(3), make_trial(4)])

    assert [t.optuna_trial_number for t in inner.single] == [2, 4]


def test_filter_sink_get_predicate_and_repr():
    predicate = lambda trial: True
    sink = FilterSink(RecordingSink(), predicate)

    assert sink.get_predicate() is predicate
    assert "FilterSink" in repr(sink)


# ---------------------------------------------------------------------------
# ResultPodiumSink
# ---------------------------------------------------------------------------


def test_result_podium_sink_empty():
    sink = ResultPodiumSink()

    assert sink.get_winner() is None
    assert sink.get_winners() == []


def test_result_podium_sink_top_k_one():
    now = datetime.datetime.now()
    best = make_trial(1, result=TrialResult.make_ok(1.0, now))
    worse = make_trial(2, result=TrialResult.make_ok(2.0, now))

    sink = ResultPodiumSink(direction="minimize", top_k=1)
    sink.put_trial(worse)
    sink.put_trial(best)

    assert sink.get_winner() is best
    assert sink.get_winners() == [best]


def test_result_podium_sink_maximize():
    now = datetime.datetime.now()
    low = make_trial(1, result=TrialResult.make_ok(1.0, now))
    high = make_trial(2, result=TrialResult.make_ok(2.0, now))

    sink = ResultPodiumSink(direction="maximize", top_k=2)
    sink.put_trial(low)
    sink.put_trial(high)

    assert sink.get_winner() is high
    assert sink.get_winners() == [high, low]


def test_result_podium_sink_callback_and_repr():
    now = datetime.datetime.now()
    updates = []
    sink = ResultPodiumSink(
        top_k=1, on_winners_update=lambda winners: updates.append(list(winners))
    )
    trial = make_trial(1, result=TrialResult.make_ok(3.0, now))

    sink.put_trial(trial)

    assert updates == [[trial]]
    assert "minimize" in repr(sink)


def test_result_podium_sink_ignores_trials_without_results():
    sink = ResultPodiumSink()

    sink.put_trial(make_trial(1))

    assert sink.get_winners() == []


# ---------------------------------------------------------------------------
# StateSink
# ---------------------------------------------------------------------------


def make_record(
    next_step_name: str | None,
    kwargs: dict | None,
) -> StepRecord:
    now = datetime.datetime.now()
    return StepRecord(
        step=None,  # type: ignore[arg-type]
        next_step_name=next_step_name,
        next_step_kwargs=kwargs,
        start=now,
        end=now,
    )


def test_state_sink_ignores_trial_without_relevant_records_when_last_only():
    received = []
    sink = StateSink("step_score", "model", lambda trial, value: received.append(value))
    trial = make_trial(1)
    trial._step_records = [make_record("other_step", {"model": "x"})]

    sink.put_trial(trial)

    assert received == []


def test_state_sink_calls_destination_with_empty_list_when_not_last_only():
    received = []
    sink = StateSink(
        "step_score",
        "model",
        lambda trial, value: received.append(value),
        last_only=False,
    )
    trial = make_trial(1)
    trial._step_records = [make_record("other_step", {"model": "x"})]

    sink.put_trial(trial)

    assert received == [[]]


def test_state_sink_ignores_records_without_next_step_kwargs():
    received = []
    sink = StateSink("step_score", "model", lambda trial, value: received.append(value))
    trial = make_trial(1)
    trial._step_records = [make_record("step_score", None)]

    sink.put_trial(trial)

    assert received == []


def test_state_sink_matches_step_name_by_suffix():
    received = []
    sink = StateSink("step_score", "model", lambda trial, value: received.append(value))
    trial = make_trial(1)
    trial._step_records = [
        make_record("package.module.step_score", {"model": "suffix-match"})
    ]

    sink.put_trial(trial)

    assert received == ["suffix-match"]


def test_state_sink_repr():
    sink = StateSink("step_score", "model", lambda trial, value: None, last_only=False)

    assert repr(sink) == (
        "StateSink(step_name=step_score, arg_name=model, last_only=False)"
    )
