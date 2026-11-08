"""Additional tests for the command-line interface and its helper methods."""

import io
import os
from types import SimpleNamespace

import pytest

from rungrid.cli import CLI
from rungrid.plumbing import CopyingMultiSink, Empty, MultiSource, Source
from rungrid.plumbing.storage import CSVSink, InMemoryStorage


@pytest.fixture
def run_in_tmp_dir(tmp_path):
    """Run a test inside a temporary working directory."""
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        yield tmp_path
    finally:
        os.chdir(old_cwd)


def write_rgrc(directory, content: str) -> None:
    (directory / "rgrc.py").write_text(content)


class RecordingSource(Source):
    """Source that records query arguments."""

    def __init__(self, trials=()):
        self.trials = list(trials)
        self.calls = []

    def get_trials(self, *, finished=None, search_tag=None):
        self.calls.append((finished, search_tag))
        yield from self.trials

    def consume_and_tag(
        self,
        applied_tag,
        *,
        finished=None,
        search_tag=None,
        predicate=lambda _: True,
    ):
        return None


def make_trial(number):
    from uuid import uuid4

    from rungrid.experiment import StaleTrial, VarNamespace

    return StaleTrial(number, uuid4(), VarNamespace(set()))


# ---------------------------------------------------------------------------
# Basic CLI state
# ---------------------------------------------------------------------------


def test_cli_get_instance_is_cached():
    CLI.get_instance.cache_clear()
    try:
        assert CLI.get_instance() is CLI.get_instance()
    finally:
        CLI.get_instance.cache_clear()


def test_cli_get_options_defaults_to_empty_dict():
    assert CLI().get_options() == {}


def test_cli_requires_a_subcommand():
    with pytest.raises(SystemExit):
        CLI().run([])


# ---------------------------------------------------------------------------
# init subcommand
# ---------------------------------------------------------------------------


def test_cli_init_default_template(run_in_tmp_dir):
    cli = CLI()

    cli.run(["init"])

    assert (run_in_tmp_dir / "rgrc.py").is_file()
    assert (run_in_tmp_dir / "main.py").is_file()
    assert "FiveSolver" in (run_in_tmp_dir / "main.py").read_text()


def test_cli_init_into_subdirectory(run_in_tmp_dir):
    cli = CLI()

    cli.run(["init", "-d", "project"])

    assert (run_in_tmp_dir / "project" / "rgrc.py").is_file()
    assert (run_in_tmp_dir / "project" / "main.py").is_file()


def test_cli_init_mnist_template(run_in_tmp_dir):
    cli = CLI()

    cli.run(["init", "-t", "mnist"])

    assert (run_in_tmp_dir / "rgrc.py").is_file()
    assert "MNISTTuning" in (run_in_tmp_dir / "main.py").read_text()


def test_cli_init_does_not_overwrite_existing_files(run_in_tmp_dir, capsys):
    target = run_in_tmp_dir / "main.py"
    target.write_text("original content")
    cli = CLI()

    cli.run(["init"])

    assert target.read_text() == "original content"
    assert "not overwriting" in capsys.readouterr().err


def test_cli_init_rejects_unknown_template(run_in_tmp_dir):
    cli = CLI()

    with pytest.raises(SystemExit):
        cli.run(["init", "-t", "not-a-template"])


# ---------------------------------------------------------------------------
# Source and sink construction
# ---------------------------------------------------------------------------


def test_get_source_selects_empty_single_and_multi(run_in_tmp_dir):
    write_rgrc(
        run_in_tmp_dir,
        """
from rungrid.plumbing import Empty
from rungrid.plumbing.storage import InMemoryStorage
src1 = InMemoryStorage()
src2 = InMemoryStorage()
empty = Empty()
snk1 = InMemoryStorage()
snk2 = InMemoryStorage()
""",
    )
    cli = CLI()

    assert isinstance(
        cli._get_source(SimpleNamespace(source=None, source_sink=None)), Empty
    )

    single = cli._get_source(SimpleNamespace(source=["src1"], source_sink=None))
    assert isinstance(single, InMemoryStorage)

    multi = cli._get_source(SimpleNamespace(source=["src1", "src2"], source_sink=None))
    assert isinstance(multi, MultiSource)
    assert len(multi._sources) == 2


def test_get_source_includes_source_sink(run_in_tmp_dir):
    write_rgrc(
        run_in_tmp_dir,
        """
from rungrid.plumbing.storage import InMemoryStorage
src = InMemoryStorage()
both = InMemoryStorage()
""",
    )
    cli = CLI()

    source = cli._get_source(SimpleNamespace(source=["src"], source_sink=["both"]))

    assert isinstance(source, MultiSource)
    assert len(source._sources) == 2


def test_get_source_rejects_non_source_expression(run_in_tmp_dir):
    write_rgrc(run_in_tmp_dir, "value = 42\n")
    cli = CLI()

    with pytest.raises(TypeError, match="expected a Source instance"):
        cli._get_source(SimpleNamespace(source=["value"], source_sink=None))


def test_get_sink_selects_empty_single_and_multi(run_in_tmp_dir):
    write_rgrc(
        run_in_tmp_dir,
        """
from rungrid.plumbing import Empty
from rungrid.plumbing.storage import InMemoryStorage
snk1 = InMemoryStorage()
snk2 = InMemoryStorage()
empty = Empty()
""",
    )
    cli = CLI()

    assert isinstance(
        cli._get_sink(SimpleNamespace(sink=None, source_sink=None, csv_sink=None)),
        Empty,
    )

    single = cli._get_sink(
        SimpleNamespace(sink=["snk1"], source_sink=None, csv_sink=None)
    )
    assert isinstance(single, InMemoryStorage)

    multi = cli._get_sink(
        SimpleNamespace(sink=["snk1", "snk2"], source_sink=None, csv_sink=None)
    )
    assert isinstance(multi, CopyingMultiSink)
    assert len(multi._sinks) == 2


def test_get_sink_includes_csv_sink(run_in_tmp_dir):
    write_rgrc(
        run_in_tmp_dir,
        """
from rungrid.plumbing.storage import InMemoryStorage
snk = InMemoryStorage()
""",
    )
    cli = CLI()
    stream = io.StringIO()

    sink = cli._get_sink(
        SimpleNamespace(sink=["snk"], source_sink=None, csv_sink=stream)
    )

    assert isinstance(sink, CopyingMultiSink)
    assert any(isinstance(s, CSVSink) for s in sink._sinks)


def test_get_sink_rejects_non_sink_expression(run_in_tmp_dir):
    write_rgrc(run_in_tmp_dir, "value = 42\n")
    cli = CLI()

    with pytest.raises(TypeError, match="expected a Sink instance"):
        cli._get_sink(SimpleNamespace(sink=["value"], source_sink=None, csv_sink=None))


# ---------------------------------------------------------------------------
# Trial iteration helper
# ---------------------------------------------------------------------------


def test_source_trial_iter_uses_predicate_status_tag_and_limit(run_in_tmp_dir):
    write_rgrc(run_in_tmp_dir, "marker = 1\n")
    cli = CLI()
    source = RecordingSource([make_trial(1), make_trial(2), make_trial(3)])
    args = SimpleNamespace(
        predicate="trial.optuna_trial_number >= 2",
        status=None,
        tag="tag",
        limit=-1,
    )

    trials = list(cli._get_source_trial_iter(source, args, finished=False))

    assert [t.optuna_trial_number for t in trials] == [2, 3]
    assert source.calls == [(False, "tag")]


@pytest.mark.parametrize(
    "status,finished,expected",
    [
        ("finished", False, True),
        ("unfinished", True, False),
        (None, True, True),
        (None, False, False),
        (None, None, None),
    ],
)
def test_source_trial_iter_status_override(run_in_tmp_dir, status, finished, expected):
    write_rgrc(run_in_tmp_dir, "marker = 1\n")
    cli = CLI()
    source = RecordingSource()

    list(
        cli._get_source_trial_iter(
            source,
            SimpleNamespace(predicate="True", status=status, tag=None, limit=-1),
            finished=finished,
        )
    )

    assert source.calls == [(expected, None)]


def test_source_trial_iter_limit_zero_and_positive(run_in_tmp_dir):
    write_rgrc(run_in_tmp_dir, "marker = 1\n")
    cli = CLI()
    source = RecordingSource([make_trial(1), make_trial(2), make_trial(3)])

    none = list(
        cli._get_source_trial_iter(
            source,
            SimpleNamespace(predicate="True", status=None, tag=None, limit=0),
            finished=None,
        )
    )
    first_two = list(
        cli._get_source_trial_iter(
            source,
            SimpleNamespace(predicate="True", status=None, tag=None, limit=2),
            finished=None,
        )
    )

    assert none == []
    assert [t.optuna_trial_number for t in first_two] == [1, 2]


# ---------------------------------------------------------------------------
# eval subcommand
# ---------------------------------------------------------------------------


def test_eval_prints_expression_result(run_in_tmp_dir, capsys):
    write_rgrc(run_in_tmp_dir, "value = 41\n")
    cli = CLI()

    cli.run(["eval", "value + 1"])

    assert capsys.readouterr().out.strip() == "42"


def test_eval_exposes_command_line_module_metadata(run_in_tmp_dir, capsys):
    write_rgrc(run_in_tmp_dir, "")
    cli = CLI()

    cli.run(["eval", "__name__"])

    assert capsys.readouterr().out.strip() == "__main__"


def test_eval_propagates_expression_errors(run_in_tmp_dir):
    write_rgrc(run_in_tmp_dir, "")
    cli = CLI()

    with pytest.raises(ZeroDivisionError):
        cli.run(["eval", "1 / 0"])


# ---------------------------------------------------------------------------
# pump and run subcommands
# ---------------------------------------------------------------------------


def test_pump_filters_by_finished_status(run_in_tmp_dir):
    write_rgrc(
        run_in_tmp_dir,
        """
import datetime
from uuid import uuid4
from rungrid.experiment import VarNamespace, StaleTrial, TrialResult
from rungrid.plumbing.storage import InMemoryStorage

v = VarNamespace(set())
finished = StaleTrial(1, uuid4(), v).with_result(
    TrialResult.make_ok(1, datetime.datetime.now())
)
unfinished = StaleTrial(2, uuid4(), v)
src = InMemoryStorage([finished, unfinished])
snk = InMemoryStorage()
snk2 = InMemoryStorage()
""",
    )

    cli1 = CLI()
    cli1.run(["pump", "-i", "src", "-o", "snk", "-F", "finished"])
    cli2 = CLI()
    cli2.run(["pump", "-i", "src", "-o", "snk2", "-F", "unfinished"])

    rgrc1 = cli1.get_rgrc_environment()
    rgrc2 = cli2.get_rgrc_environment()
    assert [t.optuna_trial_number for t in rgrc1["snk"].trials] == [1]
    assert [t.optuna_trial_number for t in rgrc2["snk2"].trials] == [2]


def test_pump_limit(run_in_tmp_dir):
    write_rgrc(
        run_in_tmp_dir,
        """
from uuid import uuid4
from rungrid.experiment import VarNamespace, StaleTrial
from rungrid.plumbing.storage import InMemoryStorage

v = VarNamespace(set())
src = InMemoryStorage([
    StaleTrial(1, uuid4(), v),
    StaleTrial(2, uuid4(), v),
    StaleTrial(3, uuid4(), v),
])
snk = InMemoryStorage()
""",
    )

    cli = CLI()
    cli.run(["pump", "-i", "src", "-o", "snk", "-L", "2"])

    rgrc = cli.get_rgrc_environment()
    assert [t.optuna_trial_number for t in rgrc["snk"].trials] == [1, 2]


def test_run_prints_pruned_trials(run_in_tmp_dir, capsys):
    write_rgrc(
        run_in_tmp_dir,
        """
from uuid import uuid4
from rungrid.experiment import Experiment, StaleTrial, VarNamespace, step_method
from rungrid.plumbing.storage import InMemoryStorage

src = InMemoryStorage([StaleTrial(1, uuid4(), VarNamespace(set()))])

class CliPruneFeatureExperiment(Experiment):
    def version(self) -> str:
        return "cli-prune-feature-1"

    def variables(self, v, s) -> None:
        pass

    @step_method(first=True)
    def step_prune(self, trial):
        self.exit_prune("cli prune reason")
""",
    )

    CLI().run(["run", "-i", "src", "-L", "1", "CliPruneFeatureExperiment"])

    captured = capsys.readouterr()
    assert "pruned" in captured.err
    assert "cli prune reason" in captured.err


def test_run_prints_errors(run_in_tmp_dir, capsys):
    write_rgrc(
        run_in_tmp_dir,
        """
from uuid import uuid4
from rungrid.experiment import Experiment, StaleTrial, VarNamespace, step_method
from rungrid.plumbing.storage import InMemoryStorage

src = InMemoryStorage([StaleTrial(1, uuid4(), VarNamespace(set()))])

class CliErrorFeatureExperiment(Experiment):
    def version(self) -> str:
        return "cli-error-feature-1"

    def variables(self, v, s) -> None:
        pass

    @step_method(first=True)
    def step_error(self, trial):
        raise RuntimeError("cli boom")
""",
    )

    CLI().run(["run", "-i", "src", "-L", "1", "CliErrorFeatureExperiment"])

    captured = capsys.readouterr()
    assert "cli boom" in captured.err
    assert "RuntimeError" in captured.err


def test_run_rejects_unknown_class_name(run_in_tmp_dir):
    write_rgrc(run_in_tmp_dir, "")
    cli = CLI()

    with pytest.raises(KeyError):
        cli.run(["run", "NoSuchExperiment"])
