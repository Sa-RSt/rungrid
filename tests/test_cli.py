import os
import pytest
from rungrid.cli import CLI


@pytest.fixture
def run_in_tmp_dir(tmp_path):
    """Fixture to safely run tests inside a temporary directory and restore CWD."""
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        yield tmp_path
    finally:
        os.chdir(old_cwd)


def test_cli_run_subcommand(run_in_tmp_dir):
    """Verify that CLI parses arguments and executes the rg-run subcommand inside a tmp dir."""
    rgrc_content = """
from rungrid.experiment import Experiment, VarNamespace, step_method
from rungrid.plumbing import Empty

my_source = Empty()
my_sink = Empty()

class CliDummyExperiment(Experiment):
    def version(self) -> str:
        return "cli-dummy-1.0"

    def variables(self, v: VarNamespace, s) -> None:
        v.sampled("param1", s.precomputed(456))

    def first_step(self):
        return self.step_start

    @step_method()
    def step_start(self, trial, **kwargs):
        return "cli-done"
"""
    with open("rgrc.py", "w") as f:
        f.write(rgrc_content)

    cli = CLI()
    cli.run(["-I", "my_source", "-O", "my_sink", "rg-run", "CliDummyExperiment"])

    from rungrid.experiment import ExperimentRegistry

    registry = ExperimentRegistry.get_instance()
    exp = registry.get_by_name("CliDummyExperiment")
    assert exp is not None
    assert exp.name == "CliDummyExperiment"


def test_cli_pump_subcommand(run_in_tmp_dir):
    """Verify that rg-pump command-line subcommand transfers filtered/unfiltered trials from source to sink."""
    rgrc_content = """
from uuid import uuid4
from rungrid.experiment import VarNamespace, StaleTrial
from rungrid.plumbing import Source, Sink

class SimpleSource(Source):
    def __init__(self, trials):
        self.trials = list(trials)
    def get_trials(self, *, search_tag=None, **kwargs):
        if search_tag:
            return [t for t in self.trials if search_tag in t.tags]
        return self.trials
    def consume_and_tag(self, applied_tag, **kwargs):
        pass

class SimpleSink(Sink):
    def __init__(self):
        self.trials = []
    def put_trial(self, trial):
        self.trials.append(trial)

v = VarNamespace(set())
trial_ok = StaleTrial(1, uuid4(), v, tags={"pass"})
trial_fail = StaleTrial(2, uuid4(), v, tags={"fail"})

src = SimpleSource([trial_ok, trial_fail])
snk = SimpleSink()
"""
    with open("rgrc.py", "w") as f:
        f.write(rgrc_content)

    cli = CLI()
    # Pump only trials tagged with "pass"
    cli.run(["-I", "src", "-O", "snk", "-T", "pass", "rg-pump"])

    # Load rgrc environment to verify the simple sink has trials
    rgrc = cli._get_rgrc_environment()
    sink_instance = rgrc["snk"]
    assert len(sink_instance.trials) == 1
    assert "pass" in sink_instance.trials[0].tags


def test_cli_csv_sink(run_in_tmp_dir):
    """Verify that CLI can log output to a CSV sink using the -C option."""
    rgrc_content = """
from uuid import uuid4
from rungrid.experiment import VarNamespace, StaleTrial, TrialResult
from rungrid.plumbing import Source
import datetime

class SimpleSource(Source):
    def __init__(self, trials):
        self.trials = list(trials)
    def get_trials(self, **kwargs):
        return self.trials
    def consume_and_tag(self, applied_tag, **kwargs):
        pass

v = VarNamespace(set())
t1 = StaleTrial(1, uuid4(), v)
t1_with_res = t1.with_result(TrialResult.make_ok(0.999, datetime.datetime.now()))

src = SimpleSource([t1_with_res])
"""
    with open("rgrc.py", "w") as f:
        f.write(rgrc_content)

    cli = CLI()
    cli.run(["-I", "src", "-C", "results.csv", "rg-pump"])

    assert os.path.exists("results.csv")
    with open("results.csv", "r") as f:
        content = f.read()
    assert "0.999" in content


def test_cli_run_with_optuna_limit(run_in_tmp_dir):
    """Verify CLI integration with Optuna when --study-factory and -L 1 are specified, executing 1 actual trial."""
    rgrc_content = """
import optuna
from rungrid.experiment import Experiment, VarNamespace, step_method
from rungrid.plumbing import Sink

class SimpleSink(Sink):
    def __init__(self):
        self.trials = []
    def put_trial(self, trial):
        self.trials.append(trial)

my_sink = SimpleSink()

def my_study_factory(exp_id):
    return optuna.create_study(study_name=exp_id, direction="minimize")

class OptunaCliExperiment(Experiment):
    def version(self) -> str:
        return "optuna-cli-1.0"

    def variables(self, v: VarNamespace, s) -> None:
        v.sampled("val", s.uniform(0.0, 10.0))

    def first_step(self):
        return self.step_one

    @step_method()
    def step_one(self, trial, **kwargs):
        return 4.5
"""
    with open("rgrc.py", "w") as f:
        f.write(rgrc_content)

    cli = CLI()
    # Run with -L 1 limit
    cli.run(
        [
            "-O",
            "my_sink",
            "-L",
            "1",
            "rg-run",
            "OptunaCliExperiment",
            "-M",
            "my_study_factory",
        ]
    )

    # Verify that the experiment registry has registered it and it ran successfully without hanging
    from rungrid.experiment import ExperimentRegistry

    registry = ExperimentRegistry.get_instance()
    exp = registry.get_by_name("OptunaCliExperiment")
    assert exp is not None

    # Load rgrc environment to verify trial output
    rgrc = cli._get_rgrc_environment()
    sink_instance = rgrc["my_sink"]
    assert len(sink_instance.trials) == 1
    assert sink_instance.trials[0].result.result == 4.5


def test_cli_scheduler_instance(run_in_tmp_dir):
    """Verify CLI accepts a registered scheduler instance instead of a class name."""
    rgrc_content = """
from rungrid.experiment import Experiment, VarNamespace, step_method
from rungrid.dispatch import LocalDispatcher
from rungrid.plumbing import Empty

my_source = Empty()
my_sink = Empty()

class SchedCliExperiment(Experiment):
    def version(self) -> str:
        return "sched-cli-1.0"

    def variables(self, v: VarNamespace, s) -> None:
        v.x = 1

    def first_step(self):
        return self.step_one

    @step_method()
    def step_one(self, trial, **kwargs):
        return "done"

my_scheduler = LocalDispatcher.make_default(SchedCliExperiment(), ".rg-cache-test")
"""
    with open("rgrc.py", "w") as f:
        f.write(rgrc_content)

    cli = CLI()
    cli.run(["-I", "my_source", "-O", "my_sink", "rg-run", "my_scheduler"])

    rgrc = cli._get_rgrc_environment()
    assert "my_scheduler" in rgrc
