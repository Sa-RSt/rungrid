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
    """Verify that CLI parses arguments and executes the run subcommand inside a tmp dir."""
    rgrc_content = """
from uuid import uuid4
from rungrid.experiment import Experiment, VarNamespace, step_method, StaleTrial, TrialResult
from rungrid.plumbing import Empty, Source, Sink
from rungrid.plumbing.storage import InMemoryStorage
import datetime


class CliDummyExperiment(Experiment):
    def version(self) -> str:
        return "cli-dummy-1.0"

    def variables(self, v: VarNamespace, s) -> None:
        v.param1= s.precomputed(456)

    def first_step(self):
        return self.step_start

    @step_method()
    def step_start(self, trial, **kwargs):
        return "cli-done"


v = VarNamespace(set())
my_source = InMemoryStorage([
    StaleTrial(1, uuid4(), v).with_result(TrialResult.make_ok(1, datetime.datetime.now())),
    StaleTrial(2, uuid4(), v),
    StaleTrial(3, uuid4(), v).with_result(TrialResult.make_error(Exception(), datetime.datetime.now())),
    StaleTrial(4, uuid4(), v),
])
my_sink = InMemoryStorage()
"""
    with open("rgrc.py", "w") as f:
        f.write(rgrc_content)

    cli = CLI()
    cli.run(["run", "-i", "my_source", "-o", "my_sink", "CliDummyExperiment"])

    from rungrid.experiment import ExperimentRegistry

    registry = ExperimentRegistry.get_instance()
    exp = registry.get_by_name("CliDummyExperiment")
    assert exp is not None
    assert exp.name == "CliDummyExperiment"

    rgrc = cli._get_rgrc_environment()
    sink_instance = rgrc["my_sink"]
    assert [x.optuna_trial_number for x in sink_instance.trials] == [2, 4]


def test_cli_pump_subcommand(run_in_tmp_dir):
    """Verify that the pump command-line subcommand transfers filtered/unfiltered trials from source to sink."""
    rgrc_content = """
from uuid import uuid4
from rungrid.experiment import VarNamespace, StaleTrial, TrialResult
from rungrid.plumbing import Source, Sink
from rungrid.plumbing.storage import InMemoryStorage
import datetime

v = VarNamespace(set())
trial_ok_result = StaleTrial(1, uuid4(), v, tags={"pass"}).with_result(TrialResult.make_ok(1, datetime.datetime.now()))
trial_ok_noresult = StaleTrial(2, uuid4(), v, tags={"pass"})
trial_fail = StaleTrial(3, uuid4(), v, tags={"fail"})

src = InMemoryStorage([trial_ok_result, trial_ok_noresult, trial_fail])
snk = InMemoryStorage()
"""
    with open("rgrc.py", "w") as f:
        f.write(rgrc_content)

    cli = CLI()
    # Pump only trials tagged with "pass"
    cli.run(["pump", "-i", "src", "-o", "snk", "-T", "pass"])

    # Load rgrc environment to verify the simple sink has trials
    rgrc = cli._get_rgrc_environment()
    sink_instance = rgrc["snk"]
    assert len(sink_instance.trials) == 2
    assert all("pass" in t.tags for t in sink_instance.trials)


def test_cli_csv_sink(run_in_tmp_dir):
    """Verify that CLI can log output to a CSV sink using the -C option."""
    rgrc_content = """
from uuid import uuid4
from rungrid.experiment import VarNamespace, StaleTrial, TrialResult
from rungrid.plumbing import Source
from rungrid.plumbing.storage import InMemoryStorage
import datetime


v = VarNamespace(set())
t1 = StaleTrial(1, uuid4(), v)
t1_with_res = t1.with_result(TrialResult.make_ok(0.999, datetime.datetime.now()))

src = InMemoryStorage([t1_with_res])
"""
    with open("rgrc.py", "w") as f:
        f.write(rgrc_content)

    cli = CLI()
    cli.run(["pump", "-i", "src", "-C", "results.csv"])

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
from rungrid.plumbing.storage import InMemoryStorage


my_sink = InMemoryStorage()

def my_study_factory(exp_id):
    return optuna.create_study(study_name=exp_id, direction="minimize")

class OptunaCliExperiment(Experiment):
    def version(self) -> str:
        return "optuna-cli-1.0"

    def variables(self, v: VarNamespace, s) -> None:
        v.val= s.uniform(0.0, 10.0)

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
            "run",
            "-o",
            "my_sink",
            "-L",
            "1",
            "-M",
            "my_study_factory",
            "OptunaCliExperiment",
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
    cli.run(["run", "-i", "my_source", "-o", "my_sink", "my_scheduler"])

    rgrc = cli._get_rgrc_environment()
    assert "my_scheduler" in rgrc


def test_cli_jobs_configuration_multiple(run_in_tmp_dir):
    """Verify that CLI sets correct number of jobs in LocalDispatcher and creates multiple worker jobs."""
    from rungrid.experiment import ExperimentRegistry

    ExperimentRegistry.get_instance()._experiments_by_name.clear()
    ExperimentRegistry.get_instance()._experiments_by_ident.clear()

    rgrc_content = """
from rungrid.experiment import Experiment, VarNamespace, step_method
from rungrid.dispatch import LocalDispatcher
from rungrid.plumbing import Empty

my_source = Empty()
my_sink = Empty()

class JobsMultipleCliExperiment(Experiment):
    def version(self) -> str:
        return "sched-cli-1.0"

    def variables(self, v: VarNamespace, s) -> None:
        v.x = 1

    def first_step(self):
        return self.step_one

    @step_method()
    def step_one(self, trial, **kwargs):
        return "done"

my_scheduler = LocalDispatcher.make_default(JobsMultipleCliExperiment(), ".rg-cache-test")
"""
    with open("rgrc.py", "w") as f:
        f.write(rgrc_content)

    cli = CLI()
    # Run with 2 jobs
    cli.run(["run", "-i", "my_source", "-o", "my_sink", "--jobs", "2", "my_scheduler"])

    rgrc = cli._get_rgrc_environment()
    assert "my_scheduler" in rgrc
    sched = rgrc["my_scheduler"]
    assert sched._job.n_jobs == 2
    # Ensure LokyBackend (or similar parallel backend) is used for parallel jobs
    assert "Sequential" not in type(sched._job._backend).__name__


def test_cli_jobs_configuration_one(run_in_tmp_dir):
    """Verify that CLI sets n_jobs=1 in LocalDispatcher and uses SequentialBackend (no subprocesses)."""
    from rungrid.experiment import ExperimentRegistry

    ExperimentRegistry.get_instance()._experiments_by_name.clear()
    ExperimentRegistry.get_instance()._experiments_by_ident.clear()

    rgrc_content = """
from rungrid.experiment import Experiment, VarNamespace, step_method
from rungrid.dispatch import LocalDispatcher
from rungrid.plumbing import Empty

my_source = Empty()
my_sink = Empty()

class JobsOneCliExperiment(Experiment):
    def version(self) -> str:
        return "sched-cli-1.0"

    def variables(self, v: VarNamespace, s) -> None:
        v.x = 1

    def first_step(self):
        return self.step_one

    @step_method()
    def step_one(self, trial, **kwargs):
        return "done"

my_scheduler = LocalDispatcher.make_default(JobsOneCliExperiment(), ".rg-cache-test")
"""
    with open("rgrc.py", "w") as f:
        f.write(rgrc_content)

    cli = CLI()
    # Run with 1 job (sequential, no subprocesses)
    cli.run(["run", "-i", "my_source", "-o", "my_sink", "--jobs", "1", "my_scheduler"])

    rgrc = cli._get_rgrc_environment()
    assert "my_scheduler" in rgrc
    sched = rgrc["my_scheduler"]
    assert sched._job.n_jobs == 1
    # Ensure SequentialBackend is used, which avoids any subprocess overhead/creation
    assert type(sched._job._backend).__name__ == "SequentialBackend"


def test_cli_pump_with_predicate(run_in_tmp_dir):
    """Verify that the CLI pump subcommand filters trials using a custom python expression via --predicate."""
    rgrc_content = """
from uuid import uuid4
from rungrid.experiment import VarNamespace, StaleTrial, TrialResult
from rungrid.plumbing import Source, Sink
from rungrid.plumbing.storage import InMemoryStorage



v = VarNamespace(set())
trial_1 = StaleTrial(1, uuid4(), v)
trial_2 = StaleTrial(2, uuid4(), v)

src = InMemoryStorage([trial_1, trial_2])
snk1 = InMemoryStorage()
snk2 = InMemoryStorage()
"""
    with open("rgrc.py", "w") as f:
        f.write(rgrc_content)

    cli1 = CLI()
    # Pump only trials with optuna_trial_number == 1
    cli1.run(
        ["pump", "-i", "src", "-o", "snk1", "-P", "trial.optuna_trial_number == 1"]
    )

    cli2 = CLI()
    # Pump with a predicate that matches none
    cli2.run(
        ["pump", "-i", "src", "-o", "snk2", "-P", "trial.optuna_trial_number == 42"]
    )

    rgrc1 = cli1._get_rgrc_environment()
    rgrc2 = cli2._get_rgrc_environment()
    assert len(rgrc1["snk1"].trials) == 1
    assert rgrc1["snk1"].trials[0].optuna_trial_number == 1
    assert len(rgrc2["snk2"].trials) == 0
