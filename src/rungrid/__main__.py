"""Command-line entry point to generate sample template files for rungrid."""

import shutil
import sys
from pathlib import Path

SAMPLE_RGRC = """# Sample rgrc.py file from rungrid template
from pathlib import Path

import optuna.storages.journal
from rungrid.plumbing.storage import BucketFileStorage

from {main_module_name} import FiveSolver


here = Path(__file__).parent
file_store = BucketFileStorage(here / '.rg-finished', here / '.rg-unfinished')


def study_factory(name: str) -> optuna.Study:
    journal_file = here / "optuna_journal.log"
    backend = optuna.storages.journal.JournalFileBackend(str(journal_file.resolve()))
    return optuna.create_study(
        study_name=name,
        storage=optuna.storages.JournalStorage(backend),
        load_if_exists=True,
        direction="minimize",
    )
"""


SAMPLE_MAIN = """# Sample main.py file from rungrid template
from collections.abc import Callable
from math import tan, pi

from rungrid.experiment import Experiment, Trial, VarNamespace, Sampler, step_method
from rungrid.version import version_from_git_commit_hash


class FiveSolver(Experiment):
    def variables(self, v: VarNamespace, s: Sampler) -> None:
        v.sampled("x", s.transformed(s.uniform(-pi / 2 + 1e-6, pi / 2 - 1e-6), tan))

    def version(self) -> str:
        return version_from_git_commit_hash(type(self))

    def first_step(self) -> Callable:
        return self.calculate_five

    @step_method()
    def calculate_five(self, trial: Trial):
        res = (trial.v.x - 5) ** 2
        self.exit_done(res)
"""


def _get_new_path(fmt: str) -> Path:
    plain = fmt.format(extra="")
    if not Path(plain).exists():
        return Path(plain).resolve()

    i = 1
    p = Path(fmt.format(extra=f"_{i}")).resolve()
    while p.exists():
        i += 1
        p = Path(fmt.format(extra=f"_{i}")).resolve()
    return p


def _get_main_path() -> Path:
    return _get_new_path("main{extra}.py")


def _get_rgrc_bak_path() -> Path:
    return _get_new_path("rgrc{extra}.py.bak")


def main():
    """Execute the command-line interface to set up sample template files."""
    main_path = _get_main_path()
    rgrc_path = Path("rgrc.py").resolve()
    if rgrc_path.exists():
        bak = _get_rgrc_bak_path()
        shutil.move(rgrc_path, bak)
        print(f"Moved {rgrc_path} to {bak}", file=sys.stderr)

    main_path.write_text(SAMPLE_MAIN)
    print(f"Wrote {main_path}", file=sys.stderr)

    rgrc_path.write_text(SAMPLE_RGRC.format(main_module_name=main_path.stem))
    print(f"Wrote {rgrc_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
