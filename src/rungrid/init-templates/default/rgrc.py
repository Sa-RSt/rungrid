# Sample rgrc.py file from rungrid template
from pathlib import Path

import optuna.storages.journal
from rungrid.plumbing.storage import BucketFileStorage

from main import FiveSolver


here = Path(__file__).parent
file_store = BucketFileStorage(here / ".rg-finished", here / ".rg-unfinished")


def study_factory(name: str) -> optuna.Study:
    journal_file = here / "optuna_journal.log"
    backend = optuna.storages.journal.JournalFileBackend(str(journal_file.resolve()))
    return optuna.create_study(
        study_name=name,
        storage=optuna.storages.JournalStorage(backend),
        load_if_exists=True,
        direction="minimize",
    )
