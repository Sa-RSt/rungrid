from pathlib import Path

import optuna.storages.journal
import torch

from rungrid.plumbing import ResultPodiumSink, StateSink
from rungrid.plumbing.storage import BucketFileStorage

from main import MNISTTuning

here = Path(__file__).parent
file_store = BucketFileStorage(here / ".rg-finished", here / ".rg-unfinished")
get_best_model = StateSink(
    step_name="step_score",
    arg_name="model",
    dest=lambda _, model: torch.save(model, "best.pkl"),
    last_only=True,
)
podium = ResultPodiumSink(
    direction="maximize",
    top_k=1,
    on_winner_update=lambda L: get_best_model.put_trial(L[0]),
)


def study_factory(name: str) -> optuna.Study:
    name_alnum = "".join(x for x in name if x.isalnum())
    journal_file = here / f"optuna_journal_{name_alnum}.log"
    backend = optuna.storages.journal.JournalFileBackend(str(journal_file.resolve()))
    return optuna.create_study(
        study_name=name,
        storage=optuna.storages.JournalStorage(backend),
        load_if_exists=True,
        direction="maximize",
    )
