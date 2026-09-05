from collections.abc import Iterable
from uuid import uuid4

import optuna

from rungrid.experiment import Experiment, LiveTrial, Sampler, Trial, VarNamespace
from rungrid.plumbing import Sink, Source, PredicateType


class OptimizingSampler(Source, Sink):
    def __init__(self, experiment: Experiment, study: optuna.Study) -> None:
        super().__init__()
        self._experiment = experiment
        self._study = study

    def get_trials(
        self, *, finished: bool | None = None, search_tag: str | None = None
    ) -> Iterable[Trial]:
        if finished or search_tag is not None:
            return
        while 1:
            optuna_trial = self._study.ask()
            s = Sampler(optuna_trial)
            v = VarNamespace(s._variable_names)
            self._experiment.variables(v, s)
            yield LiveTrial(optuna_trial, uuid4(), v)

    def consume_and_tag(
        self,
        applied_tag: str,
        *,
        finished: bool | None = None,
        search_tag: str | None = None,
        predicate: PredicateType = lambda _: True,
    ) -> Trial | None:
        for t in self.get_trials(finished=finished, search_tag=search_tag):
            if predicate(t):
                t.add_tag(applied_tag)
                return t
        return None

    def put_trial(self, trial: Trial) -> None:
        if not trial.is_finished():
            return
        assert trial.result is not None

        res = trial.result
        if res.is_error:
            self._study.tell(
                trial.optuna_trial_number, None, optuna.trial.TrialState.FAIL
            )
        elif res.is_pruned:
            self._study.tell(
                trial.optuna_trial_number, None, optuna.trial.TrialState.PRUNED
            )
        else:
            self._study.tell(trial.optuna_trial_number, res.result)
