"""Optuna integration for sampling and optimizing experiment trials."""

from collections.abc import Callable, Iterable
from uuid import uuid4

import optuna

from rungrid.experiment import (
    Experiment,
    LiveTrial,
    Sampler,
    StaleTrial,
    Trial,
    VarNamespace,
)
from rungrid.plumbing import PredicateType, Sink, Source


class OptimizingSampler(Source, Sink):
    """A dual Source and Sink implementation that integrates Optuna parameter tuning.

    Generates trials using parameter suggestion from an Optuna study, and reports back
    completed trial results to the study for sequential model-based optimization.
    """

    def __init__(
        self,
        experiment: Experiment,
        study: optuna.Study,
        sampler: None | Callable[[VarNamespace, Sampler], None] = None,
    ) -> None:
        """Initialize the OptimizingSampler.

        :param experiment: The experiment instance that defines parameters.
        :type experiment: Experiment
        :param study: The Optuna Study used to guide sampling and track results.
        :type study: optuna.Study
        :param sampler: Another callable to use instead of `Experiment.variables()` to
        populate the experiment's variables.
        :type sampler: None | Callable[[VarNamespace, Sampler], None]
        """
        super().__init__()
        self._experiment = experiment
        self._study = study
        if sampler is None:
            self._sampler = experiment.variables
        else:
            self._sampler = sampler

    def get_trials(
        self, *, finished: bool | None = None, search_tag: str | None = None
    ) -> Iterable[Trial]:
        """Generate/yield endless candidate trials suggested by Optuna.

        Does nothing if finished==True or search_tag is not None because
        new trials can't possibly be finished or have tags already attributed
        to them.

        :param finished: If True or False, filters trials by finished state.
        :type finished: bool | None
        :param search_tag: Optional tag filter to apply.
        :type search_tag: str | None
        :return: An iterable of newly suggested trial instances.
        :rtype: collections.abc.Iterable[Trial]
        """
        if finished or search_tag is not None:
            return
        while 1:
            trial_uuid = uuid4()
            s = Sampler(trial_uuid, self._study)
            v = s._make_var_namespace()
            self._sampler(v, s)
            s.dispose()
            if s.has_asked_optuna():
                yield LiveTrial(s.get_optuna_trial(), trial_uuid, v)
            else:
                yield StaleTrial(None, trial_uuid, v)

    def consume_and_tag(
        self,
        applied_tag: str,
        *,
        finished: bool | None = None,
        search_tag: str | None = None,
        predicate: PredicateType = lambda _: True,
    ) -> Trial | None:
        """Find, tag, and return a trial matching the given predicate.

        :param applied_tag: The tag to append to the found trial.
        :type applied_tag: str
        :param finished: Filters trials by finished state.
        :type finished: bool | None
        :param search_tag: Optional tag filter to apply before selection.
        :type search_tag: str | None
        :param predicate: A filter function to test trials.
        :type predicate: PredicateType
        :return: The tagged trial, or None if no match is found.
        :rtype: Trial | None
        """
        for t in self.get_trials(finished=finished, search_tag=search_tag):
            if predicate(t):
                t.add_tag(applied_tag)
                return t
        return None

    def put_trial(self, trial: Trial) -> None:
        """Report a finished trial result back to the Optuna study.

        :param trial: The trial containing results to report.
        :type trial: Trial
        """
        if trial.optuna_trial_number is None or not trial.is_finished():
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
