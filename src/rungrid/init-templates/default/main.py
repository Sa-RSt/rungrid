# Sample main.py file from rungrid template
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
