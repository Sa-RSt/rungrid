"""Command-line interface and parsing utilities for running rungrid experiments."""

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any, Iterable

import optuna
from tqdm import tqdm

from rungrid.dispatch import LocalDispatcher, Scheduler
from rungrid.experiment import Experiment, Trial
from rungrid.optimize import OptimizingSampler
from rungrid.plumbing import CopyingMultiSink, Empty, MultiSource, Sink, Source
from rungrid.plumbing.storage import CSVSink

_RGRC_PY = "rgrc.py"


class CLI(argparse.ArgumentParser):
    """Command-line interface runner and argument parser for configuring and executing experiments."""

    def _get_rgrc_environment(self) -> dict[str, Any]:
        rgrc = getattr(self, "_rgrc", None)
        if rgrc is None:
            with open(_RGRC_PY, "r") as file:
                content = file.read()
            code = compile(content, _RGRC_PY, "exec")
            rgrc = {"__file__": str(Path(_RGRC_PY).absolute()), "__name__": "rgrc"}
            exec(code, rgrc)
            setattr(self, "_rgrc", rgrc)
        return rgrc

    def _get_source(self, args, *extra_sources: Source) -> Source:
        rgrc = self._get_rgrc_environment()
        sources_code: list[str] = (args.source or []) + (args.source_sink or [])
        sources = [eval(x, rgrc) for x in sources_code]
        for expr, source in zip(sources_code, sources):
            if not isinstance(source, Source):
                raise TypeError(
                    f"command line expression {expr!r} returned {source!r}; expected a Source instance"
                )
        sources.extend(extra_sources)
        if len(sources) == 0:
            return Empty()
        elif len(sources) == 1:
            return sources[0]
        else:
            return MultiSource(sources)

    def _get_sink(self, args, *extra_sinks: Sink) -> Sink:
        rgrc = self._get_rgrc_environment()
        sinks_code: list[str] = (args.sink or []) + (args.source_sink or [])
        sinks = [eval(x, rgrc) for x in sinks_code]
        for expr, sink in zip(sinks_code, sinks):
            if not isinstance(sink, Sink):
                raise TypeError(
                    f"command line expression {expr!r} returned {sink!r}; expected a Sink instance"
                )
        sinks.extend(extra_sinks)
        if args.csv_sink is not None:
            sinks.append(CSVSink(args.csv_sink))
        if len(sinks) == 0:
            return Empty()
        elif len(sinks) == 1:
            return sinks[0]
        else:
            return CopyingMultiSink(sinks)

    def _get_source_trial_iter(self, source: Source, args) -> Iterable[Trial]:
        rgrc = self._get_rgrc_environment()
        predicate_code = compile(args.predicate, "<string>", "eval")
        status_to_finished = {"finished": True, "unfinished": False, None: None}

        def command_line_predicate(t: Trial) -> bool:
            return eval(predicate_code, rgrc, {"trial": t})

        return tqdm(
            filter(
                command_line_predicate,
                source.get_trials(
                    finished=status_to_finished[args.status], search_tag=args.tag
                ),
            )
        )

    def _subcommand_run(self, args):
        rgrc = self._get_rgrc_environment()
        study_factory: Callable[[str], optuna.Study] | None = rgrc.get(
            args.study_factory, None
        )
        experiment_class_or_sched: type[Experiment] | Scheduler = rgrc[args.class_name]
        if isinstance(experiment_class_or_sched, Scheduler):
            sched = experiment_class_or_sched
            experiment = sched.get_experiment()
        else:
            experiment = experiment_class_or_sched()
            sched = LocalDispatcher.make_default(experiment, args.cache_dir)
        if study_factory is None:
            source = self._get_source(args)
            sink = self._get_sink(args)
        else:
            study = study_factory(experiment.identifier)
            sampler = OptimizingSampler(experiment, study)
            source = self._get_source(args, sampler)
            sink = self._get_sink(args, sampler)
        it = self._get_source_trial_iter(source, args)
        for trial in sched.schedule(it):
            sink.put_trial(trial)

    def _subcommand_pump(self, args):
        source = self._get_source(args)
        sink = self._get_sink(args)
        for trial in self._get_source_trial_iter(source, args):
            sink.put_trial(trial)

    def run(self, args):
        """Parse command-line arguments and run the specified rungrid subcommand.

        :param args: The list of command-line arguments to parse.
        :type args: list[str]
        """
        self.add_rungrid_args()
        ns = self.parse_args(args)
        ns.rg_subcommand_fn(ns)

    def add_rungrid_args(self) -> None:
        """Add predefined rungrid arguments and subparsers to the parser."""
        rgrc_help = f"(must be declared or imported in {_RGRC_PY})"
        rgrc_ctx_help = f"(will be run in the context of {_RGRC_PY})"
        subparsers = self.add_subparsers(
            help="rungrid predefined subcommands", required=True
        )
        self.add_argument(
            "-I",
            "--source",
            action="append",
            help="use the given Python expression as a source for trials "
            + rgrc_ctx_help
            + ", can be specified multiple times to create a MultiSource",
        )
        self.add_argument(
            "-O",
            "--sink",
            action="append",
            help="use the given Python expression as a sink for trials "
            + rgrc_ctx_help
            + ", can be specified multiple times to create a CopyingMultiSink",
        )
        self.add_argument(
            "-S",
            "--source-sink",
            action="append",
            help="equivalent to --source X --sink X",
        )
        self.add_argument(
            "-C",
            "--csv-sink",
            type=argparse.FileType("w"),
            required=False,
            default=None,
            help="put trials in the given file using CSVSink;"
            + " implies the creation of a CopyingMultiSink if used alongside --sink",
        )
        self.add_argument(
            "-F",
            "--status",
            choices=["finished", "unfinished"],
            help="read only finished/unfinished trials",
            default=None,
        )
        self.add_argument(
            "-T",
            "--tag",
            type=str,
            help="read only trials that have a given tag",
            default=None,
        )
        self.add_argument(
            "-P",
            "--predicate",
            type=str,
            help="use a Python expression as a predicate to filter read trials; the name `trial` is"
            + " available to get the trial, alongside the names declared in rgrc.py",
            default="True",
        )
        run = subparsers.add_parser(
            "rg-run",
            help="run an experiment",
        )
        run.add_argument(
            "class_name",
            metavar="class-name-or-scheduler",
            type=str,
            help="the name of the subclass of Experiment OR the Scheduler instance to use "
            + rgrc_help,
        )
        run.add_argument(
            "-A",
            "--cache-dir",
            type=str,
            help="the directory in which to store cache of step results",
            default=".rg-cache",
        )
        run.add_argument(
            "-M",
            "--study-factory",
            type=str,
            required=False,
            help='name of the Optuna Study factory function to use, such as "study_factory" (default: don\'t use Optuna) '
            + rgrc_help,
            default=None,
        )
        run.set_defaults(rg_subcommand_fn=self._subcommand_run)
        pump = subparsers.add_parser(
            "rg-pump",
            help="pump trials from sources to sinks (can be used for exporting)",
        )
        pump.set_defaults(rg_subcommand_fn=self._subcommand_pump)
