"""Tests for the top-level ``rungrid.main`` entry point."""

import sys

import rungrid
from rungrid.cli import CLI


def test_main_invokes_cli_with_command_line_arguments(monkeypatch):
    calls = []

    monkeypatch.setattr(CLI, "run", lambda self, args: calls.append(args))
    monkeypatch.setattr(sys, "argv", ["rungrid", "eval", "1 + 1"])

    rungrid.main()

    assert calls == [["eval", "1 + 1"]]


def test_main_uses_the_cli_singleton(monkeypatch):
    class FakeCLI:
        def __init__(self):
            self.calls = []

        def run(self, args):
            self.calls.append(args)

    fake = FakeCLI()
    monkeypatch.setattr(CLI, "get_instance", classmethod(lambda cls: fake))
    monkeypatch.setattr(sys, "argv", ["rungrid", "pump", "-i", "src"])

    rungrid.main()

    assert fake.calls == [["pump", "-i", "src"]]


def test_main_slices_off_the_program_name(monkeypatch):
    calls = []

    monkeypatch.setattr(CLI, "run", lambda self, args: calls.append(args))
    monkeypatch.setattr(sys, "argv", ["/path/to/rungrid", "--help"])

    rungrid.main()

    assert calls == [["--help"]]
