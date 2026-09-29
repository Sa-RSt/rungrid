"""A framework for running and managing experiments with dispatch and optimization."""


def main():
    """Run the CLI with default configuration."""
    import sys

    from rungrid.cli import CLI

    cli = CLI.get_instance()
    cli.run(sys.argv[1:])
