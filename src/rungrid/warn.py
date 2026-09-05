from warnings import warn


def rungrid_warn(msg: str, cat: type[Warning]) -> None:
    """Emit a warning from the rungrid package.

    :param msg: The warning message to display.
    :type msg: str
    :param cat: The category class of the warning.
    :type cat: type[Warning]
    """
    warn(f"rungrid: {msg}", cat)

