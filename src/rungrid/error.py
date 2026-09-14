"""Exception classes used throughout the rungrid framework."""


class RungridError(Exception):
    """Base exception class for rungrid-specific errors."""


class FatalRungridError(BaseException):
    """Base class for fatal exceptions that should halt execution immediately."""


class StructureError(FatalRungridError):
    """Raised when an experiment's structure, registration, or configuration is invalid."""


class SubprocessBehaviorError(RungridError):
    """Raised when an external subprocess exhibits unexpected behavior."""
