class RungridError(Exception):
    pass


class FatalRungridError(BaseException):
    pass


class StructureError(FatalRungridError):
    pass


class SubprocessBehaviorError(RungridError):
    pass
