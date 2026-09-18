import pytest

from rungrid.error import (
    FatalRungridError,
    RungridError,
    StructureError,
    SubprocessBehaviorError,
)


def test_exception_instantiation():
    """Verify that exceptions can be raised and caught with their messages."""
    with pytest.raises(RungridError, match="basic error"):
        raise RungridError("basic error")

    with pytest.raises(FatalRungridError, match="fatal error"):
        raise FatalRungridError("fatal error")

    with pytest.raises(StructureError, match="structure invalid"):
        raise StructureError("structure invalid")

    with pytest.raises(SubprocessBehaviorError, match="subprocess failed"):
        raise SubprocessBehaviorError("subprocess failed")
