import pytest

from rungrid.warn import rungrid_warn


def test_rungrid_warn():
    """Verify that rungrid_warn properly triggers a warning with correct category and message."""
    with pytest.warns(UserWarning, match="rungrid: Test message"):
        rungrid_warn("Test message", UserWarning)
