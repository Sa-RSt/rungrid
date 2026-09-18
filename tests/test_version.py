import datetime
import os
import re
from unittest.mock import MagicMock, patch

import pytest

from rungrid.error import SubprocessBehaviorError
from rungrid.version import (
    version_from_git_commit_hash,
    version_from_most_recent_mtime,
)


class DummyClass:
    """A dummy class for version utility testing."""


def test_version_from_most_recent_mtime(tmp_path):
    """Test generating a version string from file modification times."""
    # Create files in tmp_path
    file1 = tmp_path / "a.py"
    file2 = tmp_path / "b.py"
    file3 = tmp_path / "c.txt"

    file1.write_text("print('hello')")
    file2.write_text("print('world')")
    file3.write_text("not a python file")

    # Set custom modification times
    now = datetime.datetime.now().timestamp()
    os.utime(file1, (now - 100, now - 100))
    os.utime(file2, (now - 50, now - 50))
    os.utime(file3, (now, now))  # c.txt has latest mtime but shouldn't match .py filter

    # Generate version based on python files (should select b.py mtime)
    version = version_from_most_recent_mtime(DummyClass, base_dir=tmp_path)
    expected_dt = datetime.datetime.fromtimestamp(now - 50)
    expected_version = re.sub(r"[^a-zA-Z0-9_.-]", "", expected_dt.isoformat())
    assert version == expected_version


def test_version_from_most_recent_mtime_not_found(tmp_path):
    """Verify FileNotFoundError is raised if no matching files are found."""
    with pytest.raises(FileNotFoundError, match="no files matching"):
        version_from_most_recent_mtime(
            DummyClass, base_dir=tmp_path, only_file_names_matching=lambda x: False
        )


@patch("subprocess.Popen")
def test_version_from_git_commit_hash_success(mock_popen):
    """Verify version_from_git_commit_hash extracts git commit hash successfully."""
    # Set up mock Popen
    mock_process = MagicMock()
    mock_process.communicate.return_value = (
        b"commit abc123xyz789\nAuthor: sasha\n",
        b"",
    )
    mock_process.returncode = 0
    mock_popen.return_value = mock_process

    # Call with DummyClass and a path
    version = version_from_git_commit_hash(DummyClass, repository_location=".")
    assert version == "abc123xyz789"


@patch("subprocess.Popen")
def test_version_from_git_commit_hash_failure(mock_popen):
    """Verify version_from_git_commit_hash raises SubprocessBehaviorError on git failure."""
    # Set up mock Popen returning error code
    mock_process = MagicMock()
    mock_process.communicate.return_value = (b"", b"fatal: not a git repository")
    mock_process.returncode = 128
    mock_popen.return_value = mock_process

    with pytest.raises(SubprocessBehaviorError, match="unexpected result from"):
        version_from_git_commit_hash(DummyClass, repository_location=".")


def test_version_mtime_extension_type_raises_runtime_error():
    """Verify calling version_from_most_recent_mtime with extension type and no base_dir raises RuntimeError."""
    with pytest.raises(RuntimeError, match="deducing base directory from extension types"):
        version_from_most_recent_mtime(int, base_dir=None)

