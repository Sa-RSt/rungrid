"""Additional tests for version string generation utilities."""

import datetime
import inspect
import os
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rungrid.error import SubprocessBehaviorError
from rungrid.version import (
    _GIT_COMMIT_HASH,
    _NEGATIVE_INFINITY,
    _STRIP_PATH_CHARS,
    _clip_bytes,
    _get_max_mtime,
    version_from_git_commit_hash,
    version_from_most_recent_mtime,
)


class DummyClass:
    """Class used to derive default source locations."""


def test_clip_bytes_short_value_is_unchanged():
    assert _clip_bytes(b"short") == b"short"
    assert _clip_bytes(b"x" * 35) == b"x" * 35


def test_clip_bytes_long_value_is_clipped():
    clipped = _clip_bytes(b"x" * 36)

    assert clipped == b"x" * 15 + b"..." + b"x" * 15


def test_get_max_mtime_ignores_non_matching_files(tmp_path):
    matching = tmp_path / "a.py"
    ignored = tmp_path / "b.txt"
    matching.write_text("")
    ignored.write_text("")

    now = datetime.datetime.now().timestamp()
    os.utime(matching, (now - 10, now - 10))
    os.utime(ignored, (now, now))

    result = _get_max_mtime(tmp_path, lambda name: name.endswith(".py"))

    assert result == pytest.approx(now - 10)


def test_get_max_mtime_traverses_nested_directories(tmp_path):
    nested = tmp_path / "nested" / "deeper"
    nested.mkdir(parents=True)
    top = tmp_path / "top.py"
    bottom = nested / "bottom.py"
    top.write_text("")
    bottom.write_text("")

    now = datetime.datetime.now().timestamp()
    os.utime(top, (now - 100, now - 100))
    os.utime(bottom, (now - 5, now - 5))

    result = _get_max_mtime(tmp_path, lambda name: name.endswith(".py"))

    assert result == pytest.approx(now - 5)


def test_get_max_mtime_returns_negative_infinity_when_nothing_matches(tmp_path):
    assert _get_max_mtime(tmp_path, lambda name: False) == _NEGATIVE_INFINITY


def test_version_from_mtime_uses_matching_regex(tmp_path):
    py = tmp_path / "a.py"
    txt = tmp_path / "b.txt"
    py.write_text("")
    txt.write_text("")

    now = datetime.datetime.now().timestamp()
    os.utime(py, (now - 100, now - 100))
    os.utime(txt, (now - 1, now - 1))

    version = version_from_most_recent_mtime(
        DummyClass, base_dir=tmp_path, only_file_names_matching=re.compile(r".*\.txt$")
    )
    expected = _STRIP_PATH_CHARS.sub(
        "", datetime.datetime.fromtimestamp(now - 1).isoformat()
    )

    assert version == expected


def test_version_from_mtime_accepts_string_base_dir(tmp_path):
    file = tmp_path / "a.py"
    file.write_text("")

    version = version_from_most_recent_mtime(DummyClass, base_dir=str(tmp_path))

    assert isinstance(version, str)
    assert not _STRIP_PATH_CHARS.search(version)


def test_version_from_mtime_accepts_callable_filter(tmp_path):
    matching = tmp_path / "special.py"
    other = tmp_path / "other.py"
    matching.write_text("")
    other.write_text("")

    now = datetime.datetime.now().timestamp()
    os.utime(matching, (now - 50, now - 50))
    os.utime(other, (now, now))

    version = version_from_most_recent_mtime(
        DummyClass,
        base_dir=tmp_path,
        only_file_names_matching=lambda name: name == "special.py",
    )
    expected = _STRIP_PATH_CHARS.sub(
        "", datetime.datetime.fromtimestamp(now - 50).isoformat()
    )

    assert version == expected


def test_version_from_mtime_empty_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no files matching"):
        version_from_most_recent_mtime(DummyClass, base_dir=tmp_path)


def test_version_from_mtime_derives_base_dir_from_class():
    version = version_from_most_recent_mtime(DummyClass)

    assert isinstance(version, str)
    assert version


def test_version_from_mtime_extension_type_raises():
    with pytest.raises(
        RuntimeError, match="deducing base directory from extension types"
    ):
        version_from_most_recent_mtime(int)


def test_git_commit_regex_matches_commit_lines():
    match = _GIT_COMMIT_HASH.match("commit abc123XYZ")

    assert match is not None
    assert match.group(1) == "abc123XYZ"


def make_process(stdout: bytes, stderr: bytes, returncode: int) -> MagicMock:
    process = MagicMock()
    process.communicate.return_value = (stdout, stderr)
    process.returncode = returncode
    return process


@patch("subprocess.Popen")
def test_git_commit_success(mock_popen, tmp_path):
    process = make_process(b"commit deadbeef123\nAuthor: someone\n", b"", 0)
    mock_popen.return_value = process

    result = version_from_git_commit_hash(DummyClass, repository_location=tmp_path)

    assert result == "deadbeef123"
    args, kwargs = mock_popen.call_args
    assert args[0] == ["git", "log"]
    assert kwargs["cwd"] == tmp_path.resolve()
    assert kwargs["stdin"] is not None


@patch("subprocess.Popen")
def test_git_commit_success_skips_blank_lines(mock_popen, tmp_path):
    process = make_process(b"\n   \n\tcommit cafe123\n", b"", 0)
    mock_popen.return_value = process

    assert (
        version_from_git_commit_hash(DummyClass, repository_location=tmp_path)
        == "cafe123"
    )


@patch("subprocess.Popen")
def test_git_commit_uses_custom_executable(mock_popen, tmp_path):
    process = make_process(b"commit abc123\n", b"", 0)
    mock_popen.return_value = process

    version_from_git_commit_hash(
        DummyClass,
        repository_location=tmp_path,
        git_executable="/custom/git",
    )

    args, _ = mock_popen.call_args
    assert args[0] == ["/custom/git", "log"]


@patch("subprocess.Popen")
def test_git_commit_accepts_file_repository_location(mock_popen, tmp_path):
    process = make_process(b"commit abc123\n", b"", 0)
    mock_popen.return_value = process
    file = tmp_path / "some_file.py"
    file.write_text("")

    version_from_git_commit_hash(DummyClass, repository_location=file)

    _, kwargs = mock_popen.call_args
    assert kwargs["cwd"] == tmp_path.resolve()


@patch("subprocess.Popen")
def test_git_commit_derives_repository_location_from_class(mock_popen):
    process = make_process(b"commit abc123\n", b"", 0)
    mock_popen.return_value = process
    expected = Path(inspect.getfile(DummyClass)).resolve().parent

    version_from_git_commit_hash(DummyClass)

    _, kwargs = mock_popen.call_args
    assert kwargs["cwd"] == expected


def test_git_commit_extension_type_raises():
    with pytest.raises(
        RuntimeError, match="deducing base directory from extension types"
    ):
        version_from_git_commit_hash(int)


@patch("subprocess.Popen")
def test_git_commit_failure_with_stderr(mock_popen, tmp_path):
    process = make_process(b"", b"fatal: not a repository", 128)
    mock_popen.return_value = process

    with pytest.raises(SubprocessBehaviorError) as error:
        version_from_git_commit_hash(DummyClass, repository_location=tmp_path)

    message = str(error.value)
    assert "unexpected result from" in message
    assert "stderr:" in message
    assert "fatal: not a repository" in message
    assert "exit code:" in message
    assert "128" in message


@patch("subprocess.Popen")
def test_git_commit_failure_with_stdout_and_stderr(mock_popen, tmp_path):
    process = make_process(b"some stdout", b"some stderr", 1)
    mock_popen.return_value = process

    with pytest.raises(SubprocessBehaviorError) as error:
        version_from_git_commit_hash(DummyClass, repository_location=tmp_path)

    message = str(error.value)
    assert "stdout:" in message
    assert "some stdout" in message
    assert "stderr:" in message
    assert "some stderr" in message


@patch("subprocess.Popen")
def test_git_commit_failure_without_exit_code(mock_popen, tmp_path):
    process = make_process(b"", b"warning", 0)
    mock_popen.return_value = process

    with pytest.raises(SubprocessBehaviorError) as error:
        version_from_git_commit_hash(DummyClass, repository_location=tmp_path)

    message = str(error.value)
    assert "stderr:" in message
    assert "warning" in message
    assert "exit code:" not in message


@patch("subprocess.Popen")
def test_git_commit_failure_with_empty_output(mock_popen, tmp_path):
    process = make_process(b"", b"", 0)
    mock_popen.return_value = process

    with pytest.raises(SubprocessBehaviorError, match="unexpected result from"):
        version_from_git_commit_hash(DummyClass, repository_location=tmp_path)


@patch("subprocess.Popen")
def test_git_commit_failure_with_malformed_output(mock_popen, tmp_path):
    process = make_process(b"this is not a commit line\n", b"", 0)
    mock_popen.return_value = process

    with pytest.raises(SubprocessBehaviorError, match="unexpected result from"):
        version_from_git_commit_hash(DummyClass, repository_location=tmp_path)


@patch("subprocess.Popen")
def test_git_commit_failure_clips_long_stderr(mock_popen, tmp_path):
    process = make_process(b"", b"e" * 100, 1)
    mock_popen.return_value = process

    with pytest.raises(SubprocessBehaviorError) as error:
        version_from_git_commit_hash(DummyClass, repository_location=tmp_path)

    assert "..." in str(error.value)
