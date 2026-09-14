"""Utilities for generating unique version strings for experiments."""

from collections.abc import Callable
import datetime
import inspect
import os
import re
import subprocess
from pathlib import Path
from typing import NoReturn

from rungrid.error import SubprocessBehaviorError

_STRIP_PATH_CHARS = re.compile(r"[^a-zA-Z0-9_.-]")
_GIT_COMMIT_HASH = re.compile(r"commit\s*([0-9a-z]+)", re.IGNORECASE)
_NEGATIVE_INFINITY = -float("inf")


def _get_max_mtime(dir: Path, flt: Callable) -> float:
    maximum_yet = _NEGATIVE_INFINITY
    for entry in dir.iterdir():
        if entry.is_dir():
            maximum_yet = max(maximum_yet, _get_max_mtime(entry, flt))
        elif flt(entry.name):
            change_dt = max(
                entry.stat().st_mtime,
                entry.stat(follow_symlinks=False).st_mtime,
            )
            maximum_yet = max(maximum_yet, change_dt)
    return maximum_yet


def version_from_most_recent_mtime(
    klass: type,
    *,
    base_dir: str | os.PathLike | None = None,
    only_file_names_matching: Callable | re.Pattern = re.compile(
        r".*\.py.?$", re.IGNORECASE
    ),
) -> str:
    """Generate a version string based on the most recent file modification time in a directory.

    Traverses the base directory, finds files matching the given pattern, and determines the
    latest modification time. Converts this time into an ISO format string.

    :param klass: The class whose file path will be used as a default base directory.
    :type klass: type
    :param base_dir: Path to the directory to traverse. If None, derived from klass.
    :type base_dir: str | os.PathLike | None
    :param only_file_names_matching: Pattern or function to filter file names.
    :type only_file_names_matching: collections.abc.Callable | re.Pattern
    :return: A version string derived from the maximum modification timestamp.
    :rtype: str
    :raises RuntimeError: If base_dir is None and klass is an extension type.
    :raises FileNotFoundError: If no files matching the pattern are found.
    """
    if base_dir is None:
        try:
            path_str = inspect.getfile(klass)
        except TypeError:
            raise RuntimeError(
                "deducing base directory from extension types is not "
                + "supported, please specify base_dir manually"
            )
        base_dir = Path(path_str).parent
    if isinstance(only_file_names_matching, re.Pattern):
        only_file_names_matching = only_file_names_matching.match
    mtime = _get_max_mtime(Path(base_dir).resolve(), only_file_names_matching)
    if mtime == _NEGATIVE_INFINITY:
        raise FileNotFoundError(
            f"no files matching {only_file_names_matching} were found while traversing {base_dir}"
        )
    dt = datetime.datetime.fromtimestamp(mtime)
    return _STRIP_PATH_CHARS.sub("", dt.isoformat())


def _clip_bytes(seq: bytes) -> bytes:
    if len(seq) > 35:
        return seq[:15] + b"..." + seq[-15:]
    return seq


def version_from_git_commit_hash(
    klass: type,
    *,
    repository_location: str | os.PathLike | None = None,
    git_executable: str | os.PathLike = "git",
) -> str:
    """Generate a version string using the most recent git commit hash.

    Runs `git log` in the repository location and extracts the hash of the latest commit.

    :param klass: The class whose file path will be used as a default repository location.
    :type klass: type
    :param repository_location: Path to the git repository or a sub-folder/file inside it. If None, derived from klass.
    :type repository_location: str | os.PathLike | None
    :param git_executable: Path or name of the git executable to run.
    :type git_executable: str | os.PathLike
    :return: The latest git commit hash.
    :rtype: str
    :raises RuntimeError: If repository_location is None and klass is an extension type.
    :raises SubprocessBehaviorError: If the git command fails or returns unexpected output.
    """
    command = [git_executable, "log"]

    def _raise_error(stdout: bytes, stderr: bytes, exit_code: int | None) -> NoReturn:
        msg = ["unexpected result from", repr(command)]
        if stdout:
            msg.append("stdout:")
            msg.append(_clip_bytes(stdout).decode())
        if stderr:
            msg.append("stderr:")
            msg.append(_clip_bytes(stderr).decode())
        if exit_code:
            msg.append("exit code:")
            msg.append(repr(exit_code))
        raise SubprocessBehaviorError(" ".join(msg))

    if repository_location is None:
        try:
            where = Path(inspect.getfile(klass)).resolve()
        except TypeError:
            raise RuntimeError(
                "deducing base directory from extension types is not "
                + "supported, please specify repository_location manually"
            )
    else:
        where = Path(repository_location).resolve()

    if where.is_dir():
        base_dir = where
    else:
        base_dir = where.parent
    git = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=base_dir,
    )
    stdout, stderr = git.communicate()
    if stderr or git.returncode:
        _raise_error(stdout, stderr, git.returncode)
    output_lines = stdout.decode().splitlines()
    try:
        first_output_line = next(x for x in output_lines if x.strip()).strip()
    except StopIteration:
        _raise_error(stdout, stderr, git.returncode)
    m = _GIT_COMMIT_HASH.match(first_output_line)
    if not m:
        _raise_error(stdout, stderr, git.returncode)
    return m.group(1)
