"""The deployment's `.env`, edited in place without losing what a human wrote in it.

An adapter for `sis.application.ports.estate.ConfigStore`. It exists because the estate's
connections are meant to be *readable*: one file, every school, visible to whoever is
holding the pager at 2am. That is worth more than the tidiness of a generated file, and
it is why this rewrites lines in place rather than regenerating the file from a mapping.

Three properties, each of which is a failure this file has to prevent:

**Comments and order survive.** `.env` here carries a lot of explanation, and a store
that round-tripped it through a dict would delete all of it the first time a school was
created. Existing keys are edited where they sit; new ones are appended under a heading.

**The write is atomic.** The file is written beside itself and moved over the original
with `os.replace`, which is atomic on both POSIX and Windows. A process killed halfway
through leaves the previous `.env` intact -- as opposed to a truncated one, which is a
service that cannot start.

**Concurrent writers are serialised.** Two managers creating schools in the same second
would otherwise both read the old `SIS_SCHOOLS`, and the second write would drop the
first school's entry while keeping its database. The lock is a file created with
`O_EXCL`, which is atomic on every platform this runs on and needs no dependency.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Final

#: Lines this rewrites: an optional `export`, a name, an `=`. Anything else -- a comment,
#: a blank, a continuation -- is copied through untouched.
_ASSIGNMENT: Final[re.Pattern[str]] = re.compile(
    r"^(?P<indent>\s*)(?P<export>export\s+)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*="
)

#: Values needing quotes in a dotenv file. A password with a `#` in it would otherwise be
#: truncated at the comment marker on the next read.
_NEEDS_QUOTING: Final[re.Pattern[str]] = re.compile(r"[\s#\"']")

_HEADING: Final[str] = (
    "\n# --- Schools provisioned by the service ---------------------------------\n"
    "# Written by sis.application.services.estate when a school is created. Each\n"
    "# school is one database on the shared server; SIS_SCHOOLS names them and one\n"
    "# SIS_DATABASE_URL_<CODE> points at each. Safe to edit by hand while the\n"
    "# service is stopped.\n"
)

_LOCK_TIMEOUT_SECONDS: Final[float] = 10.0
_LOCK_POLL_SECONDS: Final[float] = 0.05


class ConfigStoreUnavailable(RuntimeError):
    """The configuration file could not be read or written. Nothing was changed."""


def _quote(value: str) -> str:
    if not value:
        return '""'
    if _NEEDS_QUOTING.search(value):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


def _unquote(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in "\"'":
        return stripped[1:-1]
    # An unquoted value ends at the first ` #`, matching how python-dotenv reads it.
    return stripped.split(" #", 1)[0].strip()


class DotEnvConfigStore:
    """Reads and writes one `.env` file.

    `path` is passed in rather than discovered so a test writes to a temp file and the
    suite never edits the developer's real environment -- which, for a store whose whole
    job is rewriting configuration, is not a hypothetical.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._lock = self._path.with_name(self._path.name + ".lock")

    # -- reading ------------------------------------------------------------------

    def read(self) -> dict[str, str]:
        """Every assignment in the file. A missing file reads as empty, not an error."""
        if not self._path.exists():
            return {}
        values: dict[str, str] = {}
        for line in self._text().splitlines():
            if line.lstrip().startswith("#"):
                continue
            match = _ASSIGNMENT.match(line)
            if match:
                values[match.group("name")] = _unquote(line.split("=", 1)[1])
        return values

    # -- writing ------------------------------------------------------------------

    def update(self, values: dict[str, str]) -> None:
        """Set these names, leaving every other line of the file exactly as it was."""
        if not values:
            return
        with self._locked():
            original = self._text() if self._path.exists() else ""
            self._write(self._apply(original, values))

    def _apply(self, original: str, values: dict[str, str]) -> str:
        remaining = dict(values)
        lines = original.splitlines(keepends=True)
        out: list[str] = []

        for line in lines:
            match = _ASSIGNMENT.match(line)
            name = match.group("name") if match else None
            if name is not None and name in remaining:
                ending = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
                export = match.group("export") or ""
                out.append(
                    f"{match.group('indent')}{export}{name}={_quote(remaining.pop(name))}{ending}"
                )
            else:
                out.append(line)

        if remaining:
            if out and not out[-1].endswith(("\n", "\r\n")):
                out.append("\n")
            if _HEADING.strip().splitlines()[0] not in original:
                out.append(_HEADING)
            for name, value in remaining.items():
                out.append(f"{name}={_quote(value)}\n")

        return "".join(out)

    def _write(self, content: str) -> None:
        """Replace the file atomically: write a sibling, then move it over."""
        temporary = self._path.with_name(self._path.name + f".{os.getpid()}.tmp")
        try:
            temporary.write_text(content, encoding="utf-8", newline="")
            os.replace(temporary, self._path)
        except OSError as error:
            raise ConfigStoreUnavailable(
                f"could not write {self._path}: {error}"
            ) from error
        finally:
            if temporary.exists():
                temporary.unlink(missing_ok=True)

    def _text(self) -> str:
        try:
            return self._path.read_text(encoding="utf-8")
        except OSError as error:
            raise ConfigStoreUnavailable(
                f"could not read {self._path}: {error}"
            ) from error

    # -- locking ------------------------------------------------------------------

    def _locked(self):
        return _FileLock(self._lock)


class _FileLock:
    """A lock file created with `O_EXCL`, so only one writer holds it at a time.

    Chosen over `fcntl`/`msvcrt` because it behaves the same on both platforms this runs
    on, and over a library because one file and thirty lines is the whole requirement.

    A stale lock -- from a process killed while holding it -- blocks writing for
    `_LOCK_TIMEOUT_SECONDS` and then raises, naming the file. That is deliberate: the
    alternative is breaking a lock that a live process may still be holding, and the
    thing being protected is the file that decides whether the service can start.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._descriptor: int | None = None

    def __enter__(self) -> "_FileLock":
        deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
        while True:
            try:
                self._descriptor = os.open(
                    self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY
                )
                os.write(self._descriptor, str(os.getpid()).encode("ascii"))
                return self
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise ConfigStoreUnavailable(
                        f"{self._path} has been held for more than "
                        f"{_LOCK_TIMEOUT_SECONDS:.0f}s. Another provision is running, or "
                        "one was killed while holding it -- delete the file if no other "
                        "process is writing configuration."
                    ) from None
                time.sleep(_LOCK_POLL_SECONDS)

    def __exit__(self, *_exc: object) -> None:
        if self._descriptor is not None:
            os.close(self._descriptor)
        self._path.unlink(missing_ok=True)


__all__ = ["ConfigStoreUnavailable", "DotEnvConfigStore"]
