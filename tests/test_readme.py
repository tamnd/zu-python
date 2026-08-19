"""The README against a Python that runs it.

A quickstart is the most read and least executed code a client has. It
is copied by hand out of a page, and it goes wrong quietly, a rename or
a renamed keyword at a time, until somebody's first five minutes are
spent on a traceback. So the blocks that are whole programs are run
here, as printed, character for character.

A block is a whole program when it starts with an import, which is the
rule the README follows: a block that stands on its own opens with what
it imports, and a block that shows one call in the middle of a session
opens with the call. Each program runs in an interpreter of its own
with a temporary directory as its working directory, because the file
it writes is the one a reader would find beside them afterwards.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

README = Path(__file__).resolve().parent.parent / "README.md"


def blocks(language: str) -> list[str]:
    """Every fenced block of one language, in the order they appear."""
    found: list[str] = []
    current: list[str] | None = None
    for line in README.read_text(encoding="utf-8").splitlines():
        if current is None:
            if line.rstrip() == f"```{language}":
                current = []
        elif line.rstrip() == "```":
            found.append("\n".join(current) + "\n")
            current = None
        else:
            current.append(line)
    assert current is None, "a fenced block the README never closes"
    return found


def programs() -> list[str]:
    """The blocks that are whole programs."""
    return [block for block in blocks("python") if block.startswith("import ")]


def program_with(*words: str) -> str:
    """The one whole program that has all of `words` in it.

    By what it contains rather than by where it sits, because a page
    that gains a snippet should not renumber the tests below it.
    """
    found = [p for p in programs() if all(word in p for word in words)]
    assert len(found) == 1, f"one program with {words}, found {len(found)}"
    return found[0]


def run(program: str, where: Path) -> subprocess.CompletedProcess[str]:
    """A program, in an interpreter of its own, in `where`."""
    return subprocess.run(
        [sys.executable, "-c", program],
        cwd=where,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_the_readme_prints_programs_and_not_fragments() -> None:
    """The rule above holds: the page has both kinds and knows which."""
    assert len(programs()) == 5, "the README's whole programs"
    assert len(blocks("python")) > len(programs()), "and its fragments"


def test_the_sixty_second_snippet_runs_as_printed(tmp_path: Path) -> None:
    """Connect, write two rows, read them as a frame."""
    snippet = program_with("zudb.connect", "to_pandas")
    done = run(snippet, tmp_path)
    assert done.returncode == 0, done.stderr
    # The frame pandas prints, whatever pandas decides to pad it with.
    printed = done.stdout.split()
    assert printed[:2] == ["name", "uid"]
    assert "ada" in printed and "grace" in printed
    # A reader runs it in the directory they are standing in, and the
    # database is there when it finishes.
    assert (tmp_path / "social.zu1").is_file()


def test_the_event_loop_snippet_runs_as_printed(tmp_path: Path) -> None:
    """The same two calls, awaited."""
    snippet = program_with("zudb.aio.connect", "asyncio.run")
    done = run(snippet, tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["ada"]


def test_the_dbapi_snippet_runs_as_printed(tmp_path: Path) -> None:
    """A cursor, a `?` parameter, a block that commits."""
    snippet = program_with("zudb.dbapi.connect", "fetchall")
    done = run(snippet, tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[('ada',)]"


@pytest.mark.parametrize("index", range(len(programs())))
def test_every_whole_program_in_the_readme_runs(index: int, tmp_path: Path) -> None:
    """Including the ones no other test looks at the output of."""
    done = run(programs()[index], tmp_path)
    assert done.returncode == 0, done.stderr


def test_the_snippet_with_no_path_leaves_the_directory_empty(tmp_path: Path) -> None:
    """The claim the section makes is the one worth testing: a reader
    who runs it finds nothing beside them afterwards."""
    snippet = program_with("zudb.connect()")
    done = run(snippet, tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[('ada',)]"
    assert list(tmp_path.iterdir()) == []
