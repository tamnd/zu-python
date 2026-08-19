"""What a person gets, run on a machine that has nothing else on it.

    python tools/smoke.py

The suite is the thing that says this client is correct, and it needs a
compiler, a Rust toolchain, pytest and a checkout to say it. None of
those is on the machine of the person who ran `pip install zudb`, and
the failures that only that machine sees are the ones nothing else
looks for: a wheel whose extension links against a library the build
image had and the user's image does not, a file that the build put in
the tree and left out of the wheel, a stub that ships without the
package data that makes it readable, an import that works because the
checkout was on the path.

So this is written to run against an installed wheel and nothing else.
Standard library only, no pytest, no fixtures, no checkout: point an
interpreter at it in a container that holds an interpreter and a
package manager, and what it exercises is what a reader of the README
would do on their first afternoon.

It is not a second test suite and must not grow into one. Every claim
here is already covered somewhere in tests/, and it is repeated here
because the interesting variable is the machine rather than the claim.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path


def imported_from_an_install() -> None:
    """The package under test is the installed one, not a checkout.

    A smoke test that imported the source tree beside it would pass on
    a machine where the wheel had never been built, which is the one
    thing this cannot be allowed to do.
    """
    import zudb

    where = Path(zudb.__file__).resolve()
    assert "site-packages" in where.parts or "dist-packages" in where.parts, (
        f"zudb was imported from {where}, which is not an install"
    )

    # The compiled half, which is the half a wheel exists to carry, and
    # the stub beside it, which is what an editor reads.
    from zudb import _zudb

    assert Path(_zudb.__file__).parent == where.parent
    assert (where.parent / "_zudb.pyi").is_file(), "the wheel shipped no stub"
    assert (where.parent / "py.typed").is_file(), "the wheel shipped no py.typed"

    assert isinstance(zudb.__version__, str)
    assert isinstance(zudb.__abi_version__, str)


def a_graph(path: Path) -> None:
    """The quickstart, on a file, with the file reopened afterwards."""
    import zudb

    with zudb.connect(path) as conn:
        conn.execute("INSERT (p:person {uid: 1, name: 'ada'})")
        conn.execute(
            "INSERT (p:person {uid: $uid, name: $name})",
            {"uid": 2, "name": "grace"},
        )

        people = conn.execute("MATCH (p:person) RETURN p.name AS name, p.uid AS uid")
        assert people.columns == ["name", "uid"]
        assert sorted(name for name, _ in people) == ["ada", "grace"]

    # A second open of the same file, because a wheel that writes to a
    # page cache and never to a disk passes everything above.
    with zudb.connect(path, read_only=True) as conn:
        again = conn.execute("MATCH (p:person) RETURN p.name AS name")
        assert len(again) == 2


def the_bulk_path(path: Path) -> None:
    """`load`, then the appender: the two ways rows arrive in bulk."""
    import zudb

    zudb.load(
        path,
        nodes="city",
        rels="road",
        columns={"uid": [1, 2, 3], "name": ["hanoi", "kyoto", "lima"]},
        edges=[(0, 1), (1, 2)],
    )

    with zudb.connect(path) as conn:
        with conn.appender("city") as appender:
            appender.append_row([4, "oslo"])
            assert appender.close() == 1

        cities = conn.execute("MATCH (c:city) RETURN c.name AS name")
        assert len(cities) == 4

        roads = conn.execute("MATCH (:city)-[r:road]->(:city) RETURN r")
        assert len(roads) == 2


def a_failure(path: Path) -> None:
    """An error arrives as an error, with the standard's code on it."""
    import zudb

    with zudb.connect(path) as conn:
        try:
            conn.execute("MATCH (")
        except zudb.SyntaxError as failure:
            assert failure.code == "42001", failure.code
            assert failure.condition
            assert failure.doc_url
            assert failure.retryable is False
        else:
            raise AssertionError("a statement that cannot parse parsed")


def pep_249(path: Path) -> None:
    """The other front door, which is a submodule and not imported above."""
    import zudb.dbapi

    with zudb.dbapi.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute("MATCH (p:person) RETURN p.name AS name ORDER BY p.name")
        assert cursor.description is not None
        assert cursor.description[0][0] == "name"
        assert cursor.fetchall() == [("ada",), ("grace",)]
        cursor.close()


def an_event_loop(path: Path) -> None:
    """And the third, which runs the same engine off the loop's thread."""
    import zudb.aio

    async def run() -> int:
        async with zudb.aio.connect(path, read_only=True) as conn:
            rows = await conn.execute("MATCH (p:person) RETURN p.uid AS uid")
            return len(rows)

    assert asyncio.run(run()) == 2


def nothing_else_installed() -> None:
    """No dependencies, and the refusal says which one to install.

    `pip install zudb` brings nothing with it, which is a claim the
    README makes and which only a machine like this one can check: a
    developer's machine has pandas on it for some other reason.
    """
    import zudb

    for module in ("pandas", "polars", "pyarrow"):
        assert module not in sys.modules, f"{module} was imported by importing zudb"

    with tempfile.TemporaryDirectory() as where:
        with zudb.connect(Path(where) / "empty.zu1") as conn:
            rows = conn.execute("RETURN 1 AS n")
            try:
                rows.to_pandas()
            except ImportError as refusal:
                assert "zudb[pandas]" in str(refusal), refusal
            else:
                raise AssertionError("to_pandas worked without pandas")


def main() -> int:
    imported_from_an_install()

    with tempfile.TemporaryDirectory() as where:
        # Somewhere that is not the checkout and not the current
        # directory, because a client that only works when the database
        # is beside the program is a client with a bug in it.
        root = Path(where)
        a_graph(root / "social.zu1")
        the_bulk_path(root / "roads.zu1")
        a_failure(root / "social.zu1")
        pep_249(root / "social.zu1")
        an_event_loop(root / "social.zu1")

    nothing_else_installed()

    import zudb

    here = sys.version.split()[0]
    print(f"zudb {zudb.__version__}, abi {zudb.__abi_version__}, on Python {here}: it works")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
