"""What `import zudb` costs in a process that has not imported it.

An import is paid at the top of every script, at every kernel restart,
and by anyone who typed the name to see whether it is installed, all
before anything has been asked of the library. The engine is inside the
wheel, which makes this a real question rather than a rhetorical one:
the extension is around eight megabytes and the loader maps all of it.

So the budget is a number and the number is checked here. What is
measured is the interpreter's own accounting, `-X importtime`, which
leaves out the cost of starting the interpreter, because that cost is
not ours.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

#: Milliseconds, from the milestone. The reason for a budget at all is
#: that nothing else enforces one: an import that grows five
#: milliseconds a release is an import nobody notices getting slow.
BUDGET = 50.0

#: Milliseconds for the Python around the extension, which is where
#: growth would come from. The extension is the engine and is what it
#: is; the package is a dozen names, `datetime`, `typing`, and the union
#: that describes a value, and it costs about three milliseconds here,
#: most of it `typing`. A ceiling rather than a target: what it is meant
#: to catch is a module imported at package scope by somebody who did
#: not need it there, since each one of those costs milliseconds and
#: none of them costs enough to notice on its own.
OURS = 20.0

#: Runs, because a machine that measures itself is a busy machine. The
#: fastest says what the import costs and the middle one says the
#: fastest was not luck.
RUNS = 5

#: The libraries a result can be handed to, none of which is a
#: dependency and none of which may be imported until somebody asks for
#: one. Importing pandas costs more than everything measured here put
#: together.
HEAVY = ("numpy", "pandas", "polars", "pyarrow")

#: Somewhere that is not the checkout, so that the `python/zudb` beside
#: these tests cannot be what gets imported. What is measured is the
#: installed package, the one a person who ran `pip install zudb` has.
ELSEWHERE = Path(__file__).resolve().parent


def run(code: str, *flags: str) -> subprocess.CompletedProcess[str]:
    """A fresh interpreter, which is the only place a cold import
    happens. Once a process has imported a module, importing it again is
    a dictionary lookup and says nothing about anything.
    """
    return subprocess.run(
        [sys.executable, *flags, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=ELSEWHERE,
    )


def profile(module: str = "zudb") -> dict[str, float]:
    """Milliseconds under every module one import pulled in.

    `-X importtime` writes a line per module: time spent in it, then
    cumulative time under it. The cumulative column is the whole cost of
    a module including everything it imported, and the leading spaces on
    a name are what tell a child from a parent, so the names come back
    stripped and are matched exactly. The first line of the three is a
    header, which is why a line counts only once its middle column is a
    number.
    """
    done = run(f"import {module}", "-X", "importtime")
    measured = {}
    for line in done.stderr.splitlines():
        columns = line.split("|")
        if len(columns) == 3 and columns[1].strip().isdigit():
            measured[columns[2].strip()] = int(columns[1]) / 1000
    if module not in measured:
        raise AssertionError(f"no importtime line for {module}:\n{done.stderr}")
    return measured


def test_a_cold_import_is_inside_the_budget() -> None:
    costs = sorted(profile()["zudb"] for _ in range(RUNS))
    reading = ", ".join(f"{one:.1f}" for one in costs)
    assert costs[0] < BUDGET, f"fastest of {RUNS} was {costs[0]:.1f} ms of {BUDGET}: {reading}"
    assert costs[len(costs) // 2] < BUDGET, f"the middle run was not inside it either: {reading}"


def test_the_python_around_the_engine_has_a_budget_of_its_own() -> None:
    # Both numbers out of one import, because two imports are two
    # measurements of a machine that was doing something else in
    # between. The extension is the part nobody can make cheaper and
    # the package is the part that grows a module at a time.
    measured = profile()
    ours = measured["zudb"] - measured["zudb._zudb"]
    assert ours < OURS, f"the package cost {ours:.1f} ms of {OURS}, extension excluded"


def test_no_dataframe_library_is_imported_by_importing_this_one() -> None:
    heavy = "(" + ", ".join(repr(name) for name in HEAVY) + ")"
    done = run(f"import sys, zudb; print(*sorted(set(sys.modules) & set({heavy})))")
    assert done.stdout.strip() == "", f"imported without being asked: {done.stdout.strip()}"


def test_the_event_loop_module_is_not_imported_by_importing_this_one() -> None:
    """`zudb.aio` is a submodule a caller asks for by name.

    It pulls in asyncio and a thread pool, and a script that never
    awaits anything should not pay for either, so it is left out of the
    package's own imports.
    """
    done = run("import sys, zudb; print('asyncio' in sys.modules, 'zudb.aio' in sys.modules)")
    assert done.stdout.strip() == "False False"


def test_the_dbapi_module_is_not_imported_by_importing_this_one() -> None:
    """`zudb.dbapi` is a submodule a caller asks for by name too.

    It is there for code written against PEP 249, which is not most
    code, and the package that everyone imports should not carry it.
    """
    done = run("import sys, zudb; print('zudb.dbapi' in sys.modules)")
    assert done.stdout.strip() == "False"


def test_numpy_arrives_when_a_result_is_asked_for_its_arrays(tmp_path: Path) -> None:
    """The extension links numpy's C API and still does not import it.

    It is loaded at the first array and not at module init, which is the
    difference between a client that costs numpy to import and one that
    costs it to use.
    """
    pytest.importorskip("numpy")
    path = tmp_path / "arrays.zu1"
    done = run(
        "import sys, zudb\n"
        f"zudb.load({str(path)!r}, nodes='person', columns={{'uid': [1, 2, 3]}})\n"
        f"conn = zudb.connect({str(path)!r}, read_only=True)\n"
        "result = conn.execute('MATCH (p:person) RETURN p.uid AS uid')\n"
        "print('before', 'numpy' in sys.modules)\n"
        "arrays = result.fetchnumpy()\n"
        "print('after', 'numpy' in sys.modules, len(arrays['uid']))\n"
    )
    assert done.stdout.splitlines() == ["before False", "after True 3"]


def test_pyarrow_arrives_when_a_result_is_asked_for_its_columns(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    path = tmp_path / "columns.zu1"
    done = run(
        "import sys, zudb\n"
        f"zudb.load({str(path)!r}, nodes='person', columns={{'uid': [1, 2, 3]}})\n"
        f"conn = zudb.connect({str(path)!r}, read_only=True)\n"
        "result = conn.execute('MATCH (p:person) RETURN p.uid AS uid')\n"
        "print('before', 'pyarrow' in sys.modules)\n"
        "table = result.to_arrow()\n"
        "print('after', 'pyarrow' in sys.modules, table.num_rows)\n"
    )
    assert done.stdout.splitlines() == ["before False", "after True 3"]
