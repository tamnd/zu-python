"""The clean-machine program, run here as well.

`tools/smoke.py` is what the nightly install job runs inside a container
that holds an interpreter, a package manager and nothing else, and that
container is the least convenient place in the world to find out that a
line of it went stale. So each piece runs here too, against the install
this suite already has, where a failure arrives with a traceback and a
name attached rather than as a red square once a day.

The one piece that cannot run here is the one that is about the machine
rather than about the client: this machine has pandas on it, because the
rest of the suite needs pandas.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import smoke
import zudb

# `maturin develop`, which is how this is worked on, puts the checkout
# on the path instead of installing anything, so the one check that is
# about where the package came from has nothing to say here. Every job
# in CI installs the wheel, which is where it does.
WHERE = Path(zudb.__file__).resolve()
INSTALLED = "site-packages" in WHERE.parts or "dist-packages" in WHERE.parts


@pytest.mark.skipif(not INSTALLED, reason=f"zudb here is a checkout at {WHERE}, not an install")
def test_the_package_under_test_is_an_install() -> None:
    smoke.imported_from_an_install()


def test_the_quickstart_runs(tmp_path) -> None:
    smoke.a_graph(tmp_path / "social.zu1")


def test_the_bulk_paths_run(tmp_path) -> None:
    smoke.the_bulk_path(tmp_path / "roads.zu1")


def test_a_failure_is_a_failure(tmp_path) -> None:
    path = tmp_path / "social.zu1"
    smoke.a_graph(path)
    smoke.a_failure(path)


def test_the_dbapi_front_door_runs(tmp_path) -> None:
    path = tmp_path / "social.zu1"
    smoke.a_graph(path)
    smoke.pep_249(path)


def test_the_event_loop_front_door_runs(tmp_path) -> None:
    path = tmp_path / "social.zu1"
    smoke.a_graph(path)
    smoke.an_event_loop(path)


@pytest.mark.skipif(
    importlib.util.find_spec("pandas") is not None,
    reason="this check is about a machine with no pandas on it, and this one has pandas",
)
def test_nothing_arrived_with_the_wheel() -> None:
    smoke.nothing_else_installed()
