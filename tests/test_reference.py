"""The generated reference, and the one thing about it worth asserting.

Most of what a documentation generator does is not this suite's
business: pdoc's templates are pdoc's, and a test that counted headings
would fail on the week pdoc renders one differently. What is this
client's business is that the reference says what the package says, and
half of the package is compiled, so the interesting failure is the
quiet one: pdoc reads a pyo3 function object, finds no annotations on
it, and publishes a whole reference in which nothing has a type.

That is a reference that builds, looks finished, and is wrong about
every signature on the page. So the check is in the tool, and this is
the check on the check: it fires when the stub overlay is missing and
it passes when it is there.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import reference

pdoc = pytest.importorskip("pdoc", reason="the reference generator, a dev dependency")


def test_without_the_overlay_the_signatures_have_no_types_on_them() -> None:
    """The failure the tool exists to catch, uncaught."""
    assert reference.bare("zudb"), "pdoc read types off a compiled module by itself"


def test_with_the_overlay_every_public_signature_is_typed() -> None:
    """And caught. The overlay is what the tool puts on sys.path."""
    with reference.stubs():
        assert reference.bare("zudb") == []


def test_the_overlay_is_the_stub_the_type_checkers_read(tmp_path: Path) -> None:
    """Derived from `_zudb.pyi` and not a second copy of it.

    Every line but the one relative import, which a stub loaded outside
    the import machinery has no package to resolve.
    """
    stub = (package() / "_zudb.pyi").read_text(encoding="utf-8").splitlines()
    written = (
        (reference.overlay(package(), tmp_path) / "__init__.pyi")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(written) == len(stub)
    moved = [(one, two) for one, two in zip(stub, written, strict=True) if one != two]
    assert moved == [("from .types import Value", "from zudb.types import Value")]


def test_the_reference_covers_what_a_reader_can_import(tmp_path: Path) -> None:
    """One page per module, and the compiled module is not one.

    `zudb._zudb` is where the classes are defined and it is not a name
    anybody imports, so it is not in the list. Its names are documented
    on the `zudb` page, which is where they are exported from.
    """
    import zudb

    reference.build(tmp_path)
    for name in reference.MODULES:
        assert (tmp_path / (name.replace(".", "/") + ".html")).is_file(), name
    assert "zudb._zudb" not in reference.MODULES
    assert set(reference.MODULES) - {"zudb"} == {
        f"zudb.{name}" for name in ("aio", "dbapi", "errors", "magic", "types")
    }
    # The version the wheel says it is, on the page the wheel produced.
    assert zudb.__version__ in (tmp_path / "zudb.html").read_text(encoding="utf-8")


def test_the_reference_replaces_what_was_there(tmp_path: Path) -> None:
    """A page for a module that went away is worse than no page.

    It is a name a reader can still find, still in the search index
    built beside it, and gone from the package.
    """
    stale = tmp_path / "zudb" / "gone.html"
    stale.parent.mkdir(parents=True)
    stale.write_text("a module this package used to have", encoding="utf-8")
    reference.build(tmp_path)
    assert not stale.exists()


def package() -> Path:
    """Where the installed package is, which is what is documented."""
    import zudb

    return Path(zudb.__file__ or "").parent
