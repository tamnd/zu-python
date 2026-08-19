"""The API reference, generated from the package that ships.

A reference written by hand beside the code is a reference that is
wrong by the second release, and wrong in the way that costs the most:
it looks maintained. So this one is built from the installed package,
by inspection, and the release builds it from the same wheel it is
about to publish.

    python tools/reference.py docs/api

pdoc is the generator because it reads a package the way the
interpreter does and asks for nothing in return: no configuration file,
no theme to keep, no second copy of the module list. What it produces
is HTML that opens from disk, which matters for a client whose docs go
out as an artifact of the release rather than as a deployment.

Half of this package is compiled, and that is the one thing the
inspection cannot see through. `zudb.connect` is a function object
pyo3 built, so its signature is `(path, *, read_only=False, ...)` and
the types are in `_zudb.pyi`, where the type checkers read them. pdoc
knows about stub files and looks for one named after the module it is
documenting, which is `zudb`, and the stub is named after the module
the names were defined in, which is `zudb._zudb`. So the stub is put
where pdoc looks, under the `-stubs` name PEP 561 reserves for exactly
this, and the one relative import in it is rewritten because a stub
loaded outside the import machinery has no package to be relative to.

Derived rather than written, which is the point: there is no second
declaration of the surface to keep in step, and `bare()` below fails
the build if the overlay stops landing. Without that check a reference
that quietly lost every type annotation would still be a reference that
built, and nobody reads their own generated docs closely enough to
notice.
"""

from __future__ import annotations

import contextlib
import shutil
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

#: What the reference covers, which is what a reader can import. The
#: compiled module is not in the list: it is an implementation detail
#: with an underscore on it, and every name it defines is re-exported
#: from `zudb` and documented there.
MODULES = [
    "zudb",
    "zudb.aio",
    "zudb.dbapi",
    "zudb.errors",
    "zudb.magic",
    "zudb.types",
]

#: The name pdoc looks under, and PEP 561's name for a stub package
#: distributed apart from the package it describes. `zudb/__init__.pyi`
#: beside the real `__init__.py` would work as well and would be a
#: second file in the wheel claiming to be the public surface, which is
#: the thing this whole file exists to avoid.
STUBS = "zudb-stubs"


def overlay(package: Path, into: Path) -> Path:
    """Write the stub pdoc will find for `zudb`, and answer where.

    The content is `_zudb.pyi`, unchanged except for its one relative
    import. pdoc loads a stub with a loader of its own rather than
    through `import`, so the module has no package and `from .types
    import Value` raises before anything is read; naming the package
    absolutely is the whole edit.
    """
    stub = (package / "_zudb.pyi").read_text(encoding="utf-8")
    written = into / STUBS
    written.mkdir(parents=True, exist_ok=True)
    (written / "__init__.pyi").write_text(
        stub.replace("from .types import", "from zudb.types import"), encoding="utf-8"
    )
    return written


@contextlib.contextmanager
def stubs() -> Iterator[Path]:
    """The overlay, on `sys.path`, for as long as the block runs.

    pdoc remembers where a module's stub was and where it was not, and
    the answer changes here twice in one process. So the caches that
    hold it are dropped on the way in and on the way out, which is what
    lets a test ask the question both ways and get both answers.
    """
    import pdoc.doc
    import pdoc.doc_pyi
    import zudb

    def forget() -> None:
        pdoc.doc_pyi.find_stub_file.cache_clear()
        pdoc.doc.Module.from_name.cache_clear()

    with tempfile.TemporaryDirectory() as scratch:
        written = overlay(Path(zudb.__file__ or "").parent, Path(scratch))
        sys.path.insert(0, str(written.parent))
        forget()
        try:
            yield written
        finally:
            sys.path.remove(str(written.parent))
            forget()


def bare(module: str) -> list[str]:
    """The names on `module` whose signature carries no annotation.

    Which is the check that the overlay landed. Everything the compiled
    module defines is annotated in the stub, so after the overlay every
    parameter of every public function has a type on it, and a name
    here is a name pdoc documented from the object instead. An empty
    list is a reference worth publishing.

    What a reader reads is what is counted: the members a namespace
    declares itself, under names with no underscore on the front.
    Inherited ones are somebody else's, and `object.__eq__` takes an
    untyped `value` on every class in the language.
    """
    import pdoc.doc

    def functions(namespace: pdoc.doc.Namespace) -> list[pdoc.doc.Function]:
        found = []
        for doc in namespace.own_members:
            if doc.name.startswith("_"):
                continue
            if isinstance(doc, pdoc.doc.Function):
                found.append(doc)
            elif isinstance(doc, pdoc.doc.Class):
                found.extend(functions(doc))
        return found

    untyped = []
    for one in functions(pdoc.doc.Module.from_name(module)):
        wanted = [
            name
            for name, parameter in one.signature.parameters.items()
            if name not in ("self", "cls") and parameter.annotation is parameter.empty
        ]
        if wanted:
            untyped.append(f"{one.fullname}({', '.join(wanted)})")
    return untyped


def build(into: Path) -> None:
    """The reference, in `into`, replacing whatever was there.

    Replacing rather than merging, because a page for a module that was
    removed is worse than no page at all: it is a name a reader can
    still find, still linked from the index of a search built beside
    it, and gone from the package.
    """
    import pdoc

    with stubs():
        untyped = bare("zudb")
        if untyped:
            raise SystemExit(
                "the stub overlay did not land, so the reference would publish "
                f"{len(untyped)} name(s) with no types on them: {', '.join(sorted(untyped))}"
            )
        shutil.rmtree(into, ignore_errors=True)
        pdoc.pdoc(*MODULES, output_directory=into)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <directory>", file=sys.stderr)
        return 2
    into = Path(argv[1])
    build(into)
    pages = sorted(path for path in into.rglob("*.html"))
    missing = [name for name in MODULES if not (into / (name.replace(".", "/") + ".html")).exists()]
    for name in missing:
        print(f"{name} was documented and has no page", file=sys.stderr)
    print(f"{len(pages)} pages in {into}, {len(MODULES)} modules")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
