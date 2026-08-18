"""The stub for the compiled module, against the compiled module.

A stub is a promise no interpreter checks: nothing fails when it names a
method the engine does not have, or gives a parameter a name the engine
does not answer to. What fails is the caller who believed the editor.

So it is checked here, with griffe reading the stub as text and the
extension by inspection, and the two compared. PyO3 writes a text
signature for every function it exports, which is what makes the second
half of that possible: the parameter names, their kinds and their
defaults all come back out of the built module.

What is compared is the shape and not the types. The stub says a column
is a ``list[str]`` and no inspection of a compiled module can confirm
it, so that part is on the tests that call it. Everything the module
actually declares, this checks.
"""

from __future__ import annotations

import ast
import shutil
from pathlib import Path
from typing import Any

import pytest
import zudb

griffe = pytest.importorskip("griffe")

#: The installed package, not the checkout, because the stub that
#: matters is the one inside the wheel: a stub that is right in the
#: source tree and missing from the wheel helps nobody.
PACKAGE = Path(zudb.__file__).resolve().parent


@pytest.fixture(scope="module")
def stub() -> Any:
    """The stub, read as text, with nothing compiled in the way.

    Copied to a directory of its own first because griffe merges a stub
    into the extension beside it, and a merged stub would agree with the
    module by construction: a name missing from the stub would come back
    from the `.so` and the check would pass having checked nothing.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as into:
        package = Path(into) / "zudb"
        package.mkdir()
        for source in PACKAGE.iterdir():
            if source.suffix in {".py", ".pyi"} or source.name == "py.typed":
                shutil.copy(source, package / source.name)
        yield griffe.load("zudb", search_paths=[package.parent])["_zudb"]


@pytest.fixture(scope="module")
def runtime() -> Any:
    """The extension module as it was built, by inspection."""
    return griffe.load("zudb._zudb", force_inspection=True)


def resolve(member: Any) -> Any:
    """A member with its alias followed.

    Every class in the extension says `module = "zudb"`, which is where
    a caller imports it from and where its repr should point, so griffe
    sees the module it was inspected in as holding an alias to it.
    """
    return member.target if member.is_alias else member


def public(obj: Any, *, drop_imports: bool = False) -> dict[str, Any]:
    """The members worth comparing: what was written, not what was
    generated or imported.

    A stub's imports are aliases and none of its own names are, so the
    stub side drops every alias it has. The module side keeps them and
    follows them, because a class that says `module = "zudb"` is an
    alias in the module it was inspected in.
    """
    return {
        name: resolve(member)
        for name, member in obj.members.items()
        if not name.startswith("_") and not (drop_imports and member.is_alias)
    }


def declared(obj: Any) -> set[str]:
    """The dunders the stub declares, which are the ones a caller is
    told to call.

    `__init__` is not one of them: PyO3 exports a constructor as
    `__new__`, so the stub writes the `__init__` a checker expects and
    the two are compared on their own below.
    """
    return {
        name
        for name in obj.members
        if name.startswith("__") and name.endswith("__") and name != "__init__"
    }


def kinds(member: Any) -> str:
    """A member as either a value or a call, which is the distinction a
    caller sees. A property and a plain attribute are both read, and a
    stub writes a property where PyO3 exports a getter."""
    return "function" if member.kind.value == "function" else "attribute"


def default(text: str | None) -> str | None:
    """A default value, spelled one way.

    The stub is read as source and the module as text signatures, so
    the same string arrives as `"rel"` from one and `'rel'` from the
    other. What is compared is the value, when it is one a literal can
    hold, and the text when it is not.
    """
    if text is None:
        return None
    try:
        return repr(ast.literal_eval(text))
    except (ValueError, SyntaxError):
        return text


def signature(member: Any) -> list[tuple[str, str, str | None]]:
    """The parameters of a function, as they can be compared.

    `self` is dropped: a stub writes it as an ordinary parameter and an
    inspected method descriptor reports it as positional-only, and the
    difference is a fact about descriptors rather than about the method.
    """
    return [
        (
            parameter.name,
            parameter.kind.name,
            None if parameter.kind.name.startswith("var_") else default(parameter.default),
        )
        for parameter in member.parameters
        if parameter.name != "self"
    ]


def test_the_stub_names_everything_the_module_exports(stub: Any, runtime: Any) -> None:
    assert set(public(runtime)) == set(public(stub, drop_imports=True))


def test_every_dunder_the_stub_promises_is_there(stub: Any, runtime: Any) -> None:
    # One way only. `__eq__`, `__lt__` and the rest of what
    # `#[pyclass(eq)]` generates are real and are not worth writing out.
    assert declared(stub) <= set(runtime.members)


@pytest.mark.parametrize(
    "name", ["Connection", "Result", "Node", "Rel", "Path", "Duration", "connect", "load"]
)
def test_a_name_is_the_same_kind_in_both(stub: Any, runtime: Any, name: str) -> None:
    assert resolve(stub[name]).kind.value == resolve(runtime[name]).kind.value


@pytest.mark.parametrize("name", ["Connection", "Result", "Node", "Rel", "Path", "Duration"])
def test_a_class_has_the_members_the_stub_gives_it(stub: Any, runtime: Any, name: str) -> None:
    theirs, ours = public(resolve(runtime[name])), public(resolve(stub[name]))
    assert set(theirs) == set(ours)
    assert {n: kinds(m) for n, m in theirs.items()} == {n: kinds(m) for n, m in ours.items()}
    assert declared(resolve(stub[name])) <= set(resolve(runtime[name]).members)


def test_a_function_takes_what_the_stub_says_it_takes(stub: Any, runtime: Any) -> None:
    for name in ("connect", "load"):
        assert signature(stub[name]) == signature(resolve(runtime[name])), name


@pytest.mark.parametrize("name", ["Connection", "Result", "Node", "Rel", "Path", "Duration"])
def test_a_method_takes_what_the_stub_says_it_takes(stub: Any, runtime: Any, name: str) -> None:
    theirs, ours = resolve(runtime[name]), resolve(stub[name])
    for method, member in public(ours).items():
        if member.kind.value != "function":
            continue
        assert signature(member) == signature(theirs[method]), f"{name}.{method}"


def test_a_constructor_takes_what_the_stub_says_it_takes(stub: Any) -> None:
    # Not through griffe, which reads a constructor off `__init__` and
    # finds PyO3's `__new__` reported as `(*args, **kwargs)`. The real
    # signature is on the class, where `inspect` looks, and where a
    # stub's `__init__` has to agree with it.
    import inspect

    import zudb

    for name in ("Node", "Rel", "Path", "Duration"):
        theirs = [
            (p.name, p.kind.name.lower(), None if p.default is p.empty else repr(p.default))
            for p in inspect.signature(getattr(zudb, name)).parameters.values()
        ]
        assert theirs == signature(stub[name]["__init__"]), name


def test_the_stub_ships_in_the_package(stub: Any) -> None:
    # Beside `py.typed`, which is what tells a checker to read it at
    # all, and both are picked up by maturin because they sit in the
    # Python source tree.
    assert (PACKAGE / "_zudb.pyi").is_file()
    assert (PACKAGE / "py.typed").is_file()
