"""What a result looks like on the way out through Arrow.

A client that reads rows one at a time and a client that exports a
million of them to a dataframe are the same client, and only one of
those paths is covered by a case that asserts values. The other one has
its own contract: a column of dates is a ``Date32`` and not a string of
digits, a year-month duration is a month-day-nano interval because that
is the interval every reader implements, a node is a struct of the name
of its table and the row it is, and a time with an offset is refused
rather than quietly moved to UTC. None of that shows up in a row a case
compares.

So a case may say what the export gives as well as what the rows are,
and the runner checks both against one statement. What it checks is the
schema, field by field and into the nested types, and how many rows came
back through the stream. The schema is spelled in the C Data Interface's
own format strings, ``l`` for an int64 and ``+s`` for a struct, because
that is the one spelling every language sees the same.

The schema is read off the interface itself rather than out of
``pyarrow``. Two reasons, and the second is the one that decided it. The
first is that ``pyarrow`` is not installed to run the corpus and a check
that needed it would be a check that quietly does not run. The second is
that the C Data Interface is what the C runner has at this point too, so
the two read the same bytes and report them in the same words, which is
the whole reason a format string is what the case writes down.

Values are not read back here. A consumer that decoded every array by
hand in each of nine languages would be nine new decoders under test,
which is more of our own code and not more of the contract; the rows the
case already asserts are the same values by another road.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from dataclasses import field as _field
from typing import Any

from .reader import CorpusError, Node, quote

__all__ = ["ArrowError", "Field", "Refused", "REFUSED", "RESULT", "parse", "exported", "schema"]


@dataclass(frozen=True)
class Field:
    """One field of the schema an export gives, and the fields under it
    when it is a struct or a list.

    A list has exactly one field under it, which Arrow names ``item``,
    and a case writes that out rather than leaving it implied: a client
    that named it ``element`` would export something no reader lines up
    with what another client wrote."""

    name: str
    #: The C Data Interface format string, ``l`` for an int64, ``u`` for
    #: a string, ``tsn:`` for a timestamp in nanoseconds with no zone.
    format: str
    children: tuple[Field, ...] = _field(default_factory=tuple)


class Refused:
    """Arrow has no type for one of the columns, so there is no export to
    describe.

    A time with an offset is the one a statement can write today: Arrow
    has a time and a timestamp and nothing in between, and dropping the
    offset would move the value."""

    def __repr__(self) -> str:
        return "refused"


#: The one value of the class above, so that what a case says about the
#: export is either a list of fields or this.
REFUSED = Refused()

#: How a report names the whole result, which is the place the columns of
#: an export are in.
RESULT = "the result"


def parse(node: Node) -> list[Field] | Refused:
    """The ``arrow:`` of a case, or what is wrong with it."""
    text = node.str_()
    if text is not None:
        if text == "refused":
            return REFUSED
        raise CorpusError(
            f"line {node.line}: `arrow:` is the columns the export gives, or `refused` for a "
            f"result Arrow has no type for, and this is {quote(text)}"
        )
    return _fields(node)


def _fields(node: Node) -> list[Field]:
    items = node.seq()
    if items is None:
        raise CorpusError(
            f"line {node.line}: `arrow:` is a sequence of fields, and this is {node.what()}"
        )
    return [_one(item) for item in items]


def _one(node: Node) -> Field:
    line = node.line
    if node.map() is None:
        raise CorpusError(
            f"line {line}: an Arrow field is a mapping of `name` and `format`, and this is "
            f"{node.what()}"
        )
    unknown = node.unknown(("name", "format", "children"))
    if unknown:
        raise CorpusError(f"line {line}: an Arrow field has no key {quote(unknown[0])}")

    def text(key: str) -> str:
        value = node.get(key)
        spelled = value.str_() if value is not None else None
        if spelled is None:
            raise CorpusError(f"line {line}: an Arrow field has a `{key}:`")
        return spelled

    name = text("name")
    fmt = text("format")
    if not fmt:
        raise CorpusError(f"line {line}: an empty format string is not a type Arrow has")
    children_node = node.get("children")
    children = tuple(_fields(children_node)) if children_node is not None else ()
    # A nested format is the one thing about a format string this reader
    # knows, and it is worth knowing here: a case that wrote the fields
    # of a struct under a `u` would be asserting something the export
    # cannot produce, and finding that out at load time says so with a
    # line number rather than as a failure in a report.
    nested = fmt.startswith("+")
    if nested and not children:
        raise CorpusError(
            f"line {line}: {quote(fmt)} is a nested type and the fields under it are part of it"
        )
    if not nested and children:
        raise CorpusError(f"line {line}: {quote(fmt)} holds no fields, so nothing goes under it")
    return Field(name, fmt, children)


class _Schema(ctypes.Structure):
    """``ArrowSchema``, laid out as the C Data Interface writes it."""


_Schema._fields_ = [
    ("format", ctypes.c_char_p),
    ("name", ctypes.c_char_p),
    ("metadata", ctypes.c_char_p),
    ("flags", ctypes.c_int64),
    ("n_children", ctypes.c_int64),
    ("children", ctypes.POINTER(ctypes.POINTER(_Schema))),
    ("dictionary", ctypes.POINTER(_Schema)),
    ("release", ctypes.CFUNCTYPE(None, ctypes.POINTER(_Schema))),
    ("private_data", ctypes.c_void_p),
]


class _Array(ctypes.Structure):
    """``ArrowArray``. Only the length is read: the values a case cares
    about it already asserts as rows."""


_Array._fields_ = [
    ("length", ctypes.c_int64),
    ("null_count", ctypes.c_int64),
    ("offset", ctypes.c_int64),
    ("n_buffers", ctypes.c_int64),
    ("n_children", ctypes.c_int64),
    ("buffers", ctypes.POINTER(ctypes.c_void_p)),
    ("children", ctypes.POINTER(ctypes.POINTER(_Array))),
    ("dictionary", ctypes.POINTER(_Array)),
    ("release", ctypes.CFUNCTYPE(None, ctypes.POINTER(_Array))),
    ("private_data", ctypes.c_void_p),
]


class _Stream(ctypes.Structure):
    """``ArrowArrayStream``, which is what a result hands over."""


_Get = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(_Stream), ctypes.POINTER(_Schema))
_Next = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(_Stream), ctypes.POINTER(_Array))

_Stream._fields_ = [
    ("get_schema", _Get),
    ("get_next", _Next),
    ("get_last_error", ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.POINTER(_Stream))),
    ("release", ctypes.CFUNCTYPE(None, ctypes.POINTER(_Stream))),
    ("private_data", ctypes.c_void_p),
]


class ArrowError(Exception):
    """The stream said no, with what it said."""


def _pointer(capsule: object, name: bytes, kind: Any) -> Any:
    """The pointer inside a capsule, which is how the C Data Interface
    travels between two Python objects that have never heard of each
    other."""
    get = ctypes.pythonapi.PyCapsule_GetPointer
    get.restype = ctypes.c_void_p
    get.argtypes = [ctypes.py_object, ctypes.c_char_p]
    address = get(capsule, name)
    if not address:
        raise ArrowError("the result gave a capsule with nothing in it")
    return ctypes.cast(address, ctypes.POINTER(kind))


def _walked(one: _Schema) -> Field:
    """One field of an exported schema, and everything under it."""
    children = [_walked(one.children[i].contents) for i in range(one.n_children)]
    return Field(
        name=(one.name or b"").decode("utf-8"),
        format=(one.format or b"").decode("utf-8"),
        children=tuple(children),
    )


def exported(result: Any) -> tuple[list[Field], int]:
    """The columns a result gives through Arrow and how many rows came
    out of the stream.

    The stream is taken once and both answers come out of that one
    taking, because a stream is consumed by reading it and a second
    export would be a second statement in all but name.

    A column Arrow cannot hold is found when the stream is asked for,
    before a row moves, and this client says so with the Python class the
    mistake belongs to: a `TypeError` for a value of a type Arrow does
    not have and a `ValueError` for one that will not fit. Both are the
    export refusing, which is what a case writing `refused` means, so
    both come back out of here under the one name and only around the
    call that can raise them."""
    try:
        capsule = result.__arrow_c_stream__()
    except (TypeError, ValueError) as e:
        raise ArrowError(str(e)) from e
    stream = _pointer(capsule, b"arrow_array_stream", _Stream)

    def said(code: int) -> str:
        text = stream.contents.get_last_error(stream)
        return (text or b"").decode("utf-8", "replace") or f"errno {code}"

    schema_out = _Schema()
    code = stream.contents.get_schema(stream, ctypes.byref(schema_out))
    if code != 0:
        raise ArrowError(said(code))
    try:
        # The stream's schema is a struct of the columns, so what the
        # case is compared against is the fields under it.
        top = _walked(schema_out)
    finally:
        if schema_out.release:
            schema_out.release(ctypes.byref(schema_out))

    count = 0
    while True:
        batch = _Array()
        code = stream.contents.get_next(stream, ctypes.byref(batch))
        if code != 0:
            raise ArrowError(said(code))
        # A released batch is how the interface says there are no more.
        if not batch.release:
            break
        count += batch.length
        batch.release(ctypes.byref(batch))
    return list(top.children), count


def schema(got: list[Field], want: list[Field]) -> str | None:
    """What the export gave that the case did not want, or ``None`` when
    the two agree.

    The comparison walks the schema and the case's fields together and
    stops at the first difference, for the reason the row comparison
    does: the first is nearly always the cause of the rest."""
    return _fields_of("", got, want)


def _fields_of(prefix: str, got: list[Field], want: list[Field]) -> str | None:
    """The fields under one place, where the place is the dotted path of
    the field they are under and the empty one is the result itself."""
    place = RESULT if not prefix else quote(prefix)
    if len(got) != len(want):
        return f"arrow gives {len(got)} fields in {place} where the case wants {len(want)}"
    for i, (one, expected) in enumerate(zip(got, want, strict=True)):
        if one.name != expected.name:
            return (
                f"arrow field {i + 1} in {place} is named {quote(one.name)} where the case wants "
                f"{quote(expected.name)}"
            )
        # The path is the case's own names joined with dots, which is how
        # a field inside a path inside a column is pointed at without
        # printing the whole schema at somebody.
        path = expected.name if not prefix else f"{prefix}.{expected.name}"
        if one.format != expected.format:
            return (
                f"arrow field {quote(path)} is {quote(one.format)} where the case wants "
                f"{quote(expected.format)}"
            )
        why = _fields_of(path, list(one.children), list(expected.children))
        if why is not None:
            return why
    return None
