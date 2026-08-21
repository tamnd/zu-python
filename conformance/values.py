"""The ``{type, value}`` encoding a case writes its values in.

Every value in the corpus is a mapping with a ``type`` naming the GQL
type and a ``value`` holding the payload. The type is written down
rather than inferred because the corpus is read by nine languages and
inference is where they differ: a bare ``1`` is an integer in YAML, and
which integer it becomes is a decision each host language makes on its
own.

The payload is a YAML scalar where a YAML scalar is exact, and a string
where it is not. An integer wider than 53 bits is a string, because most
YAML readers hand a number to a double. A float is a string, for that
reason and for ``NaN``, ``inf`` and ``-0.0``. A temporal value is a
string, because YAML has no type that keeps an offset.

``NODE``, ``EDGE`` and ``PATH`` are the values a graph has and a table
does not, and they are written as names rather than as the numbers the
engine holds. A node is ``"person#1"``, the table it is a row of and
which row of it. An edge is ``"knows#0->1"``, its table and the two rows
it runs between. A path is a sequence, like a list, holding a node and
then an edge and a node for each hop.

Refusing the wrong form is half the point, and refusing it here is what
makes this a second reader of the corpus rather than a consumer of it.

What a decoded value becomes is a Python value, because that is what a
statement gives back through this client and what a comparison has to be
against. The declared width is dropped in the process, so a case that
says INT8 and one that says INT64 both become ``int``: that is a fact
about Python rather than about the corpus, and the reference runner in
Rust drops it too, since the engine's own value is one signed 64 bit
integer either way.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass

from zudb import Duration, Node, Path, Rel

from .reader import CorpusError, quote
from .reader import Node as YamlNode

__all__ = [
    "decode",
    "typed",
    "payload",
    "cell",
    "same",
    "show",
    "form",
    "Edge",
    "Walk",
    "TooFine",
    "too_fine",
]

#: Whether a type's payload is written as a quoted string. ``False`` is
#: a type a YAML scalar carries without loss, ``True`` is one it does
#: not.
_TYPES: dict[str, bool] = {
    "NULL": False,
    "BOOL": False,
    "INT8": False,
    "INT16": False,
    "INT32": False,
    "INT64": True,
    "UINT8": False,
    "UINT16": False,
    "UINT32": False,
    "UINT64": True,
    "FLOAT32": True,
    "FLOAT64": True,
    "STRING": False,
    "DATE": True,
    "LOCALTIME": True,
    "ZONEDTIME": True,
    "LOCALDATETIME": True,
    "ZONEDDATETIME": True,
    "DURATION": True,
    "LIST": False,
    # A node and an edge are written in quotes because what a case
    # spells is a name and two numbers with punctuation between them,
    # which is text in every reader and a number in none.
    "NODE": True,
    "EDGE": True,
    # A path is a sequence, like a list, because that is what it is: the
    # nodes and edges of a walk, in the order they were walked.
    "PATH": False,
}

#: The types the encoding reserves a name for and the engine has no
#: runtime value for yet, kept apart from an outright typo so that the
#: error says which of the two it is.
_RESERVED = ("DECIMAL", "BYTES")

#: The range each integer width holds, so that a case writing a value
#: its own type cannot carry is refused rather than stored wider than it
#: says. UINT64 stops at the signed maximum because the engine's integer
#: is signed and 64 bits wide, and wrapping the top half into a negative
#: would be a case that passes while meaning the opposite of what it says.
_RANGES: dict[str, tuple[int, int]] = {
    "INT8": (-(2**7), 2**7 - 1),
    "INT16": (-(2**15), 2**15 - 1),
    "INT32": (-(2**31), 2**31 - 1),
    "INT64": (-(2**63), 2**63 - 1),
    "UINT8": (0, 2**8 - 1),
    "UINT16": (0, 2**16 - 1),
    "UINT32": (0, 2**32 - 1),
    "UINT64": (0, 2**63 - 1),
}


@dataclass(frozen=True)
class Edge:
    """An edge as a case names it: the rel table it is in and the two
    rows it runs between.

    Not ``zudb.Rel``, which carries a fourth field the corpus does not
    write. ``ord`` is where the edge's properties sit, which is its
    place in the order the table was loaded in, and that is a number the
    loader chose rather than one the case did. A pair may run more than
    once, and a case that has to tell two parallel edges apart asserts a
    property of them instead."""

    table: str
    src: int
    dst: int


@dataclass(frozen=True)
class Walk:
    """A path as a case writes it: nodes and edges alternating, a node
    at each end.

    Not ``zudb.Path`` for the reason ``Edge`` is not ``zudb.Rel``, and
    for one more: what a case compares is the walk, and two walks that
    cross the same edges are the same walk whichever copy of a parallel
    edge the engine happened to hand back."""

    elements: tuple[object, ...]


@dataclass(frozen=True)
class TooFine:
    """A temporal value written finer than this client can hold.

    The engine keeps a time to the nanosecond and Python's ``datetime``
    keeps one to the microsecond, so a case asserting nine digits is a
    case this client's value mapping cannot answer. Decoding it to a
    ``datetime`` would truncate it, and the case would then pass by
    comparing one truncated value against another, which is the exact
    defect the case was written to catch.

    So it decodes to this instead. Nothing equals it, it prints as the
    text that was written, and the runner turns a case holding one into
    an unsupported rather than a failure, because a value mapping that
    cannot carry the value is a limit of the client and not a wrong
    answer from the engine."""

    ty: str
    text: str


def truncated(value: object) -> object:
    """The nearest thing Python has to a value it cannot hold exactly.

    Only the load needs this. A column has to go into the file for the
    suite to have a graph at all, and refusing to load it would take out
    every case of the suite rather than the ones that read the column.
    So the column goes in truncated, and the cases that read it back get
    a truncated answer against an expectation nothing equals, which is
    the report those cases should give."""
    if isinstance(value, TooFine):
        head, _, rest = value.text.partition(".")
        digits = ""
        while rest[len(digits) : len(digits) + 1].isdigit():
            digits += rest[len(digits)]
        # The offset is after the fraction and belongs to the value, so
        # what is cut is the digits and not the tail behind them.
        out = _scalar(value.ty, f"{head}.{digits[:6]}{rest[len(digits) :]}")
        if out is _NOT_ONE:
            raise CorpusError(f"{value.ty} {value.text} does not truncate to a value")
        return out
    if isinstance(value, list):
        return [truncated(item) for item in value]
    return value


def too_fine(value: object) -> TooFine | None:
    """The first value inside this one that this client cannot hold, if
    there is one. Recursive, because a list of times is a list."""
    if isinstance(value, TooFine):
        return value
    if isinstance(value, list):
        for item in value:
            found = too_fine(item)
            if found is not None:
                return found
    return None


def form(ty: str) -> bool | None:
    """Whether a type is written quoted, or ``None`` if it is not a
    type."""
    return _TYPES.get(ty)


def _unknown(ty: str) -> str:
    if ty in _RESERVED:
        return f"{ty} is a type the encoding reserves and the engine has no value for"
    return f"{ty} is not a type this encoding knows"


def decode(node: YamlNode) -> object:
    """The value a ``{type, value}`` mapping describes."""
    if node.map() is None:
        raise CorpusError(
            f"line {node.line}: a value is a mapping of `type` and `value`, and this is "
            f"{node.what()}"
        )
    unknown = node.unknown(("type", "value"))
    if unknown:
        raise CorpusError(f"line {node.line}: a value has no key {quote(unknown[0])}")
    return typed(node)


def typed(node: YamlNode) -> object:
    """The ``type`` and ``value`` of a mapping that carries more than
    those two, which is a parameter: it is a value with a name, and the
    name belongs to the case rather than to the encoding."""
    line = node.line
    ty_node = node.get("type")
    if ty_node is None:
        raise CorpusError(f"line {line}: a value with no `type`")
    ty = ty_node.str_()
    if ty is None:
        raise CorpusError(f"line {line}: a `type` that is not a name")

    # Checked here as well as in `payload`, because a value whose type is
    # not a type and which also has no `value` under it should be told
    # about the type first: that is the mistake, and the missing payload
    # is a consequence of it.
    if form(ty) is None:
        raise CorpusError(f"line {line}: {_unknown(ty)}")

    if ty == "NULL":
        if node.get("value") is not None:
            raise CorpusError(f"line {line}: NULL carries no `value`")
        return None
    value = node.get("value")
    if value is None:
        raise CorpusError(f"line {line}: a {ty} with no `value`")
    return payload(ty, value)


def payload(ty: str, value: YamlNode) -> object:
    """The value a payload spells under a type that has already been
    read.

    A row of a case names its type beside every value. A column of a load
    names it once at the top and every value under it is a bare payload,
    which is the same encoding with the type factored out, so it is the
    same function reading it."""
    quoted_form = form(ty)
    if quoted_form is None:
        raise CorpusError(f"line {value.line}: {_unknown(ty)}")

    if ty in ("LIST", "PATH"):
        # The empty list is a value worth a case and needs a spelling,
        # which is a `value:` with nothing under it.
        items = value.seq_or_empty()
        if items is None:
            raise CorpusError(
                f"line {value.line}: a {ty} holds a sequence of values, and this is {value.what()}"
            )
        decoded = [decode(item) for item in items]
        return decoded if ty == "LIST" else _walk(decoded, value.line)

    scalar = value.scalar()
    if scalar is None:
        raise CorpusError(f"line {value.line}: a {ty} holds one scalar, and this is {value.what()}")
    text, quoted = scalar
    line = value.line
    # The one rule the whole encoding exists for, checked before the text
    # is looked at, because a value that parses is exactly the case where
    # a silent misread would survive review.
    if quoted_form and not quoted:
        # A node and an edge are quoted for a different reason from the
        # numbers, so they are told a different reason. Both reasons are
        # the same rule: a payload is quoted where a bare one would read
        # as something else in some reader of this file.
        if ty in ("NODE", "EDGE"):
            raise CorpusError(
                f"line {line}: {ty} is written in quotes, because {text} is a name and two "
                "numbers and no reader has a scalar for that"
            )
        raise CorpusError(
            f"line {line}: {ty} is written in quotes, because a bare {text} is a number and some "
            "reader of this file will round it"
        )
    if not quoted_form and quoted and ty != "STRING":
        raise CorpusError(
            f"line {line}: {ty} is written without quotes, so that a reader cannot take it for a "
            "string"
        )
    if ty == "NODE":
        out = _node_at(text)
    elif ty == "EDGE":
        out = _edge_at(text)
    else:
        out = _scalar(ty, text)
    if out is _NOT_ONE:
        raise CorpusError(f"line {line}: {quote(text)} is not a {ty}")
    return out


def _walk(items: list[object], line: int) -> Walk:
    """The nodes and edges of a walk, or what is wrong with the sequence
    somebody wrote.

    A path alternates and ends at both ends with a node, so a sequence
    that does not is a case that could never pass. Refusing it here
    rather than at the comparison is the difference between a message
    naming the line and a report saying the row differs."""
    if len(items) % 2 == 0:
        raise CorpusError(
            f"line {line}: a PATH is a node, then an edge and a node for each hop, so it holds an "
            f"odd number of values and this holds {len(items)}"
        )
    for i, item in enumerate(items):
        want_node = i % 2 == 0
        if isinstance(item, Node):
            ok, was = want_node, "a NODE"
        elif isinstance(item, Edge):
            ok, was = not want_node, "an EDGE"
        else:
            ok, was = False, "neither a NODE nor an EDGE"
        if not ok:
            wanted = "a NODE" if want_node else "an EDGE"
            raise CorpusError(
                f"line {line}: a PATH alternates, so value {i + 1} is {was} where it should be "
                f"{wanted}"
            )
    return Walk(tuple(items))


def _node_at(text: str) -> object:
    """A node, written as its table and the offset of its row:
    ``person#1``.

    The table's name rather than its id, because the id is a number the
    file decided and every client builds its own file. Split from the
    right, so that a table whose name holds a ``#`` is still readable."""
    table, hash_, offset = text.rpartition("#")
    if not hash_ or not table or not offset.isdigit():
        return _NOT_ONE
    return Node(table, int(offset))


def _edge_at(text: str) -> object:
    """An edge, written as its table and the rows it runs between:
    ``knows#0->1``."""
    table, hash_, ends = text.rpartition("#")
    if not hash_ or not table:
        return _NOT_ONE
    src, arrow, dst = ends.partition("->")
    if not arrow or not src.isdigit() or not dst.isdigit():
        return _NOT_ONE
    return Edge(table, int(src), int(dst))


def cell(value: object) -> object:
    """The corpus's own shape for a value that came back from a
    statement.

    Everything a table holds is spelled the same on both sides and comes
    through untouched. A graph value is not: the engine's edge carries an
    ``ord`` the corpus does not write, so an edge and a path are put into
    the shapes above before anything is compared, which is what the Rust
    runner's ``from_engine`` does for the same reason."""
    if isinstance(value, Rel):
        return Edge(value.table, value.src, value.dst)
    if isinstance(value, Path):
        return Walk(tuple(cell(item) for item in value.elements))
    if isinstance(value, list):
        return [cell(item) for item in value]
    return value


#: What `_scalar` gives back for text that does not spell a value of the
#: type, which cannot be `None` because `None` is a value NULL spells.
_NOT_ONE = object()


def _scalar(ty: str, text: str) -> object:
    if ty == "BOOL":
        return {"true": True, "false": False}.get(text, _NOT_ONE)
    if ty == "STRING":
        return text
    if ty in _RANGES:
        try:
            n = int(text)
        except ValueError:
            return _NOT_ONE
        # Python's int is unbounded and the corpus's is not, so the
        # width has to be checked rather than trusted, which is the one
        # place this reader does work its Rust counterpart gets from the
        # type system.
        if text != str(n):
            return _NOT_ONE
        low, high = _RANGES[ty]
        return n if low <= n <= high else _NOT_ONE
    if ty in ("FLOAT32", "FLOAT64"):
        f = _float(text)
        if f is _NOT_ONE:
            return f
        assert isinstance(f, float)
        return _to_float32(f) if ty == "FLOAT32" else f
    if ty == "DATE":
        return _parse(text, _date)
    if ty in ("LOCALTIME", "ZONEDTIME", "LOCALDATETIME", "ZONEDDATETIME"):
        fn = {
            "LOCALTIME": _local_time,
            "ZONEDTIME": _zoned_time,
            "LOCALDATETIME": _local_datetime,
            "ZONEDDATETIME": _zoned_datetime,
        }[ty]
        out = _parse(text, fn)
        # Parsed first, so that text which is not a time of any precision
        # is refused as such rather than reported as one this client
        # cannot hold.
        if out is not _NOT_ONE and _finer_than_a_microsecond(text):
            return TooFine(ty, text)
        return out
    if ty == "DURATION":
        return _parse(text, _duration)
    return _NOT_ONE


def _parse(text: str, fn) -> object:
    try:
        return fn(text)
    except ValueError:
        return _NOT_ONE


def _float(text: str) -> object:
    """A float, including the three spellings YAML has no opinion about.

    They are spelled the way Rust prints them, because that is what the
    reference runner writes into a failure report and what a case is
    pasted from."""
    if text == "NaN":
        return math.nan
    if text == "inf":
        return math.inf
    if text == "-inf":
        return -math.inf
    # A float is exact here, so `1` is not a FLOAT64 and neither is
    # `1e400`. The first is an integer somebody meant to write as `1.0`
    # and the second is `inf` under another name.
    if not any(c in text for c in ".eE"):
        return _NOT_ONE
    try:
        f = float(text)
    except ValueError:
        return _NOT_ONE
    return f if math.isfinite(f) else _NOT_ONE


def _to_float32(f: float) -> float:
    """The double a float rounds to when it is stored as one, which is
    what the engine gives back for a FLOAT32 column."""
    import struct

    return struct.unpack("<f", struct.pack("<f", f))[0]


def _finer_than_a_microsecond(text: str) -> bool:
    """Whether a temporal's text carries a digit Python's ``datetime``
    would drop. Read off the text rather than off the parsed value,
    because the parse is where the digits go."""
    _, dot, rest = text.partition(".")
    if not dot:
        return False
    digits = ""
    for c in rest:
        if not c.isdigit():
            break
        digits += c
    return any(c != "0" for c in digits[6:])


def _date(text: str) -> datetime.date:
    return datetime.date.fromisoformat(text)


def _local_time(text: str) -> datetime.time:
    t = datetime.time.fromisoformat(text)
    if t.tzinfo is not None:
        raise ValueError("a local time carries no offset")
    return t


def _zoned_time(text: str) -> datetime.time:
    t = datetime.time.fromisoformat(text.replace("Z", "+00:00"))
    if t.tzinfo is None:
        raise ValueError("a zoned time carries an offset")
    return t


def _local_datetime(text: str) -> datetime.datetime:
    d = datetime.datetime.fromisoformat(text)
    if d.tzinfo is not None:
        raise ValueError("a local datetime carries no offset")
    return d


def _zoned_datetime(text: str) -> datetime.datetime:
    d = datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
    if d.tzinfo is None:
        raise ValueError("a zoned datetime carries an offset")
    return d


def _duration(text: str) -> Duration:
    """An ISO 8601 duration, in the two kinds the engine keeps apart.

    A duration is months or it is nanoseconds and never both, because a
    month is not a number of days and adding one to a date is not the
    same operation. The text says which: a duration whose only fields
    are years and months is a year-month one, and everything else is
    day-time."""
    negative = text.startswith("-")
    if negative or text.startswith("+"):
        text = text[1:]
    if not text.startswith("P"):
        raise ValueError("a duration starts with P")
    body = text[1:]
    date_part, _, time_part = body.partition("T")
    months = 0
    nanos = 0
    for value, unit in _fields(date_part):
        if unit == "Y":
            months += int(value * 12)
        elif unit == "M":
            months += int(value)
        elif unit == "W":
            nanos += int(value * 7 * 86_400 * 1_000_000_000)
        elif unit == "D":
            nanos += int(value * 86_400 * 1_000_000_000)
        else:
            raise ValueError(f"{unit} is not a date field of a duration")
    for value, unit in _fields(time_part):
        if unit == "H":
            nanos += int(value * 3_600 * 1_000_000_000)
        elif unit == "M":
            nanos += int(value * 60 * 1_000_000_000)
        elif unit == "S":
            nanos += int(round(value * 1_000_000_000))
        else:
            raise ValueError(f"{unit} is not a time field of a duration")
    if months and nanos:
        raise ValueError("a duration is months or it is nanoseconds, not both")
    if not date_part and not time_part:
        raise ValueError("a duration with nothing in it")
    if negative:
        months, nanos = -months, -nanos
    return Duration(months=months, nanoseconds=nanos)


def _fields(text: str) -> list[tuple[float, str]]:
    out: list[tuple[float, str]] = []
    number = ""
    for c in text:
        if c.isdigit() or c in ".-+":
            number += c
            continue
        if not number:
            raise ValueError(f"{c} with no number before it")
        out.append((float(number), c))
        number = ""
    if number:
        raise ValueError(f"{number} with no unit after it")
    return out


def same(want: object, got: object) -> bool:
    """Whether two values are the same value.

    Not ``==``, for one reason: a float. ``NaN`` is not equal to itself
    and a case asserting ``NaN`` has to pass, and ``0.0`` equals
    ``-0.0`` and a case asserting ``-0.0`` has to fail on ``0.0``,
    because the sign of zero is exactly the sort of thing that survives
    one binding and not another.

    Python needs one rule its Rust counterpart does not: ``True == 1``
    there is false and here it is true, so a case asserting a boolean
    must not be answered with an integer, and the types are compared
    before the values are."""
    if isinstance(want, bool) != isinstance(got, bool):
        return False
    # A value this client cannot hold is equal to nothing, including
    # itself, so that a case carrying one never passes quietly.
    if isinstance(want, TooFine) or isinstance(got, TooFine):
        return False
    if isinstance(want, float) and isinstance(got, float):
        if math.isnan(want) and math.isnan(got):
            return True
        return math.copysign(1.0, want) == math.copysign(1.0, got) and want == got
    if isinstance(want, list) and isinstance(got, list):
        return len(want) == len(got) and all(same(a, b) for a, b in zip(want, got, strict=True))
    if isinstance(want, Walk) and isinstance(got, Walk):
        return same(list(want.elements), list(got.elements))
    if isinstance(want, dict) and isinstance(got, dict):
        return list(want) == list(got) and all(same(want[k], got[k]) for k in want)
    return type(want) is type(got) and want == got


def show(value: object) -> str:
    """How a value reads in a failure report, in the encoding's own
    spelling so that it can be pasted into a case, and line for line
    what the Rust runner prints so that two reports can be diffed."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return f"BOOL {'true' if value else 'false'}"
    if isinstance(value, int):
        return f'INT64 "{value}"'
    if isinstance(value, float):
        return f'FLOAT64 "{_show_float(value)}"'
    if isinstance(value, str):
        return f"STRING {quote(value)}"
    if isinstance(value, Duration):
        return f'DURATION "{_show_duration(value)}"'
    if isinstance(value, TooFine):
        return f'{value.ty} "{value.text}"'
    if isinstance(value, datetime.datetime):
        name = "ZONEDDATETIME" if value.tzinfo else "LOCALDATETIME"
        text = f"{_show_date(value)}T{_show_time(value)}{_show_offset(value)}"
        return f'{name} "{text}"'
    if isinstance(value, datetime.date):
        return f'DATE "{_show_date(value)}"'
    if isinstance(value, datetime.time):
        name = "ZONEDTIME" if value.tzinfo else "LOCALTIME"
        return f'{name} "{_show_time(value)}{_show_offset(value)}"'
    if isinstance(value, list):
        return f"LIST [{', '.join(show(item) for item in value)}]"
    if isinstance(value, dict):
        fields = ", ".join(f"{name}: {show(v)}" for name, v in value.items())
        return f"RECORD {{{fields}}}"
    if isinstance(value, Node):
        return f'NODE "{value.table}#{value.offset}"'
    if isinstance(value, Edge):
        return f'EDGE "{value.table}#{value.src}->{value.dst}"'
    if isinstance(value, Walk):
        return f"PATH [{', '.join(show(item) for item in value.elements)}]"
    # A `zudb.Rel` or a `zudb.Path` reaching here is a value the runner
    # did not put through `cell`, which is a defect in this runner rather
    # than a wrong answer, so it prints as itself and says so by not
    # looking like anything a case could be written from.
    return repr(value)


def _show_float(f: float) -> str:
    if math.isnan(f):
        return "NaN"
    if f == math.inf:
        return "inf"
    if f == -math.inf:
        return "-inf"
    # Rust's `{:?}` is the shortest text that reads back as the same
    # double and always carries a point or an exponent, which is what
    # `repr` gives here except for two things: a float that is a whole
    # number wants the point written, and an exponent is written bare
    # rather than with the sign and the padding Python puts on it.
    text = repr(f)
    if "e" in text:
        mantissa, _, exponent = text.partition("e")
        return f"{mantissa}e{int(exponent)}"
    return text if "." in text else text + ".0"


def _show_date(d: datetime.date) -> str:
    return f"{d.year:04}-{d.month:02}-{d.day:02}"


def _show_time(t: datetime.time | datetime.datetime) -> str:
    """A time the way the engine prints one, which is seconds always and
    a fraction of nine digits when there is one. Python writes six and
    drops them when they are zero, and a report that is diffed against
    the reference one cannot do either."""
    text = f"{t.hour:02}:{t.minute:02}:{t.second:02}"
    if t.microsecond:
        text += f".{t.microsecond * 1000:09d}"
    return text


def _show_offset(t: datetime.time | datetime.datetime) -> str:
    """An offset, which is ``Z`` at zero rather than ``+00:00``."""
    offset = t.utcoffset()
    if offset is None:
        return ""
    minutes = int(offset.total_seconds()) // 60
    if minutes == 0:
        return "Z"
    sign = "-" if minutes < 0 else "+"
    hours, rest = divmod(abs(minutes), 60)
    return f"{sign}{hours:02}:{rest:02}"


def _show_duration(d: Duration) -> str:
    """The ISO 8601 text the engine prints a duration as, which is the
    text it parses back: a field that is zero is left out, and a
    duration with nothing left in it is written ``PT0S`` or ``P0M``,
    because ``P`` on its own is not a value."""
    if d.kind == "year_month":
        count = d.months
        sign = "-" if count < 0 else ""
        years, months = divmod(abs(count), 12)
        out = f"{sign}P"
        if years:
            out += f"{years}Y"
        if months or not years:
            out += f"{months}M"
        return out
    nanos = d.nanoseconds
    sign = "-" if nanos < 0 else ""
    days, rest = divmod(abs(nanos), 86_400 * 1_000_000_000)
    out = f"{sign}P"
    if days:
        out += f"{days}D"
    if rest == 0 and days:
        return out
    out += "T"
    hours, rest = divmod(rest, 3_600 * 1_000_000_000)
    minutes, rest = divmod(rest, 60 * 1_000_000_000)
    seconds, fraction = divmod(rest, 1_000_000_000)
    if hours:
        out += f"{hours}H"
    if minutes:
        out += f"{minutes}M"
    if seconds or fraction or (not hours and not minutes):
        out += str(seconds)
        if fraction:
            out += f".{fraction:09d}"
        out += "S"
    return out
