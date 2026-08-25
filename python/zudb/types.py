"""The type of a value, for a caller who annotates.

One name for the union a row holds and a parameter takes, so a function
that passes rows around can say what it passes. Written here rather than
in the stub for the compiled module because a type alias a checker knows
and the interpreter does not is a name that fails at the first
``from zudb import Value``.
"""

from __future__ import annotations

import datetime
import decimal
from typing import TypeAlias

from ._zudb import Duration, Node, Path, Rel

__all__ = ["Value"]

#: What a statement gives back and what a parameter may be. Recursive,
#: because a list holds values and one of them may be a list. A
#: ``timedelta`` goes in and never comes out: zu stores it as a day-time
#: duration and hands one back, since a ``timedelta`` cannot hold every
#: duration zu can. ``bytes`` and not ``bytearray``, because a parameter
#: is read after the call that takes it returns and a mutable buffer is a
#: promise the caller can break. A ``decimal.Decimal`` goes both ways
#: and is the one exact number here: a price read into a ``float`` would
#: not be the price, which is why the engine has a type for it at all.
Value: TypeAlias = (
    None
    | bool
    | int
    | float
    | decimal.Decimal
    | str
    | bytes
    | datetime.date
    | datetime.time
    | datetime.datetime
    | datetime.timedelta
    | Duration
    | Node
    | Rel
    | Path
    | list["Value"]
    | dict[str, "Value"]
)
