"""zu for Python: an embedded property-graph database, in your process.

    import zudb

    with zudb.connect("social.zu1") as conn:
        rows = conn.execute("MATCH (p:person) RETURN p.name AS name")
        for (name,) in rows:
            print(name)

The engine is compiled into the wheel, so there is nothing to install,
nothing to run, and no server to connect to. Statements are ISO/IEC
39075 GQL. `connect()` with no path is a database in memory, which
makes no file anywhere and is gone when the last connection to it is.

On an event loop the same calls are awaited, from `zudb.aio`. Code
written against PEP 249 gets what it expects from `zudb.dbapi`. Both
are submodules to ask for by name rather than ones imported here, so a
script that uses neither pays for neither.
"""

from __future__ import annotations

from ._zudb import (
    Appender,
    Connection,
    Duration,
    Node,
    Path,
    Plan,
    PlanNode,
    Prepared,
    Profile,
    ProfileOp,
    ProfileStage,
    Rel,
    Result,
    ScalarPlan,
    Stream,
    StreamBatches,
    StreamSummary,
    Transaction,
    __abi_version__,
    connect,
    load,
)
from .errors import (
    ConnectionError,
    DataError,
    Error,
    InternalError,
    Interrupted,
    ProgrammingError,
    SyntaxError,
    TransactionError,
)
from .types import Value

__version__ = "0.0.1"

__all__ = [
    "connect",
    "load",
    "Connection",
    "Transaction",
    "Appender",
    "Prepared",
    "Result",
    "Stream",
    "StreamBatches",
    "StreamSummary",
    "Plan",
    "PlanNode",
    "ScalarPlan",
    "Profile",
    "ProfileStage",
    "ProfileOp",
    "Node",
    "Rel",
    "Path",
    "Duration",
    "Value",
    "Error",
    "ConnectionError",
    "DataError",
    "TransactionError",
    "SyntaxError",
    "ProgrammingError",
    "InternalError",
    "Interrupted",
    "__version__",
    "__abi_version__",
]
