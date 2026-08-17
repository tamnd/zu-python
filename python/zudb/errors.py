"""The exceptions zu raises, and the fields they carry.

Every failure the engine reports is a GQLSTATUS condition: a
five-character code from ISO/IEC 39075, a severity, and often the place
in the statement that raised it. All of that arrives here as fields on
the exception, so a caller reads ``e.code`` and never a regular
expression over ``str(e)``.

There is one class per condition class, which is what the two
characters that open a code are for. Catching :class:`DataError`
catches every one of the forty-two conditions in class 22 without
listing them, and a condition zu adds to that class later is caught by
the same ``except``.
"""

from __future__ import annotations

__all__ = [
    "Error",
    "ConnectionError",
    "DataError",
    "TransactionError",
    "SyntaxError",
    "ProgrammingError",
    "InternalError",
    "Interrupted",
]


class Error(Exception):
    """What every zu failure is, and the class to catch to catch them all.

    The fields are ``None`` when the condition has no answer for them
    rather than being filled with a guess. A division by zero happens
    at runtime and has no token to point at, so it carries a code and
    no position; a statement that failed to parse carries both.
    """

    #: The five-character GQLSTATUS code, ``"42001"`` for a syntax
    #: error. ``None`` for the few failures the standard has no
    #: condition for, such as a statement that was interrupted.
    code: str | None
    #: The standard's own words for the condition, never paraphrased,
    #: for example ``"syntax error or access rule violation, invalid
    #: syntax"``.
    condition: str | None
    #: ``"exception"``, ``"warning"``, ``"informational"``,
    #: ``"no_data"`` or ``"success"``.
    severity: str | None
    #: Line of the statement the condition was raised at, counting from
    #: one. ``None``, with ``column`` and ``offset``, when the
    #: condition happened somewhere the text cannot name.
    line: int | None
    #: Column on that line, counting from one, and a valid index into
    #: ``excerpt`` after subtracting that one.
    column: int | None
    #: Bytes into the statement, counting from zero, for a caller that
    #: wants to slice the text rather than print it.
    offset: int | None
    #: The whole line the position is on, quoted out of the statement,
    #: for the caller who has the error and no longer has the text.
    excerpt: str | None
    #: The page that documents this condition.
    doc_url: str | None
    #: Whether running the same statement again could succeed. True
    #: only for a transaction that rolled back with nothing done.
    retryable: bool

    def __init__(
        self,
        message: str = "",
        *,
        code: str | None = None,
        condition: str | None = None,
        severity: str | None = None,
        line: int | None = None,
        column: int | None = None,
        offset: int | None = None,
        excerpt: str | None = None,
        doc_url: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.condition = condition
        self.severity = severity
        self.line = line
        self.column = column
        self.offset = offset
        self.excerpt = excerpt
        self.doc_url = doc_url
        self.retryable = retryable

    def caret(self) -> str | None:
        """The excerpt with a caret under the column, ready to print.

        ``None`` when there is no excerpt to point at. This is the one
        piece of formatting the library does, because every caller that
        prints an error writes it otherwise and half of them count the
        column wrong.
        """
        if self.excerpt is None or self.column is None:
            return None
        return f"{self.excerpt}\n{' ' * (self.column - 1)}^"


class ConnectionError(Error):
    """Class 08: the database could not be reached or could not be read."""


class DataError(Error):
    """Class 22: a value was wrong. Division by zero, a bad cast, a number that did not fit."""


class TransactionError(Error):
    """Classes 25, 2D and 40: the transaction, rather than the statement, is what went wrong.

    Check :attr:`Error.retryable` before running it again. A rollback
    with nothing done can be retried; a statement whose completion is
    unknown cannot, because a retry could do the work twice.
    """


class SyntaxError(Error):  # noqa: A001 - the standard's name for class 42
    """Class 42: the statement could not be parsed, or named something that is not there."""


class ProgrammingError(Error):
    """The caller made a mistake in Python.

    A statement on a closed connection, a parameter of a type zu has no
    place for. Nothing reached the engine, so nothing happened to the
    database.
    """


class InternalError(Error):
    """A failure the engine could not describe as a condition.

    A corrupt file, an assumption that did not hold. Worth reporting at
    https://github.com/tamnd/zu/issues.
    """


class Interrupted(Error):
    """The statement was asked to stop and did.

    Nothing failed and nothing was lost: the connection keeps its plans
    and its caches and runs the next statement normally. A ``Ctrl-C``
    raises ``KeyboardInterrupt`` instead, since that is what a person
    at a keyboard means by it.
    """
