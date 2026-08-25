"""What an engine value is once it is a Python object."""

from __future__ import annotations

import datetime
import decimal

import pytest
import zudb


def test_a_node_carries_its_table_and_offset(social: zudb.Connection) -> None:
    rows = social.execute("MATCH (p:person) RETURN p ORDER BY p.uid")
    nodes = [node for (node,) in rows]
    assert [node.table for node in nodes] == ["person"] * 3
    assert [node.offset for node in nodes] == [0, 1, 2]
    assert repr(nodes[0]) == "Node(person, 0)"


def test_two_reads_of_one_node_are_equal_and_hash_alike(social: zudb.Connection) -> None:
    one = social.execute("MATCH (p:person) WHERE p.uid = 10 RETURN p").fetchone()[0]
    two = social.execute("MATCH (p:person) WHERE p.uid = 10 RETURN p").fetchone()[0]
    assert one == two
    assert len({one, two}) == 1


@pytest.mark.parametrize(
    "statement,answer",
    [
        ("RETURN 1 AS v", 1),
        ("RETURN 1.5 AS v", 1.5),
        ("RETURN 'ada' AS v", "ada"),
        ("RETURN X'00AB00' AS v", b"\x00\xab\x00"),
        ("RETURN X'' AS v", b""),
        ("RETURN true AS v", True),
        ("RETURN false AS v", False),
        ("RETURN null AS v", None),
        ("RETURN [1, 2, 3] AS v", [1, 2, 3]),
        ("RETURN [[1], [2, 3]] AS v", [[1], [2, 3]]),
        ("RETURN [] AS v", []),
    ],
)
def test_a_literal_reads_back_as_the_python_object_it_is(
    empty: zudb.Connection, statement: str, answer: object
) -> None:
    got = empty.execute(statement).fetchone()[0]
    assert got == answer
    assert type(got) is type(answer)


def test_a_byte_string_is_bytes_and_not_the_string_that_spells_it(
    empty: zudb.Connection,
) -> None:
    """A byte string is octets and a string is characters, and the whole
    reason the type exists is that the two are not the same value."""
    (same,) = empty.execute("RETURN X'0041' = 'A' AS v").fetchone()
    assert same is not True
    got = empty.execute("RETURN X'0041' AS v").fetchone()[0]
    assert got == b"\x00A"
    assert got != "\x00A"


def test_a_byte_string_goes_in_as_a_parameter_and_comes_back_the_same(
    empty: zudb.Connection,
) -> None:
    got = empty.execute("RETURN $b AS v", {"b": b"\xde\xad\xbe\xef"}).fetchone()[0]
    assert got == b"\xde\xad\xbe\xef"
    assert type(got) is bytes


def test_a_mutable_buffer_is_not_a_byte_string_parameter(empty: zudb.Connection) -> None:
    """A parameter is read after the call that takes it returns, so a
    buffer the caller can still write through is a promise this client
    does not take. The refusal names the types it does take, which is
    the shortest way to say `bytes(...)`."""
    with pytest.raises(TypeError, match="byte strings"):
        empty.execute("RETURN $b AS v", {"b": bytearray(b"\x00")})


def test_a_duration_counts_months_or_nanoseconds_and_never_both() -> None:
    with pytest.raises(ValueError, match="never both"):
        zudb.Duration(months=1, nanoseconds=1)


def test_a_duration_says_which_kind_it_is() -> None:
    assert zudb.Duration(months=14).kind == "year_month"
    assert zudb.Duration(nanoseconds=1).kind == "day_time"
    assert zudb.Duration().kind == "day_time"


def test_a_day_time_duration_converts_to_a_timedelta() -> None:
    hour = zudb.Duration(nanoseconds=3_600_000_000_000)
    assert hour.to_timedelta() == datetime.timedelta(hours=1)


def test_a_conversion_that_rounds_rounds_towards_zero() -> None:
    assert zudb.Duration(nanoseconds=1_500).to_timedelta() == datetime.timedelta(microseconds=1)
    assert zudb.Duration(nanoseconds=-1_500).to_timedelta() == datetime.timedelta(microseconds=-1)


def test_a_year_month_duration_has_no_timedelta() -> None:
    with pytest.raises(ValueError, match="no timedelta"):
        zudb.Duration(months=3).to_timedelta()


def test_a_duration_repr_names_the_count_it_carries() -> None:
    assert repr(zudb.Duration(months=3)) == "Duration(months=3)"
    assert repr(zudb.Duration(nanoseconds=3)) == "Duration(nanoseconds=3)"


def test_a_decimal_keeps_the_digits_it_was_written_with(empty: zudb.Connection) -> None:
    """A tenth is not a binary fraction, so a price held as a float is
    not the price and neither is the two places it printed at. The whole
    reason the engine has the type is that both survive."""
    got = empty.execute("RETURN CAST('1.20' AS DECIMAL(5, 2)) AS v").fetchone()[0]
    assert type(got) is decimal.Decimal
    assert got == decimal.Decimal("1.20")
    assert str(got) == "1.20"
    assert got.as_tuple().exponent == -2


@pytest.mark.parametrize(
    "text",
    ["0", "1.20", "-0.05", "1234", "-1234.5678", "0.005", "0.000"],
)
def test_a_decimal_reads_back_as_the_number_and_the_scale_it_was_cast_at(
    empty: zudb.Connection, text: str
) -> None:
    places = len(text.partition(".")[2])
    statement = f"RETURN CAST('{text}' AS DECIMAL(38, {places})) AS v"
    got = empty.execute(statement).fetchone()[0]
    assert got == decimal.Decimal(text)
    assert str(got) == text


def test_a_decimal_is_not_the_float_that_looks_like_it(empty: zudb.Connection) -> None:
    """0.1 as a float is a number slightly larger than a tenth, and this
    is the case a client that reached for `float` would pass by getting
    both of them wrong the same way."""
    got = empty.execute("RETURN CAST('0.1' AS DECIMAL(5, 1)) AS v").fetchone()[0]
    assert got == decimal.Decimal("0.1")
    assert got != decimal.Decimal(0.1)
    assert float(got) == 0.1


def test_a_decimal_wider_than_an_int64_arrives_whole(empty: zudb.Connection) -> None:
    """Thirty eight digits is the largest precision DECIMAL(p, s) takes
    and the largest the engine's carrier holds, so this is the widest
    value that can come across."""
    digits = "12345678901234567890123456789012345678"
    got = empty.execute(f"RETURN CAST('{digits}' AS DECIMAL(38, 0)) AS v").fetchone()[0]
    assert got == decimal.Decimal(digits)


def test_a_decimal_goes_in_as_a_parameter_and_comes_back_the_same(
    empty: zudb.Connection,
) -> None:
    sent = decimal.Decimal("1.20")
    got = empty.execute("RETURN $d AS v", {"d": sent}).fetchone()[0]
    assert got == sent
    assert str(got) == "1.20"
    assert type(got) is decimal.Decimal


def test_a_decimal_parameter_is_not_read_as_a_float(empty: zudb.Connection) -> None:
    """`decimal.Decimal` is neither an int nor a float, so a client that
    fell through to `float` here would send a number that is not the one
    the caller built, and the caller picked the type to avoid exactly
    that."""
    got = empty.execute("RETURN $d AS v", {"d": decimal.Decimal("0.1")}).fetchone()[0]
    assert type(got) is decimal.Decimal
    assert got == decimal.Decimal("0.1")


def test_a_decimal_parameter_written_with_an_exponent_arrives_written_out(
    empty: zudb.Connection,
) -> None:
    """Python prints some decimals with an exponent and `Decimal("1E+2")`
    is a hundred at no places rather than a one at two of them, so the
    scale is taken off the digits written out rather than off the
    spelling."""
    got = empty.execute("RETURN $d AS v", {"d": decimal.Decimal("1E+2")}).fetchone()[0]
    assert got == decimal.Decimal(100)
    assert str(got) == "100"


@pytest.mark.parametrize("text", ["NaN", "-NaN", "sNaN", "Infinity", "-Infinity"])
def test_a_decimal_that_is_not_a_number_is_refused_at_the_call(
    empty: zudb.Connection, text: str
) -> None:
    """These are values `decimal.Decimal` holds and an exact number is
    not, so there is nothing to send. Failing here is the point: a
    parameter that arrived as something else would be a query answering
    about a value nobody asked about."""
    with pytest.raises(ValueError, match="exact number"):
        empty.execute("RETURN $d AS v", {"d": decimal.Decimal(text)})


def test_a_decimal_with_more_places_than_the_engine_holds_says_so(
    empty: zudb.Connection,
) -> None:
    with pytest.raises(ValueError, match="after the point"):
        empty.execute("RETURN $d AS v", {"d": decimal.Decimal("1E-100")})


def test_a_decimal_wider_than_the_carrier_says_so(empty: zudb.Connection) -> None:
    with pytest.raises(ValueError, match="at most 38"):
        empty.execute("RETURN $d AS v", {"d": decimal.Decimal("1" * 40)})
