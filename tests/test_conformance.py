"""The corpus reader and runner, checked against the reference ones.

The corpus is one set of files read by three readers, and the whole
value of a second reader is that it refuses what the first refuses. So
the tables below are the tables in `crates/zu-corpus/src/*.rs`, case for
case: a document, and the words the refusal has to contain. A reader
that grew a hole would pass its own tests and fail these.

The run over the cases themselves needs the cases, which live in the
engine's repository rather than this one. `ZU_CASES` points at them and
the run is skipped without it, so a checkout with no engine beside it
still tests everything that does not need one.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
import zudb

from conformance import arrow, cases, reader, runner, values
from conformance.cases import MAIN

CASES = os.environ.get("ZU_CASES")

needs_cases = pytest.mark.skipif(not CASES, reason="ZU_CASES does not point at the case files")

HEAD = "schema: 4\nsuite: int\ndoc: the integer tower\n"


def rows_of(value: str, ty: str = "INT64", column: str = "n") -> str:
    """One column and one row of it, which is the shape most of these
    fixtures want and none of them is about."""
    return (
        f"    columns:\n      - {column}\n    rows:\n      - values:\n"
        f'          - type: {ty}\n            value: "{value}"\n'
    )


def suite(text: str) -> cases.Suite:
    return cases.read(f"{HEAD}\ncases:\n{text}")


def one(text: str) -> cases.Case:
    return suite(text).cases[0]


def value(text: str) -> object:
    return values.decode(reader.parse(text))


# The YAML subset.


def test_a_mapping_of_scalars_is_the_shape_everything_else_is_made_of() -> None:
    doc = reader.parse("schema: 1\nsuite: int\n")
    assert doc.get("schema").str_() == "1"
    assert doc.get("suite").str_() == "int"
    assert doc.get("nothing") is None


def test_a_sequence_item_carrying_a_mapping_is_the_same_shape_as_one_written_out() -> None:
    doc = reader.parse(
        "cases:\n  - name: a\n    query: RETURN 1\n  - name: b\n    query: RETURN 2\n"
    )
    items = doc.get("cases").seq()
    assert len(items) == 2
    assert items[0].get("name").str_() == "a"
    assert items[0].get("query").str_() == "RETURN 1"
    assert items[1].get("name").str_() == "b"


def test_a_sequence_of_scalars_is_not_read_as_anything_cleverer() -> None:
    doc = reader.parse("columns:\n  - n\n  - m\n")
    assert [c.str_() for c in doc.get("columns").seq()] == ["n", "m"]


def test_nesting_goes_as_deep_as_a_list_of_records_needs() -> None:
    doc = reader.parse(
        "rows:\n  - values:\n      - type: LIST\n        value:\n          - type: INT64\n"
        '            value: "1"\n'
    )
    row = doc.get("rows").seq()[0].get("values").seq()[0]
    assert row.get("type").str_() == "LIST"
    assert row.get("value").seq()[0].get("value").str_() == "1"


def test_whether_a_scalar_was_quoted_survives_because_the_encoding_turns_on_it() -> None:
    doc = reader.parse("bare: 42\nquoted: \"42\"\nsingle: '42'\n")
    assert doc.get("bare").scalar() == ("42", False)
    assert doc.get("quoted").scalar() == ("42", True)
    assert doc.get("single").scalar() == ("42", True)


def test_a_colon_inside_a_value_is_part_of_it_and_not_another_key() -> None:
    doc = reader.parse("query: RETURN datetime('2024-01-01T00:00:00')\n")
    assert doc.get("query").str_() == "RETURN datetime('2024-01-01T00:00:00')"


def test_a_hash_is_a_comment_only_where_a_comment_can_start() -> None:
    doc = reader.parse(
        '# the whole line\nname: a  # and the end of this one\nhash: "a # b"\nword: c#d\n'
    )
    assert doc.get("name").str_() == "a"
    assert doc.get("hash").str_() == "a # b"
    assert doc.get("word").str_() == "c#d"


def test_a_quote_that_never_closes_is_an_ordinary_character_inside_a_plain_scalar() -> None:
    """The second quote has a space before it, so it looks like the start
    of a run, and there is nothing after it to close one."""
    doc = reader.parse("query: RETURN cast('  42  ' AS INT64) AS n  # a note\n")
    assert doc.get("query").str_() == "RETURN cast('  42  ' AS INT64) AS n"


def test_an_escape_and_a_doubled_quote_are_the_two_ways_a_quote_gets_in() -> None:
    doc = reader.parse("a: \"say \\\"no\\\"\"\nb: 'say ''no'''\n")
    assert doc.get("a").str_() == 'say "no"'
    assert doc.get("b").str_() == "say 'no'"


def test_the_control_characters_a_query_can_hold_all_have_an_escape() -> None:
    doc = reader.parse('a: "one\\ntwo\\rthree\\tfour\\0five"\n')
    assert doc.get("a").str_() == "one\ntwo\rthree\tfour\0five"


def test_a_negative_number_is_a_scalar_and_not_a_sequence() -> None:
    assert reader.parse("value: -1\n").get("value").str_() == "-1"


def test_every_node_says_which_line_it_started_on() -> None:
    doc = reader.parse("schema: 1\n\ncases:\n  - name: a\n    query: RETURN 1\n")
    case = doc.get("cases").seq()[0]
    assert case.line == 4
    assert case.get("query").line == 5


def test_a_key_with_nothing_under_it_is_a_node_and_every_accessor_says_no_to_it() -> None:
    rows = reader.parse("rows:\n").get("rows")
    assert rows.what() == "nothing"
    assert rows.str_() is None
    assert rows.seq() is None
    # The one caller for whom empty is an answer, which is a case that
    # expects no rows back.
    assert rows.seq_or_empty() == ()


def test_a_key_nobody_reads_can_be_asked_for() -> None:
    doc = reader.parse("name: a\nqeury: RETURN 1\n")
    assert doc.unknown(("name", "query")) == ["qeury"]


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("", "nothing in it"),
        ("  name: a\n", "line 1"),
        ("name:\ta\n", "line 1"),
        ("name: a\n   doc: b\n", "line 2"),
        ("a: 1\n b: 2\n", "line 2"),
        ("name: a\nname: b\n", "line 2"),
        ("cases:\n  -\n", "line 2"),
        ("cases:\n  -  name: a\n", "line 2"),
        ("cases:\n  - - a\n", "line 2"),
        ("columns: [n, m]\n", "line 1"),
        ("doc: >\n  folded\n", "line 1"),
        ("doc: |\n  literal\n", "line 1"),
        ("anchor: &a 1\n", "line 1"),
        ("---\nname: a\n", "line 1"),
        ('a: "unterminated\n', "line 1"),
        ('a: "bad \\q escape"\n', "line 1"),
        ("a: 'unterminated\n", "line 1"),
        ("a: 1\ncases:\n      - b\n", "line 3"),
    ],
)
def test_what_the_reader_does_not_read_it_refuses_and_says_where(text: str, want: str) -> None:
    with pytest.raises(reader.CorpusError) as raised:
        reader.parse(text)
    assert want in str(raised.value)


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ('a: "one" and "two"\n', "after the scalar ends"),
        ("a: [1]\n", "a plain scalar opening with '['"),
        ("...\na: 1\n", '"..." opens or closes'),
    ],
)
def test_a_refusal_quotes_what_it_found_the_way_the_reference_reader_does(
    text: str, want: str
) -> None:
    """Rust writes a string with double quotes and a character with
    single ones, and a report that is diffed against the reference one
    has to write them the same way."""
    with pytest.raises(reader.CorpusError) as raised:
        reader.parse(text)
    assert want in str(raised.value)


def test_the_quoting_helper_writes_what_rust_writes() -> None:
    assert reader.quote("a") == '"a"'
    assert reader.quote('say "no"') == '"say \\"no\\""'
    assert reader.quote("one\ntwo") == '"one\\ntwo"'
    assert reader.quote("a\\b") == '"a\\\\b"'


# The value encoding.


def test_the_widths_a_yaml_number_carries_are_written_as_numbers() -> None:
    assert value("type: INT8\nvalue: -128\n") == -128
    assert value("type: INT32\nvalue: 2147483647\n") == 2147483647
    assert value("type: BOOL\nvalue: true\n") is True
    assert value("type: NULL\n") is None


def test_the_widths_it_does_not_carry_are_written_as_strings() -> None:
    assert value('type: INT64\nvalue: "9223372036854775807"\n') == 2**63 - 1
    assert value('type: INT64\nvalue: "-9223372036854775808"\n') == -(2**63)


def test_an_int64_written_bare_is_the_defect_the_encoding_exists_to_stop() -> None:
    with pytest.raises(reader.CorpusError) as raised:
        value("type: INT64\nvalue: 9223372036854775807\n")
    assert "written in quotes" in str(raised.value)
    assert "will round it" in str(raised.value)


def test_a_number_written_in_quotes_is_refused_the_other_way_round() -> None:
    with pytest.raises(reader.CorpusError) as raised:
        value('type: INT8\nvalue: "42"\n')
    assert "without quotes" in str(raised.value)
    # A string is the one type whose payload is quoted or not as YAML
    # pleases, because either way it is the same text.
    assert value('type: STRING\nvalue: "42"\n') == "42"
    assert value("type: STRING\nvalue: 42\n") == "42"


@pytest.mark.parametrize(
    "text",
    [
        "type: INT8\nvalue: 128\n",
        "type: UINT8\nvalue: -1\n",
        "type: INT32\nvalue: 2147483648\n",
        'type: UINT64\nvalue: "18446744073709551615"\n',
    ],
)
def test_a_value_too_wide_for_the_type_it_claims_is_not_quietly_widened(text: str) -> None:
    with pytest.raises(reader.CorpusError) as raised:
        value(text)
    assert "is not a" in str(raised.value)


def test_a_float_says_which_float_because_the_three_awkward_ones_have_no_yaml_spelling() -> None:
    assert value('type: FLOAT64\nvalue: "1.5"\n') == 1.5
    assert math.isnan(value('type: FLOAT64\nvalue: "NaN"\n'))
    assert value('type: FLOAT64\nvalue: "inf"\n') == math.inf
    assert value('type: FLOAT64\nvalue: "-inf"\n') == -math.inf
    # A negative zero is a different value from a zero, and saying so is
    # why `same` compares bits.
    minus = value('type: FLOAT64\nvalue: "-0.0"\n')
    assert not values.same(minus, 0.0)
    assert values.same(minus, -0.0)


def test_a_nan_matches_a_nan_whatever_the_hardware_put_in_its_sign_and_payload() -> None:
    want = value('type: FLOAT64\nvalue: "NaN"\n')
    assert values.same(want, math.nan)
    assert values.same(want, -math.nan)
    assert values.same(want, float("nan"))
    assert not values.same(want, 0.0)


def test_a_float32_is_narrowed_so_a_case_asserts_what_the_narrower_type_can_hold() -> None:
    narrowed = value('type: FLOAT32\nvalue: "0.1"\n')
    assert not values.same(narrowed, 0.1)
    assert values.same(narrowed, narrowed)


def test_an_integer_written_as_a_float_is_refused_rather_than_promoted() -> None:
    with pytest.raises(reader.CorpusError) as raised:
        value('type: FLOAT64\nvalue: "1"\n')
    assert "is not a FLOAT64" in str(raised.value)


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ('type: DATE\nvalue: "2024-02-29"\n', "2024-02-29"),
        ('type: LOCALTIME\nvalue: "12:34:56"\n', "12:34:56"),
        ('type: ZONEDDATETIME\nvalue: "2024-01-01T00:00:00+07:00"\n', "2024-01-01T00:00:00+07:00"),
        ('type: DURATION\nvalue: "P1Y2M"\n', "P1Y2M"),
    ],
)
def test_a_temporal_is_written_the_way_the_engine_prints_it(text: str, want: str) -> None:
    got = values.show(value(text))
    assert got.split(" ", 1)[1] == f'"{want}"'


def test_a_duration_carries_its_sign_because_a_difference_can_go_either_way() -> None:
    assert value('type: DURATION\nvalue: "-PT1H"\n') == zudb.Duration(nanoseconds=-3600_000_000_000)


def test_a_list_holds_encoded_values_and_not_bare_ones() -> None:
    assert value(
        "type: LIST\nvalue:\n  - type: INT8\n    value: 1\n  - type: STRING\n    value: two\n"
    ) == [1, "two"]
    with pytest.raises(reader.CorpusError) as raised:
        value("type: LIST\nvalue:\n  - 1\n")
    assert "a value is a mapping" in str(raised.value)
    # The empty list is a value, and a `value:` with nothing under it is
    # how it is written.
    assert value("type: LIST\nvalue:\n") == []


def test_a_node_is_the_name_of_its_table_and_the_row_it_is() -> None:
    assert value('type: NODE\nvalue: "person#1"\n') == zudb.Node("person", 1)
    # The last hash is the separator, so a table whose name holds one is
    # still read the way it was written.
    assert value('type: NODE\nvalue: "a#b#2"\n') == zudb.Node("a#b", 2)


def test_an_edge_is_read_without_the_field_no_case_writes() -> None:
    """The engine's edge carries the position it holds among the edges
    out of its source, which the loader chose and no case picked. It is
    dropped on both sides rather than guessed at on one."""
    assert value('type: EDGE\nvalue: "knows#0->1"\n') == values.Edge("knows", 0, 1)
    assert values.cell(zudb.Rel("knows", 0, 1, 7)) == values.Edge("knows", 0, 1)


def test_a_path_alternates_nodes_and_edges_and_has_to_start_and_end_on_a_node() -> None:
    walk = value(
        'type: PATH\nvalue:\n  - type: NODE\n    value: "person#0"\n'
        '  - type: EDGE\n    value: "knows#0->1"\n  - type: NODE\n    value: "person#1"\n'
    )
    assert walk == values.Walk(
        (zudb.Node("person", 0), values.Edge("knows", 0, 1), zudb.Node("person", 1))
    )
    # The one-node path is the shortest one there is, and it is a path.
    assert value('type: PATH\nvalue:\n  - type: NODE\n    value: "person#0"\n') == values.Walk(
        (zudb.Node("person", 0),)
    )


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("type: NODE\nvalue: person#1\n", "is a name and two numbers"),
        ("type: EDGE\nvalue: knows#0->1\n", "is a name and two numbers"),
        ('type: NODE\nvalue: "person"\n', '"person" is not a NODE'),
        ('type: NODE\nvalue: "person#x"\n', '"person#x" is not a NODE'),
        ('type: NODE\nvalue: "#1"\n', '"#1" is not a NODE'),
        ('type: EDGE\nvalue: "knows#0"\n', '"knows#0" is not a EDGE'),
        ('type: EDGE\nvalue: "knows#a->b"\n', '"knows#a->b" is not a EDGE'),
        ('type: EDGE\nvalue: "knows#0->"\n', '"knows#0->" is not a EDGE'),
        (
            'type: PATH\nvalue:\n  - type: NODE\n    value: "person#0"\n'
            '  - type: EDGE\n    value: "knows#0->1"\n',
            "an odd number",
        ),
        (
            'type: PATH\nvalue:\n  - type: EDGE\n    value: "knows#0->1"\n'
            '  - type: NODE\n    value: "person#0"\n  - type: EDGE\n    value: "knows#0->1"\n',
            "alternates",
        ),
        ("type: PATH\nvalue:\n", "an odd number"),
    ],
)
def test_a_graph_value_written_wrong_is_refused_where_it_is_written(text: str, want: str) -> None:
    with pytest.raises(reader.CorpusError) as raised:
        value(text)
    assert want in str(raised.value)


def test_a_graph_value_prints_the_way_a_case_writes_one() -> None:
    assert values.show(zudb.Node("person", 1)) == 'NODE "person#1"'
    assert values.show(values.Edge("knows", 0, 1)) == 'EDGE "knows#0->1"'
    assert (
        values.show(values.Walk((zudb.Node("person", 0), values.Edge("knows", 0, 1))))
        == 'PATH [NODE "person#0", EDGE "knows#0->1"]'
    )


def test_a_type_the_engine_cannot_hold_yet_says_so_rather_than_looking_like_a_typo() -> None:
    with pytest.raises(reader.CorpusError) as raised:
        value('type: DECIMAL\nvalue: "1.00"\n')
    assert "reserves" in str(raised.value)
    with pytest.raises(reader.CorpusError) as raised:
        value('type: INT65\nvalue: "1"\n')
    assert "not a type this encoding knows" in str(raised.value)


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("type: INT8\n", "with no `value`"),
        ("value: 1\n", "with no `type`"),
        ("type: NULL\nvalue: 1\n", "carries no `value`"),
        ("type: INT8\nvalue: 1\nnote: hi\n", 'no key "note"'),
        ("type: INT8\nvalue:\n  - 1\n", "holds one scalar"),
    ],
)
def test_a_mapping_that_is_not_a_value_is_refused_with_its_line(text: str, want: str) -> None:
    with pytest.raises(reader.CorpusError) as raised:
        value(text)
    assert want in str(raised.value)


def test_what_a_report_prints_is_what_a_case_would_be_written_as() -> None:
    assert values.show(7) == 'INT64 "7"'
    assert values.show(1.0) == 'FLOAT64 "1.0"'
    assert values.show(math.nan) == 'FLOAT64 "NaN"'
    assert values.show(None) == "NULL"
    assert values.show([True, "a"]) == 'LIST [BOOL true, STRING "a"]'


@pytest.mark.parametrize(
    ("f", "want"),
    [
        (1.0, "1.0"),
        (0.1, "0.1"),
        (-0.0, "-0.0"),
        (1e16, "1e16"),
        (1e300, "1e300"),
        (1e-7, "1e-7"),
        (3.4028234663852886e38, "3.4028234663852886e38"),
    ],
)
def test_a_float_prints_the_way_rust_prints_one(f: float, want: str) -> None:
    """Python writes an exponent with a sign and a padded width and Rust
    writes it bare, and a report diffed against the reference one cannot
    have either."""
    assert values.show(f) == f'FLOAT64 "{want}"'


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ('type: LOCALTIME\nvalue: "12:34:56"\n', 'LOCALTIME "12:34:56"'),
        ('type: LOCALTIME\nvalue: "12:34:56.789"\n', 'LOCALTIME "12:34:56.789000000"'),
        ('type: ZONEDTIME\nvalue: "12:34:56Z"\n', 'ZONEDTIME "12:34:56Z"'),
        ('type: ZONEDTIME\nvalue: "12:34:56+00:00"\n', 'ZONEDTIME "12:34:56Z"'),
        ('type: ZONEDTIME\nvalue: "12:34:56-05:30"\n', 'ZONEDTIME "12:34:56-05:30"'),
        ('type: DATE\nvalue: "0001-01-01"\n', 'DATE "0001-01-01"'),
        (
            'type: ZONEDDATETIME\nvalue: "2024-01-01T00:00:00+07:00"\n',
            'ZONEDDATETIME "2024-01-01T00:00:00+07:00"',
        ),
    ],
)
def test_a_temporal_prints_the_way_the_engine_prints_one(text: str, want: str) -> None:
    """Nine digits of fraction and never six, seconds always, and ``Z``
    at a zero offset."""
    assert values.show(value(text)) == want


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("P1Y2M", "P1Y2M"),
        ("P1Y", "P1Y"),
        ("P2M", "P2M"),
        ("PT1S", "PT1S"),
        ("PT1H30M", "PT1H30M"),
        ("P1D", "P1D"),
        ("PT0S", "PT0S"),
        ("-PT1H", "-PT1H"),
        ("PT0.5S", "PT0.500000000S"),
        ("P1DT2H", "P1DT2H"),
    ],
)
def test_a_duration_leaves_out_the_fields_that_are_zero(text: str, want: str) -> None:
    """The text a duration prints as is the text it parses back, so a
    field that is zero is left out and a duration with nothing left in
    it is written out anyway."""
    assert values.show(value(f'type: DURATION\nvalue: "{text}"\n')) == f'DURATION "{want}"'


def test_a_zero_duration_is_the_one_the_two_kinds_cannot_be_told_apart_by() -> None:
    """The engine keeps which kind a duration is beside the count, and
    this client keeps two fields and reads the kind off whichever is set,
    so a zero of either kind is a zero of the other. The engine writes
    `P0M` for one and `PT0S` for the other and this writes `PT0S` for
    both. No case turns on it, and it is written down here rather than
    left for somebody to find."""
    assert values.show(value('type: DURATION\nvalue: "P0M"\n')) == 'DURATION "PT0S"'
    assert values.show(value('type: DURATION\nvalue: "PT0S"\n')) == 'DURATION "PT0S"'


def test_a_time_finer_than_this_client_holds_is_equal_to_nothing() -> None:
    """Python's ``datetime`` keeps microseconds and the engine keeps
    nanoseconds, so a case asserting nine digits would otherwise pass by
    comparing one truncated value against another, which is what it was
    written to catch."""
    fine = value('type: LOCALTIME\nvalue: "12:34:56.123456789"\n')
    assert isinstance(fine, values.TooFine)
    assert values.show(fine) == 'LOCALTIME "12:34:56.123456789"'
    assert not values.same(fine, fine)
    assert not values.same(fine, value('type: LOCALTIME\nvalue: "12:34:56.123456"\n'))
    assert values.too_fine([1, [fine]]) is fine
    assert values.too_fine([1, "a"]) is None


def test_a_time_the_client_cannot_hold_still_truncates_for_a_load() -> None:
    """A column has to go into the file for the suite to have a graph at
    all, and the offset behind the fraction is part of the value rather
    than part of what is cut."""
    import datetime

    fine = value('type: LOCALTIME\nvalue: "23:59:59.999999999"\n')
    assert values.truncated(fine) == datetime.time(23, 59, 59, 999999)
    zoned = value('type: ZONEDTIME\nvalue: "12:34:56.123456789+07:00"\n')
    assert values.truncated(zoned).utcoffset() == datetime.timedelta(hours=7)
    assert values.truncated([fine, 1]) == [datetime.time(23, 59, 59, 999999), 1]


def test_a_fraction_of_six_digits_or_fewer_is_a_value_this_client_holds() -> None:
    assert not isinstance(value('type: LOCALTIME\nvalue: "12:34:56.123456"\n'), values.TooFine)
    assert not isinstance(value('type: LOCALTIME\nvalue: "12:34:56.789000000"\n'), values.TooFine)


def test_a_boolean_is_not_an_integer_however_python_stores_it() -> None:
    """Python's ``True`` is ``1`` and its ``bool`` is a subclass of
    ``int``, which no other client in the corpus has to think about. A
    case wanting BOOL true and getting INT64 1 back is a failure, and it
    would pass without this."""
    assert not values.same(True, 1)
    assert not values.same(1, True)
    assert values.same(True, True)
    assert values.show(True) == "BOOL true"
    assert values.show(1) == 'INT64 "1"'


# What a case is.


def test_a_case_is_a_statement_and_the_rows_it_owes() -> None:
    case = one(
        "  - name: int64-max\n    doc: the largest INT64\n"
        "    query: RETURN 9223372036854775807 AS n\n" + rows_of("9223372036854775807")
    )
    assert case.name == "int64-max"
    assert case.doc == "the largest INT64"
    assert case.query == "RETURN 9223372036854775807 AS n"
    assert case.setup == []
    assert case.columns == ["n"]
    assert case.rows == [[2**63 - 1]]


def test_a_case_may_load_its_own_data_first() -> None:
    case = one(
        "  - name: with-setup\n    doc: a case that needs a graph\n    setup:\n"
        "      - CREATE NODE TABLE Person(name STRING)\n      - INSERT (:Person {name: 'a'})\n"
        "    query: MATCH (p:Person) RETURN p.name AS name\n    columns:\n      - name\n"
        "    rows:\n      - values:\n          - type: STRING\n            value: a\n"
    )
    assert len(case.setup) == 2
    assert case.setup[0].query.startswith("CREATE NODE TABLE")
    # A setup written as a bare statement runs on the connection the case
    # itself runs on, which is the one every case had before any of them
    # named a second.
    assert [step.on for step in case.setup] == [MAIN, MAIN]
    assert case.on == MAIN


def test_a_case_may_say_which_connection_it_runs_on() -> None:
    case = one(
        "  - name: on-another\n    doc: a case run from a second connection\n    setup:\n"
        "      - CREATE NODE TABLE Person(name STRING)\n"
        "      - on: writer\n        query: INSERT (:Person {name: 'a'})\n"
        "    on: reader\n    query: MATCH (p:Person) RETURN p.name AS name\n"
        "    columns:\n      - name\n    rows:\n      - values:\n"
        "          - type: STRING\n            value: a\n"
    )
    assert [(step.on, step.query.split(" ")[0]) for step in case.setup] == [
        (MAIN, "CREATE"),
        ("writer", "INSERT"),
    ]
    assert case.on == "reader"


def test_a_connection_is_named_the_way_everything_else_in_a_case_is() -> None:
    """Lower case words joined by dashes, so that a report citing one
    reads like the rest of a report and no client has to decide what to
    do with a name another client would have written differently."""
    case = one(
        "  - name: a\n    doc: d\n    on: read-only-one\n    query: RETURN 1\n    raises: 22012\n"
    )
    assert case.on == "read-only-one"


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("    on: Writer\n", "lower case words joined by dashes"),
        ("    on: a_b\n", "lower case words joined by dashes"),
        ("    on:\n", "is the name of a connection"),
        ("    on:\n      - a\n", "is the name of a connection"),
    ],
)
def test_a_connection_written_wrong_is_refused_where_it_is_written(text: str, want: str) -> None:
    with pytest.raises(reader.CorpusError) as raised:
        suite(f"  - name: a\n    doc: d\n{text}    query: RETURN 1\n    raises: 22012\n")
    assert want in str(raised.value)


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("      - query: RETURN 1\n", "names the connection it runs on"),
        ("      - on: writer\n", "no `query:`"),
        ("      - on: writer\n        stmt: RETURN 1\n", 'has no key "stmt"'),
        ("      - on: Writer\n        query: RETURN 1\n", "lower case words joined by dashes"),
    ],
)
def test_a_setup_step_written_as_a_mapping_says_both_halves(text: str, want: str) -> None:
    with pytest.raises(reader.CorpusError) as raised:
        suite(
            f"  - name: a\n    doc: d\n    setup:\n{text}    query: RETURN 1\n    raises: 22012\n"
        )
    assert want in str(raised.value)


def test_a_case_may_bind_parameters_and_they_keep_the_order_they_were_written_in() -> None:
    case = one(
        "  - name: bound\n    doc: a statement with two parameters in it\n    params:\n"
        '      - name: n\n        type: INT64\n        value: "42"\n'
        "      - name: s\n        type: STRING\n        value: ada\n"
        "    query: RETURN $n AS n, $s AS s\n    columns:\n      - n\n      - s\n"
        '    rows:\n      - values:\n          - type: INT64\n            value: "42"\n'
        "          - type: STRING\n            value: ada\n"
    )
    assert case.params == [("n", 42), ("s", "ada")]


def test_a_parameter_is_a_value_of_the_same_encoding_and_is_read_the_same_way() -> None:
    """A NULL parameter carries no `value`, the quoting rule is the one
    every other value follows, and a list is a list. The whole point of
    `params:` is that it is the row encoding with a name added, so what
    is checked here is that it did not become a second encoding."""
    case = one(
        "  - name: null-param\n    doc: a parameter that is nothing\n    params:\n"
        "      - name: n\n        type: NULL\n    query: RETURN $n AS n\n"
        "    columns:\n      - n\n    rows:\n      - values:\n          - type: NULL\n"
    )
    assert case.params == [("n", None)]
    case = one(
        "  - name: list-param\n    doc: a parameter holding a list of two\n    params:\n"
        "      - name: xs\n        type: LIST\n        value:\n"
        "          - type: INT8\n            value: 1\n"
        "          - type: INT8\n            value: 2\n"
        "    query: RETURN size($xs) AS n\n    columns:\n      - n\n"
        '    rows:\n      - values:\n          - type: INT64\n            value: "2"\n'
    )
    assert case.params == [("xs", [1, 2])]
    with pytest.raises(reader.CorpusError) as raised:
        suite(
            "  - name: a\n    doc: d\n    params:\n      - name: n\n        type: INT64\n"
            "        value: 42\n    query: RETURN $n\n    raises: 22012\n"
        )
    assert "written in quotes" in str(raised.value)


@pytest.mark.parametrize(
    ("text", "want"),
    [
        (
            "  - name: a\n    doc: d\n    params:\n      - type: INT8\n        value: 1\n"
            "    query: RETURN $n\n    raises: 22012\n",
            "no `name:`",
        ),
        (
            "  - name: a\n    doc: d\n    params:\n      - name: n one\n        type: INT8\n"
            "        value: 1\n    query: RETURN $n\n    raises: 22012\n",
            "is a parameter name",
        ),
        (
            "  - name: a\n    doc: d\n    params:\n      - name: n\n        type: INT8\n"
            "        value: 1\n      - name: n\n        type: INT8\n        value: 2\n"
            "    query: RETURN $n\n    raises: 22012\n",
            'two parameters are called "n"',
        ),
        (
            "  - name: a\n    doc: d\n    params:\n      - name: n\n        type: INT8\n"
            "        value: 1\n        note: hi\n    query: RETURN $n\n    raises: 22012\n",
            'a parameter has no key "note"',
        ),
        (
            "  - name: a\n    doc: d\n    params:\n      - $n\n    query: RETURN $n\n"
            "    raises: 22012\n",
            "a parameter is a mapping",
        ),
        (
            "  - name: a\n    doc: d\n    params: n\n    query: RETURN $n\n    raises: 22012\n",
            "`params:` is a sequence",
        ),
    ],
)
def test_a_parameter_a_statement_could_not_name_is_refused_where_it_is_written(
    text: str, want: str
) -> None:
    with pytest.raises(reader.CorpusError) as raised:
        suite(text)
    assert want in str(raised.value)


def test_a_suite_may_load_a_table_every_case_in_it_reads_back() -> None:
    read = cases.read(
        "schema: 4\nsuite: int\ndoc: d\nload:\n  nodes: person\n  edges: knows\n  count: 1\n"
        '  columns:\n    - name: age\n      type: INT64\n      values:\n        - "30"\n'
        "cases:\n  - name: a\n    doc: d\n    query: MATCH (p:person) RETURN p.age AS n\n"
        + rows_of("30")
    )
    assert read.load.nodes == "person"
    assert read.load.columns[0].values == [30]


def test_a_suite_of_expressions_has_no_load() -> None:
    assert suite("  - name: a\n    doc: d\n    query: RETURN 1\n    raises: 22012\n").load is None


def test_a_case_may_expect_a_condition_instead_of_rows() -> None:
    case = one(
        "  - name: divide-by-zero\n"
        "    doc: division by zero raises rather than returning inf\n"
        "    query: RETURN 1 / 0\n    raises: 22012\n"
    )
    assert case.raises == "22012"
    assert case.rows is None


def test_a_case_expecting_nothing_back_says_so_out_loud() -> None:
    case = one(
        "  - name: empty\n    doc: a filter nothing satisfies gives no rows\n"
        "    query: UNWIND [1] AS n WHERE false RETURN n\n    columns:\n      - n\n    rows:\n"
    )
    assert case.rows == []


def test_a_suite_carries_the_line_of_every_case_so_a_failure_can_be_opened() -> None:
    read = suite(
        "  - name: a\n    doc: the first\n    query: RETURN 1 AS n\n    columns:\n      - n\n"
        "    rows:\n      - values:\n          - type: INT8\n            value: 1\n"
        "  - name: b\n    doc: the second\n    query: RETURN 2 AS n\n    columns:\n      - n\n"
        "    rows:\n      - values:\n          - type: INT8\n            value: 2\n"
    )
    assert [c.line for c in read.cases] == [6, 15]


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("  - name: a\n", "no `doc:`"),
        ("  - doc: a\n    query: RETURN 1\n", "no `name:`"),
        ("  - name: A\n    doc: d\n    query: RETURN 1\n    raises: 22012\n", "is a case name"),
        ("  - name: a\n    doc: d\n    query: RETURN 1\n", "says what it produces"),
        (
            "  - name: a\n    doc: d\n    query: RETURN 1\n    raises: 22012\n"
            "    columns:\n      - n\n    rows:\n",
            "has no rows",
        ),
        (
            "  - name: a\n    doc: d\n    query: RETURN 1\n    raises: 999999\n",
            "not the shape of a GQLSTATUS",
        ),
        (
            "  - name: a\n    doc: d\n    query: RETURN 1\n    columns:\n      - n\n",
            "with no `rows:`",
        ),
        (
            "  - name: a\n    doc: d\n    query: RETURN 1 AS n, 2 AS m\n"
            "    columns:\n      - n\n      - m\n"
            "    rows:\n      - values:\n          - type: INT8\n            value: 1\n",
            "a row of 1 against 2 columns",
        ),
        (
            "  - name: a\n    doc: d\n    qeury: RETURN 1\n    raises: 22012\n",
            'no key "qeury"',
        ),
    ],
)
def test_a_file_the_runner_cannot_read_says_which_line_and_why(text: str, want: str) -> None:
    with pytest.raises(reader.CorpusError) as raised:
        suite(text)
    assert want in str(raised.value)


def test_the_same_case_name_twice_is_refused_because_a_report_cites_names() -> None:
    with pytest.raises(reader.CorpusError) as raised:
        suite(
            "  - name: a\n    doc: d\n    query: RETURN 1\n    raises: 22012\n"
            "  - name: a\n    doc: e\n    query: RETURN 2\n    raises: 22012\n"
        )
    assert 'two cases are called "a"' in str(raised.value)


def test_a_file_from_another_schema_says_so_rather_than_failing_in_the_middle() -> None:
    with pytest.raises(reader.CorpusError) as raised:
        cases.read("schema: 1\nsuite: int\ndoc: d\ncases:\n  - name: a\n")
    assert "schema 1 and the runner reads schema 4" in str(raised.value)


def test_a_suite_whose_name_is_not_its_file_name_is_refused(tmp_path: Path) -> None:
    """A report cites a suite by name and a reader opens it by file name,
    so the two disagreeing is a failure nobody can find."""
    (tmp_path / "float.yaml").write_text(
        f"{HEAD}\ncases:\n  - name: a\n    doc: d\n    query: RETURN 1\n    raises: 22012\n"
    )
    with pytest.raises(reader.CorpusError) as raised:
        cases.read_dir(tmp_path)
    assert 'calls itself "int" and the file calls it "float"' in str(raised.value)


def test_a_directory_with_no_case_files_is_refused_rather_than_passing_empty(
    tmp_path: Path,
) -> None:
    with pytest.raises(reader.CorpusError) as raised:
        cases.read_dir(tmp_path)
    assert "no case files" in str(raised.value)


# What a case says about the export.


def test_a_case_may_say_what_the_export_gives_field_by_field() -> None:
    case = one(
        "  - name: exported\n    doc: a statement whose export is the point\n"
        "    query: RETURN 1 AS n\n    columns:\n      - n\n"
        "    rows:\n      - values:\n          - type: INT8\n            value: 1\n"
        "    arrow:\n      - name: n\n        format: l\n"
    )
    assert case.arrow == [arrow.Field("n", "l")]


def test_a_nested_field_carries_the_fields_under_it() -> None:
    case = one(
        "  - name: exported\n    doc: a list of strings, which is a field inside a field\n"
        "    query: RETURN ['a'] AS xs\n    columns:\n      - xs\n"
        "    rows:\n      - values:\n          - type: LIST\n            value:\n"
        "              - type: STRING\n                value: a\n"
        "    arrow:\n      - name: xs\n        format: +l\n        children:\n"
        "          - name: item\n            format: u\n"
    )
    assert case.arrow == [arrow.Field("xs", "+l", (arrow.Field("item", "u"),))]


def test_a_result_arrow_has_no_type_for_is_written_as_a_refusal() -> None:
    case = one(
        "  - name: refused\n    doc: a time with an offset, which Arrow has no type for\n"
        "    query: RETURN 1 AS n\n    raises: 22012\n    arrow: refused\n"
    )
    assert case.arrow is arrow.REFUSED


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("    arrow: yes\n", "or `refused` for a result"),
        ("    arrow:\n      - n\n", "an Arrow field is a mapping"),
        ("    arrow:\n      - format: l\n", "an Arrow field has a `name:`"),
        ("    arrow:\n      - name: n\n", "an Arrow field has a `format:`"),
        ('    arrow:\n      - name: n\n        format: ""\n', "not a type Arrow has"),
        ("    arrow:\n      - name: n\n        format: l\n        kind: int\n", 'no key "kind"'),
        (
            "    arrow:\n      - name: n\n        format: +l\n",
            "is a nested type and the fields under it are part of it",
        ),
        (
            "    arrow:\n      - name: n\n        format: l\n        children:\n"
            "          - name: item\n            format: u\n",
            "holds no fields, so nothing goes under it",
        ),
    ],
)
def test_an_export_written_wrong_is_refused_where_it_is_written(text: str, want: str) -> None:
    with pytest.raises(reader.CorpusError) as raised:
        suite(f"  - name: a\n    doc: d\n    query: RETURN 1\n    raises: 22012\n{text}")
    assert want in str(raised.value)


def test_what_the_export_gave_that_the_case_did_not_want_is_said_the_way_rust_says_it() -> None:
    """One wording per difference, checked here rather than only through
    a run, because these are the lines the nine reports are diffed by."""
    n = arrow.Field("n", "l")
    assert arrow.schema([n], [n]) is None
    assert arrow.schema([n, n], [n]) == "arrow gives 2 fields in the result where the case wants 1"
    assert (
        arrow.schema([arrow.Field("m", "l")], [n])
        == 'arrow field 1 in the result is named "m" where the case wants "n"'
    )
    assert (
        arrow.schema([arrow.Field("n", "u")], [n])
        == 'arrow field "n" is "u" where the case wants "l"'
    )
    nested = arrow.Field("xs", "+l", (arrow.Field("item", "u"),))
    assert (
        arrow.schema([arrow.Field("xs", "+l", (arrow.Field("item", "l"),))], [nested])
        == 'arrow field "xs.item" is "l" where the case wants "u"'
    )
    assert (
        arrow.schema([arrow.Field("xs", "+l", ())], [nested])
        == 'arrow gives 0 fields in "xs" where the case wants 1'
    )


# Running.


def _write(tmp_path: Path, name: str, body: str) -> Path:
    directory = tmp_path / "cases"
    directory.mkdir(exist_ok=True)
    (directory / f"{name}.yaml").write_text(
        f"schema: 4\nsuite: {name}\ndoc: a suite written by a test\ncases:\n{body}"
    )
    return directory


def _run(tmp_path: Path, name: str, body: str) -> runner.Report:
    directory = _write(tmp_path, name, body)
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    return runner.run(cases.read_dir(directory), work)


def test_a_case_that_asks_for_what_the_engine_gives_passes(tmp_path: Path) -> None:
    report = _run(
        tmp_path,
        "small",
        "  - name: one\n    doc: the smallest statement there is\n    query: RETURN 1 AS n\n"
        + rows_of("1"),
    )
    assert report.summary() == "1 cases, 1 passed, 0 failed, 0 unsupported"


def test_a_case_that_asks_for_a_condition_passes_on_the_code_and_not_the_message(
    tmp_path: Path,
) -> None:
    report = _run(
        tmp_path,
        "small",
        "  - name: over-zero\n    doc: division by zero raises\n    query: RETURN 1 / 0 AS n\n"
        "    raises: 22012\n",
    )
    assert report.count(runner.PASSED) == 1


def test_a_wrong_row_is_reported_in_the_encoding_a_case_is_written_in(tmp_path: Path) -> None:
    report = _run(
        tmp_path,
        "small",
        "  - name: one\n    doc: a case that wants the wrong number\n    query: RETURN 1 AS n\n"
        + rows_of("2"),
    )
    assert (
        report.failures()[0].detail == 'row 1 column n is INT64 "1" where the case wants INT64 "2"'
    )


def test_a_wrong_column_list_is_reported_before_the_rows_are_looked_at(tmp_path: Path) -> None:
    report = _run(
        tmp_path,
        "small",
        "  - name: one\n    doc: a case that wants another name\n    query: RETURN 1 AS n\n"
        + rows_of("1", column="m"),
    )
    assert report.failures()[0].detail == 'columns ["n"] where the case wants ["m"]'


def test_a_row_count_that_differs_is_reported_after_the_rows_that_match(tmp_path: Path) -> None:
    report = _run(
        tmp_path,
        "small",
        "  - name: two\n    doc: a case that wants a row that is not there\n"
        "    query: UNWIND [1] AS n RETURN n\n    columns:\n      - n\n"
        '    rows:\n      - values:\n          - type: INT64\n            value: "1"\n'
        '      - values:\n          - type: INT64\n            value: "2"\n',
    )
    assert report.failures()[0].detail == "1 rows where the case wants 2"


def test_a_case_the_engine_cannot_parse_is_unsupported_rather_than_failed(tmp_path: Path) -> None:
    report = _run(
        tmp_path,
        "small",
        "  - name: ahead\n    doc: a statement this engine does not have yet\n"
        "    query: RETURN nonesuch(1) AS n\n    columns:\n      - n\n    rows:\n",
    )
    assert report.count(runner.UNSUPPORTED) == 1
    assert report.count(runner.FAILED) == 0


def test_a_case_that_wanted_a_condition_and_got_rows_is_a_failure(tmp_path: Path) -> None:
    report = _run(
        tmp_path,
        "small",
        "  - name: quiet\n    doc: a statement that was meant to raise\n    query: RETURN 1 AS n\n"
        "    raises: 22012\n",
    )
    assert report.failures()[0].detail == "returned rows where the case wants 22012"


def test_a_case_that_raised_the_wrong_code_says_both(tmp_path: Path) -> None:
    report = _run(
        tmp_path,
        "small",
        "  - name: wrong-code\n    doc: a case naming the wrong condition\n"
        "    query: RETURN 1 / 0 AS n\n    raises: 22003\n",
    )
    assert report.failures()[0].detail.startswith("raised 22012 where the case wants 22003")


def test_a_suite_load_reaches_every_case_and_no_case_reaches_another(tmp_path: Path) -> None:
    """Each case gets its own database with its own copy of the load, so
    what one case inserts is not there for the next one."""
    directory = tmp_path / "cases"
    directory.mkdir()
    (directory / "graph.yaml").write_text(
        "schema: 4\nsuite: graph\ndoc: a loaded suite\nload:\n  nodes: person\n  edges: knows\n"
        "  count: 2\n  columns:\n    - name: name\n      type: STRING\n      values:\n"
        "        - ada\n        - bob\n  pairs:\n    - from: 0\n      to: 1\n"
        "cases:\n"
        "  - name: reads-the-load\n    doc: the load is there\n"
        "    query: MATCH (p:person) RETURN count(p) AS n\n    columns:\n      - n\n"
        '    rows:\n      - values:\n          - type: INT64\n            value: "2"\n'
        "  - name: inserts-a-row\n    doc: a case that adds one\n"
        "    query: INSERT (p:person {name: 'cyd'}) RETURN p.name AS n\n    columns:\n      - n\n"
        "    rows:\n      - values:\n          - type: STRING\n            value: cyd\n"
        "  - name: does-not-see-it\n    doc: the row the case before inserted is not here\n"
        "    query: MATCH (p:person) RETURN count(p) AS n\n    columns:\n      - n\n"
        '    rows:\n      - values:\n          - type: INT64\n            value: "2"\n'
    )
    work = tmp_path / "work"
    work.mkdir()
    report = runner.run(cases.read_dir(directory), work)
    assert report.count(runner.FAILED) == 0, [str(r) for r in report.failures()]


def test_a_case_binds_its_parameters_through_this_client(tmp_path: Path) -> None:
    report = _run(
        tmp_path,
        "small",
        "  - name: bound\n    doc: a parameter crosses the boundary and comes back\n"
        '    params:\n      - name: n\n        type: INT64\n        value: "42"\n'
        "    query: RETURN $n AS n\n    columns:\n      - n\n"
        '    rows:\n      - values:\n          - type: INT64\n            value: "42"\n',
    )
    assert report.count(runner.PASSED) == 1


def test_a_second_connection_sees_what_the_first_one_committed(tmp_path: Path) -> None:
    """The whole point of naming a connection: a write that came back
    from one is a write the other reads. The two are duplicates of each
    other and share the write side, so this is the answer a pool gives
    and not the answer two opens of one file give."""
    report = _run(
        tmp_path,
        "two",
        "  - name: across\n    doc: a committed write is another connection's to read\n"
        '    setup:\n      - on: writer\n        query: "INSERT (:thing {n: 1})"\n'
        "    on: reader\n    query: MATCH (t:thing) RETURN count(t) AS n\n" + rows_of("1"),
    )
    assert report.summary() == "1 cases, 1 passed, 0 failed, 0 unsupported"


def test_a_case_naming_no_connection_runs_everything_on_the_one_it_opened(
    tmp_path: Path,
) -> None:
    report = _run(
        tmp_path,
        "one",
        "  - name: alone\n    doc: a case that says nothing about connections\n"
        '    setup:\n      - "INSERT (:thing {n: 1})"\n'
        "    query: MATCH (t:thing) RETURN count(t) AS n\n" + rows_of("1"),
    )
    assert report.count(runner.PASSED) == 1


def test_the_export_is_checked_against_the_same_result_the_rows_came_from(
    tmp_path: Path,
) -> None:
    report = _run(
        tmp_path,
        "small",
        "  - name: one\n    doc: an int64 column is an int64 on the way out\n"
        "    query: RETURN 1 AS n\n" + rows_of("1") + "    arrow:\n      - name: n\n"
        "        format: l\n",
    )
    assert report.summary() == "1 cases, 1 passed, 0 failed, 0 unsupported"


def test_an_export_that_is_not_what_the_case_wants_is_a_failure_and_says_which_field(
    tmp_path: Path,
) -> None:
    report = _run(
        tmp_path,
        "small",
        "  - name: one\n    doc: a case that wants the wrong Arrow type\n"
        "    query: RETURN 1 AS n\n" + rows_of("1") + "    arrow:\n      - name: n\n"
        "        format: u\n",
    )
    assert report.failures()[0].detail == 'arrow field "n" is "l" where the case wants "u"'


def test_a_result_arrow_will_not_take_is_a_refusal_and_not_a_crash(tmp_path: Path) -> None:
    """A time with an offset is the one a statement can write today.
    Arrow has a time and a timestamp and nothing in between, so the
    export says no, and a case saying `refused` is a case that passes on
    it saying no."""
    body = (
        "  - name: offset\n    doc: a time with an offset has no Arrow type\n"
        "    query: RETURN ZONED TIME '12:34:56+07:00' AS n\n    columns:\n      - n\n"
        "    rows:\n      - values:\n          - type: ZONEDTIME\n"
        '            value: "12:34:56+07:00"\n'
    )
    assert _run(tmp_path, "zoned", body + "    arrow: refused\n").count(runner.PASSED) == 1
    wanted = _run(tmp_path, "zoned", body + "    arrow:\n      - name: n\n        format: l\n")
    assert wanted.failures()[0].detail.startswith("arrow refused the result: ")


def test_an_export_giving_a_different_number_of_rows_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rows are counted off the stream rather than taken from the
    rows already read, because an export that gave the right schema and
    half the rows would pass every other check in here. The engine does
    not disagree with itself about how many rows a result has, so the
    only way to see the report is to make it disagree."""
    field = arrow.Field("n", "l")
    monkeypatch.setattr(arrow, "exported", lambda result: ([field], 2))
    assert runner._exported([field], object(), 2) is None
    assert runner._exported([field], object(), 3) == "arrow gives 2 rows where the case wants 3"


def test_the_command_line_exits_zero_on_a_run_that_passes(tmp_path: Path) -> None:
    directory = _write(
        tmp_path,
        "small",
        "  - name: one\n    doc: the smallest statement there is\n    query: RETURN 1 AS n\n"
        + rows_of("1"),
    )
    assert runner.main([str(directory), "--quiet"]) == 0


def test_the_command_line_exits_one_on_a_corpus_it_cannot_read(tmp_path: Path) -> None:
    """One and not two, because the reference runner exits one and a
    report compared line for line is worth less if the two runners
    disagree about what the run came to."""
    directory = tmp_path / "cases"
    directory.mkdir()
    (directory / "small.yaml").write_text("schema: 4\nsuite: small\ndoc: d\ncases:\n  - name: a\n")
    assert runner.main([str(directory), "--quiet"]) == 1


def test_strict_turns_an_unsupported_case_into_a_failed_run(tmp_path: Path) -> None:
    directory = _write(
        tmp_path,
        "small",
        "  - name: ahead\n    doc: a statement this engine does not have yet\n"
        "    query: RETURN nonesuch(1) AS n\n    columns:\n      - n\n    rows:\n",
    )
    assert runner.main([str(directory), "--quiet"]) == 0
    assert runner.main([str(directory), "--quiet", "--strict"]) == 1


def test_the_module_runs_as_a_command(tmp_path: Path) -> None:
    directory = _write(
        tmp_path,
        "small",
        "  - name: one\n    doc: the smallest statement there is\n    query: RETURN 1 AS n\n"
        + rows_of("1"),
    )
    done = subprocess.run(
        [sys.executable, "-m", "conformance", str(directory)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "1 cases, 1 passed, 0 failed, 0 unsupported"


#: The cases this client cannot answer, and the only ones it may leave
#: unanswered. Written out rather than counted, so that a seventh one
#: arriving is a failure here and not a number nobody looks at. All six
#: are a time to the nanosecond, which is a digit finer than Python's
#: `datetime` holds.
UNHELD = {
    "arrow/a-local-time-is-nanoseconds-from-midnight",
    "param/localtime-to-the-nanosecond",
    "stored/a-localtime-column-keeps-every-digit",
    "stored/the-columns-of-one-row-belong-to-that-row",
    "temporal/local-time-nanoseconds",
    "temporal/a-time-carries-a-single-nanosecond",
}


@pytest.fixture(scope="session")
def corpus() -> Iterator[runner.Report]:
    """The whole corpus, run once for everything below that reads it.

    Every case gets a database of its own and there are over a thousand
    of them, so a run is a couple of gigabytes that exist for as long as
    it takes. Once per session rather than once per test, and under a
    directory that goes with the run rather than one pytest keeps three
    generations of: two tests reading one report is not a reason to fill
    a disk twice over, and it is a disk this has already filled."""
    with tempfile.TemporaryDirectory(prefix="zudb-corpus-") as work:
        yield runner.run(cases.read_dir(Path(CASES)), Path(work))


@needs_cases
def test_every_case_the_engine_ships_passes_through_this_client(corpus: runner.Report) -> None:
    """The corpus itself, which is the check the other two runners run
    and the reason this one exists."""
    assert corpus.count(runner.FAILED) == 0, [str(r) for r in corpus.failures()][:10]
    unheld = {f"{r.suite}/{r.case}" for r in corpus.ran if r.outcome == runner.UNSUPPORTED}
    assert unheld == UNHELD
    assert corpus.count(runner.PASSED) == len(corpus.ran) - len(UNHELD)


@needs_cases
def test_the_cases_this_client_cannot_hold_are_a_precision_and_not_a_wrong_answer(
    corpus: runner.Report,
) -> None:
    """Every one of them says which value it is and why, because a case
    reported as unsupported with no reason is a case nobody revisits."""
    for ran in corpus.ran:
        if ran.outcome == runner.UNSUPPORTED:
            assert "finer" in ran.detail, str(ran)
