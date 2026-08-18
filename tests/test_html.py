"""What a notebook shows.

A result in a cell is drawn as a table because Jupyter asks an object
for `_repr_html_` before it falls back to `repr`. What is checked here
is the markup itself, since nothing else can: a notebook renders it and
a person looks at it, and neither of those is a test.

The two that matter are the escaping and the cut. A string column
holding a tag is a string column, and a client that pasted one into the
page would run a caller's data as code in their notebook. A result of a
million rows is a result somebody typed by accident, and a million rows
of markup is a notebook file that will not open again.
"""

from __future__ import annotations

import re
from pathlib import Path

import zudb


def cells(html: str) -> list[str]:
    """The text of every body cell, tags taken out."""
    body = html.split("<tbody>")[1].split("</tbody>")[0]
    return [re.sub(r"<[^>]+>", "", cell) for cell in re.findall(r"<td[^>]*>(.*?)</td>", body)]


def headers(html: str) -> list[str]:
    return re.findall(r"<th>(.*?)</th>", html)


def note(html: str) -> str:
    return re.findall(r'<div class="zu-note">(.*?)</div>', html)[0]


def test_a_result_is_a_table_of_its_rows(social: zudb.Connection) -> None:
    html = social.execute("MATCH (p:person) RETURN p.uid AS uid, p.name AS name")._repr_html_()
    assert headers(html) == ["uid", "name"]
    assert cells(html) == ["10", "ada", "20", "grace", "30", "kay"]
    assert note(html) == "3 rows, 2 columns"


def test_the_stylesheet_travels_with_the_table(social: zudb.Connection) -> None:
    """No install, no script, and nothing to load from anywhere."""
    html = social.execute("MATCH (p:person) RETURN p.uid AS uid")._repr_html_()
    assert html.startswith("<style>")
    assert "<script" not in html
    assert "http" not in html


def test_a_number_lines_up_on_the_right_and_a_string_does_not(
    social: zudb.Connection,
) -> None:
    html = social.execute(
        "MATCH (p:person) RETURN p.uid AS uid, p.name AS name, p.score AS score"
    )._repr_html_()
    row = re.findall(r"<tr>(.*?)</tr>", html.split("<tbody>")[1])[0]
    assert row.count('class="zu-num"') == 2, "the integer and the float"


def test_a_boolean_is_not_a_number_to_line_up(empty: zudb.Connection) -> None:
    empty.execute("INSERT (p:person {uid: 1, ok: true})")
    html = empty.execute("MATCH (p:person) RETURN p.ok AS ok")._repr_html_()
    assert cells(html) == ["True"]
    assert 'class="zu-num"' not in html


def test_a_null_says_so_rather_than_being_blank(empty: zudb.Connection) -> None:
    empty.execute("INSERT (p:person {uid: 1, name: 'ada'})")
    html = empty.execute("MATCH (p:person) RETURN null AS nothing")._repr_html_()
    assert cells(html) == ["null"]
    assert 'class="zu-null"' in html


def test_a_tag_in_a_value_is_text_and_not_markup(empty: zudb.Connection) -> None:
    empty.execute("INSERT (p:person {uid: 1, name: '<script>alert(1)</script>'})")
    html = empty.execute("MATCH (p:person) RETURN p.name AS name")._repr_html_()
    assert "<script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_a_tag_in_a_column_name_is_text_too(empty: zudb.Connection) -> None:
    empty.execute("INSERT (p:person {uid: 1})")
    html = empty.execute("MATCH (p:person) RETURN p.uid AS `<b>uid</b>`")._repr_html_()
    assert "<b>" not in html
    assert "&lt;b&gt;uid&lt;/b&gt;" in headers(html)


def test_a_long_value_is_cut_and_marked(social: zudb.Connection) -> None:
    social.execute("INSERT (p:person {uid: 40, name: $name, score: 1.0})", {"name": "a" * 500})
    html = social.execute("MATCH (p:person) WHERE p.uid = 40 RETURN p.name AS name")._repr_html_()
    shown = cells(html)[0]
    assert shown == "a" * 200 + "…"


def test_a_long_result_is_cut_and_says_how_long_it_was(tmp_path: Path) -> None:
    path = tmp_path / "many.zu1"
    zudb.load(path, nodes="person", columns={"uid": list(range(5_000))})
    with zudb.connect(path, read_only=True) as conn:
        html = conn.execute("MATCH (p:person) RETURN p.uid AS uid")._repr_html_()
    assert len(cells(html)) == 100
    assert note(html) == "5,000 rows, 1 column, first 100 shown"


def test_drawing_a_result_does_not_take_any_of_its_rows(social: zudb.Connection) -> None:
    """A person who looked at a result has not read it."""
    result = social.execute("MATCH (p:person) RETURN p.name AS name")
    result._repr_html_()
    assert result.fetchone() == ("ada",)
    result._repr_html_()
    assert result.fetchone() == ("grace",)


def test_a_result_with_no_rows_is_a_table_with_none(social: zudb.Connection) -> None:
    html = social.execute("MATCH (p:person) WHERE p.uid > 99 RETURN p.name AS name")._repr_html_()
    assert headers(html) == ["name"]
    assert cells(html) == []
    assert note(html) == "0 rows, 1 column"


def test_a_statement_that_writes_says_it_has_no_columns(empty: zudb.Connection) -> None:
    html = empty.execute("INSERT (p:person {uid: 1})")._repr_html_()
    assert "no columns" in html
    assert "<table" not in html


def test_a_node_is_drawn_as_the_pair_that_names_it(loaded: zudb.Connection) -> None:
    (node,) = loaded.execute("MATCH (p:person) RETURN p AS p").fetchone()
    html = node._repr_html_()
    assert "person" in html and "#0" in html


def test_an_edge_says_which_rows_it_joins(loaded: zudb.Connection) -> None:
    (rel,) = loaded.execute("MATCH ()-[r:knows]->() RETURN r AS r").fetchone()
    html = rel._repr_html_()
    assert "#0" in html and "#1" in html and "-[knows]-&gt;" in html


def test_a_path_is_drawn_as_a_walk(loaded: zudb.Connection) -> None:
    (path,) = loaded.execute(
        "MATCH q = (a:person)-[:knows]->(b:person)-[:knows]->(c:person) RETURN q AS q"
    ).fetchone()
    html = path._repr_html_()
    assert html.count("zu-node") == 3
    assert html.count("zu-rel") == 2
    assert "#0" in html and "#1" in html and "#2" in html


def test_a_graph_value_in_a_cell_is_drawn_like_it_is_on_its_own(
    loaded: zudb.Connection,
) -> None:
    html = loaded.execute("MATCH (p:person) RETURN p AS p")._repr_html_()
    assert html.count("zu-node") == 3
    assert "&lt;" not in html.split("<tbody>")[1].split("</tbody>")[0], "drawn, not escaped repr"
