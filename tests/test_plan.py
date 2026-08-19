"""What a statement would do, and what it did.

A plan belongs to the engine and this client only carries it, so what
these assert is that the carrying is faithful: every operator, in the
shape the tree had, with the fields that mean something and `None`
where the engine had nothing to say. The listing is asserted against the
tree it was rendered from rather than against a string written out here,
because a test that pinned the exact words would fail every time the
optimizer learned to print one better.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import zudb

BY_NAME = "MATCH (p:person) WHERE p.name = $name RETURN p.uid AS uid"


def operators(node: zudb.PlanNode | None) -> list[zudb.PlanNode]:
    """Every operator of a plan, depth first, which is the order the
    listing prints them in.
    """
    if node is None:
        return []
    return [node, *[found for child in node.children for found in operators(child)]]


def test_a_plan_is_the_tree_of_operators_the_statement_would_run(
    social: zudb.Connection,
) -> None:
    plan = social.explain(BY_NAME)

    assert isinstance(plan, zudb.Plan)
    assert [op.op for op in operators(plan.root)] == ["Project", "Filter", "ScanNodes"]
    assert plan.columns == ["uid"]
    assert plan.params == ["name"]
    assert plan.notes == []
    assert plan.scalars == []


def test_an_operator_carries_what_it_works_on_binds_and_touches(social: zudb.Connection) -> None:
    project, filtered, scan = operators(social.explain(BY_NAME).root)

    assert project.detail == "p.uid AS uid"
    assert project.binds == ["uid"]
    assert project.tables == []

    assert filtered.detail == "p.name = $name"
    assert filtered.binds == []

    assert scan.detail == "p: person"
    assert scan.binds == ["p"]
    assert scan.tables == ["person"]
    assert scan.children == []


def test_an_operator_inside_a_bracket_is_named_for_it_and_is_not_it(
    loaded: zudb.Connection,
) -> None:
    plan = loaded.explain(
        "MATCH (a:person) OPTIONAL MATCH (a)-[:knows]->(b:person) RETURN a.name AS a, b.name AS b"
    )
    expand = next(op for op in operators(plan.root) if op.op == "Expand")

    assert expand.name == "OptionalExpand"
    assert expand.bracket == "Optional"
    assert expand.tables == ["knows"]


def test_an_operator_outside_a_bracket_has_none_and_is_named_for_itself(
    loaded: zudb.Connection,
) -> None:
    plan = loaded.explain("MATCH (a:person)-[:knows]->(b:person) RETURN a.name AS a")
    expand = next(op for op in operators(plan.root) if op.op == "Expand")

    assert expand.name == "Expand"
    assert expand.bracket is None


def test_printing_a_plan_gives_the_listing_and_the_listing_is_the_tree(
    social: zudb.Connection,
) -> None:
    plan = social.explain(BY_NAME)

    assert str(plan) == plan.text
    assert plan.text == "Project p.uid AS uid\n  Filter p.name = $name\n    ScanNodes p: person\n"
    # Written twice on purpose: the listing is what a person reads and
    # the tree is what a program walks, and this is the one assertion
    # that says the two describe the same plan.
    printed = [line.strip().split(" ")[0] for line in plan.text.rstrip("\n").split("\n")]
    assert printed == [op.name for op in operators(plan.root)]


def test_a_query_written_where_a_value_belongs_is_a_plan_of_its_own(
    social: zudb.Connection,
) -> None:
    plan = social.explain(
        "MATCH (p:person) RETURN VALUE "
        "{ MATCH (q:person) WHERE q.name = p.name RETURN q.uid LIMIT 1 } AS v"
    )

    assert len(plan.scalars) == 1
    scalar = plan.scalars[0]
    # It reads a name from the query around it, which is the whole test
    # for whether it runs once or once a row.
    assert scalar.reads == ["p"]
    assert scalar.exists is False
    assert scalar.plan.root.op == "Limit"
    assert "ScanNodes q: person" in scalar.plan.text


def test_a_subquery_that_reads_nothing_runs_once_and_says_so(social: zudb.Connection) -> None:
    plan = social.explain(
        "MATCH (p:person) RETURN VALUE { MATCH (q:person) RETURN q.uid LIMIT 1 } AS v"
    )

    assert plan.scalars[0].reads == []
    assert "(once)" in plan.text


def test_explaining_does_not_run_the_statement(social: zudb.Connection) -> None:
    social.explain("INSERT (p:person {uid: 40, name: 'hedy', score: 1.0})")

    assert social.execute("MATCH (p:person) RETURN count(*) AS n").fetchall() == [(3,)]


def test_a_statement_that_does_not_compile_fails_at_the_explain(social: zudb.Connection) -> None:
    with pytest.raises(zudb.SyntaxError):
        social.explain("MATCH (")


def test_explaining_on_a_closed_connection_is_refused(tmp_path: Path) -> None:
    conn = zudb.connect(tmp_path / "shut.zu1")
    conn.close()

    with pytest.raises(zudb.ProgrammingError, match="closed"):
        conn.explain("MATCH (p:person) RETURN p.name AS name")


def test_a_plan_says_what_it_is(social: zudb.Connection) -> None:
    plan = social.explain(BY_NAME)

    assert repr(plan) == "<zudb.Plan Project p.uid AS uid>"
    assert repr(plan.root) == "<zudb.PlanNode Project p.uid AS uid>"


def test_a_notebook_gets_the_listing_preformatted(social: zudb.Connection) -> None:
    plan = social.explain(BY_NAME)
    markup = plan._repr_html_()

    assert "<pre" in markup
    assert "Project p.uid AS uid" in markup
    # The listing is escaped like every other piece of text this client
    # puts in a page, so a property named `<script>` is not markup.
    assert "$name" in markup


def test_a_profile_is_what_the_operators_really_did(social: zudb.Connection) -> None:
    run = social.profile("MATCH (p:person) RETURN p.name AS name")

    assert isinstance(run, zudb.Profile)
    assert len(run.stages) == 1
    stage = run.stages[0]
    assert stage.sink == "Project"
    assert stage.rows == 3
    assert stage.nanos > 0
    assert [op.op for op in stage.ops] == ["Source", "Scan"]

    scan = next(op for op in stage.ops if op.op == "Scan")
    assert scan.detail == "p: person"
    assert scan.pulls == 1
    assert scan.rows == 3
    assert scan.flat == 3
    assert scan.estimate == 3
    # The optimizer was right about a table it has the statistics for,
    # which is what a q-error of one means.
    assert scan.qerror == 1
    assert scan.nanos > 0


def test_an_operator_the_optimizer_had_nothing_to_say_about_carries_none(
    social: zudb.Connection,
) -> None:
    run = social.profile("MATCH (p:person) RETURN p.name AS name")
    source = next(op for op in run.stages[0].ops if op.op == "Source")

    assert source.estimate is None
    assert source.bound is None
    assert source.qerror is None


def test_the_profile_totals_its_stages_and_prints_them(social: zudb.Connection) -> None:
    run = social.profile("MATCH (p:person) RETURN p.name AS name")

    assert run.nanos == sum(stage.nanos for stage in run.stages)
    assert str(run) == run.text
    assert run.text.startswith("stage 1: Project")
    assert "Scan p: person" in run.text


def test_the_operators_are_in_the_order_they_ran(social: zudb.Connection) -> None:
    run = social.profile("MATCH (p:person) WHERE p.score > 40.0 RETURN p.name AS name")
    stage = run.stages[0]

    assert [op.op for op in stage.ops] == ["Source", "Scan", "Filter"]
    # The listing reads the other way, top down from the sink, and this
    # is the assertion that says the reversal is the only difference.
    lines = run.text.rstrip("\n").split("\n")
    printed = [line.strip().split(" ")[0] for line in lines if line.startswith("  ")]
    assert printed == [op.op for op in reversed(stage.ops)]


def test_the_counts_are_whole_numbers_python_can_hold(social: zudb.Connection) -> None:
    run = social.profile("MATCH (p:person) RETURN p.name AS name")

    for stage in run.stages:
        assert isinstance(stage.rows, int)
        assert isinstance(stage.nanos, int)
        for op in stage.ops:
            assert isinstance(op.pulls, int)
            assert isinstance(op.rows, int)
            assert isinstance(op.flat, int)
            assert isinstance(op.nanos, int)


def test_a_profile_binds_its_parameters(social: zudb.Connection) -> None:
    run = social.profile(BY_NAME, {"name": "ada"})
    filtered = next(op for op in run.stages[0].ops if op.op == "Filter")

    assert filtered.detail == "p.name = $name"
    assert run.stages[0].rows == 1


def test_a_profile_that_binds_nothing_fails_the_way_the_statement_would(
    social: zudb.Connection,
) -> None:
    with pytest.raises(zudb.SyntaxError, match=r"\$name"):
        social.profile(BY_NAME)


def test_an_expand_is_its_own_operator_with_the_rows_it_walked(loaded: zudb.Connection) -> None:
    run = loaded.profile("MATCH (a:person)-[:knows]->(b:person) RETURN b.name AS name")
    expand = next(op for op in run.stages[0].ops if op.op == "Expand")

    assert "knows" in expand.detail
    assert run.stages[0].rows == 2


def test_a_statement_that_writes_is_refused_rather_than_profiled(social: zudb.Connection) -> None:
    with pytest.raises(zudb.Error, match="profiling a statement that writes"):
        social.profile("INSERT (p:person {uid: 40, name: 'hedy', score: 1.0})")

    assert social.execute("MATCH (p:person) RETURN count(*) AS n").fetchall() == [(3,)]


def test_profiling_on_a_closed_connection_is_refused(tmp_path: Path) -> None:
    conn = zudb.connect(tmp_path / "shut.zu1")
    conn.close()

    with pytest.raises(zudb.ProgrammingError, match="closed"):
        conn.profile("MATCH (p:person) RETURN p.name AS name")


def test_a_profile_says_how_long_it_took_in_words(social: zudb.Connection) -> None:
    run = social.profile("MATCH (p:person) RETURN p.name AS name")

    # A repr that said `0 ms` for a statement measured in microseconds
    # would read as though nothing had been measured at all.
    assert repr(run).startswith("<zudb.Profile 1 stages, ")
    assert " ms>" in repr(run) or " us>" in repr(run) or " ns>" in repr(run)
    assert repr(run.stages[0]) == "<zudb.ProfileStage Project 3 rows>"
    assert repr(run.stages[0].ops[0]).startswith("<zudb.ProfileOp Source ")
