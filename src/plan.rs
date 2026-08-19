//! What a statement would do, and what it did.
//!
//! ```python
//! plan = conn.explain("MATCH (p:person) WHERE p.name = $name RETURN p.id AS id")
//! print(plan)
//! # Project p.id AS id
//! #   Filter p.name = $name
//! #     ScanNodes p: person
//! ```
//!
//! Both calls answer twice over, and that is on purpose. `print(plan)`
//! gives the engine's own listing, which is what a person reads and is
//! rendered by the engine rather than here so the two cannot drift
//! apart from one release to the next. `plan.root` gives the same plan
//! as objects, which is what a program walks: a test that wants to know
//! a scan became a seek asks the tree rather than matching on a string
//! that was written to be read.
//!
//! `explain` takes no parameters, which is not an oversight. A plan is
//! chosen from the shape of the statement and the values are bound when
//! it runs, so a plan asked for with values would suggest the values
//! had changed it. `profile` does take them, because it is a run.
//!
//! Every count here is an `int`, which in Python is exact however large
//! it gets, so nothing is lost and nothing has to be converted before
//! it is added up.

use pyo3::prelude::*;
use zudb::{OpProfile, Profile as EngineProfile, QueryPlan, StageProfile};

use crate::html;

/// One operator of a plan.
///
/// `op` is what the operator is and `name` is what the listing calls
/// it, and they differ where an operator sits inside a bracket: an
/// expand inside an `OPTIONAL MATCH` has `op` `Expand` and `name`
/// `OptionalExpand`. Matching on `op` is what a program should do,
/// since it is the one of the two that does not change with the
/// company an operator keeps.
#[pyclass(module = "zudb", frozen)]
pub struct PlanNode {
    /// The kind of operator this is.
    #[pyo3(get)]
    op: String,
    /// What the listing calls it, bracket and all.
    #[pyo3(get)]
    name: String,
    /// The bracket it sits inside, if it sits inside one: `"Optional"`,
    /// `"Semi"`, `"Anti"` or `"Mark"`.
    #[pyo3(get)]
    bracket: Option<String>,
    /// What it works on, in the words the listing prints.
    #[pyo3(get)]
    detail: String,
    /// The variables it introduces.
    #[pyo3(get)]
    binds: Vec<String>,
    /// The tables it reads.
    #[pyo3(get)]
    tables: Vec<String>,
    /// What it pulls from, in the order the listing prints them.
    #[pyo3(get)]
    children: Vec<Py<PlanNode>>,
}

#[pymethods]
impl PlanNode {
    fn __repr__(&self) -> String {
        format!("<zudb.PlanNode {} {}>", self.name, self.detail)
    }
}

/// A query written where a value belongs, and the plan it gets.
///
/// `reads` is which variables of the query around it this one reads,
/// and it is the whole of the difference between a subquery that runs
/// once and one that runs once a row: a plan that reads nothing is a
/// plan the executor can run once and keep.
#[pyclass(module = "zudb", frozen)]
pub struct ScalarPlan {
    /// The variables it reads from the query it is written inside.
    #[pyo3(get)]
    reads: Vec<String>,
    /// Whether it is asking whether there is a row rather than for one.
    #[pyo3(get)]
    exists: bool,
    /// The plan itself.
    #[pyo3(get)]
    plan: Py<Plan>,
}

#[pymethods]
impl ScalarPlan {
    fn __repr__(&self) -> String {
        let reads = if self.reads.is_empty() {
            "once".to_string()
        } else {
            format!("reads {}", self.reads.join(", "))
        };
        format!("<zudb.ScalarPlan {reads}>")
    }
}

/// What a statement would do, without doing it.
///
/// Take one with `Connection.explain`. `print` it for the listing and
/// walk `root` for the tree, which are the same plan said twice.
#[pyclass(module = "zudb", frozen)]
pub struct Plan {
    /// The operator everything else feeds, or `None` for a statement
    /// that compiled to no operator at all.
    #[pyo3(get)]
    root: Option<Py<PlanNode>>,
    /// The columns the statement projects, in order.
    #[pyo3(get)]
    columns: Vec<String>,
    /// The parameters it wants bound.
    #[pyo3(get)]
    params: Vec<String>,
    /// What the planner has to say about it, if anything.
    #[pyo3(get)]
    notes: Vec<String>,
    /// The plans of the queries written where values belong.
    #[pyo3(get)]
    scalars: Vec<Py<ScalarPlan>>,
    /// The engine's own listing, which is what `print` gives.
    #[pyo3(get)]
    text: String,
}

#[pymethods]
impl Plan {
    fn __str__(&self) -> String {
        self.text.clone()
    }

    fn __repr__(&self) -> String {
        format!("<zudb.Plan {}>", self.text.lines().next().unwrap_or(""))
    }

    /// The listing as a notebook shows it, which is the listing.
    ///
    /// Preformatted rather than a table, because the indentation is
    /// what says which operator pulls from which and a browser that
    /// collapsed the spaces would take the plan's shape away.
    fn _repr_html_(&self) -> String {
        html::wrap(&format!(
            "<pre style=\"margin:0\">{}</pre>",
            html::escape(&self.text)
        ))
    }
}

/// One operator of a statement that ran, and what it really did.
///
/// `estimate` is what the optimizer thought the operator would produce
/// and `rows` is what it did, so `qerror` is the larger over the
/// smaller: one where the estimate was right, ten where it was out by
/// an order of magnitude either way. All three are `None` on an
/// operator the optimizer had nothing to say about.
#[pyclass(module = "zudb", frozen)]
pub struct ProfileOp {
    /// The kind of operator this is.
    #[pyo3(get)]
    op: String,
    /// What it worked on, in the words the listing prints.
    #[pyo3(get)]
    detail: String,
    /// How many times the operator above it asked for rows.
    #[pyo3(get)]
    pulls: u64,
    /// How many rows it answered with.
    #[pyo3(get)]
    rows: u64,
    /// The same count with the vectors unpacked.
    #[pyo3(get)]
    flat: u64,
    /// What the optimizer thought it would answer.
    #[pyo3(get)]
    estimate: Option<f64>,
    /// The upper bound the optimizer had for it.
    #[pyo3(get)]
    bound: Option<f64>,
    /// How long it spent, in nanoseconds.
    #[pyo3(get)]
    nanos: u64,
    /// The estimate over the truth, or the truth over the estimate,
    /// whichever is the larger.
    #[pyo3(get)]
    qerror: Option<f64>,
}

#[pymethods]
impl ProfileOp {
    fn __repr__(&self) -> String {
        format!("<zudb.ProfileOp {} {} rows>", self.op, self.rows)
    }
}

/// One stage of a statement that ran.
///
/// A stage is the run of operators between two points the executor has
/// to gather at, and `sink` is what it gathers into.
#[pyclass(module = "zudb", frozen)]
pub struct ProfileStage {
    /// What the stage feeds.
    #[pyo3(get)]
    sink: String,
    /// How many rows came out of it.
    #[pyo3(get)]
    rows: u64,
    /// How long it took, in nanoseconds.
    #[pyo3(get)]
    nanos: u64,
    /// Its operators, from the one that read to the one that fed the
    /// sink, which is the order they ran in and the reverse of the
    /// order the listing prints them.
    #[pyo3(get)]
    ops: Vec<Py<ProfileOp>>,
}

#[pymethods]
impl ProfileStage {
    fn __repr__(&self) -> String {
        format!("<zudb.ProfileStage {} {} rows>", self.sink, self.rows)
    }
}

/// What a statement did, measured while it did it.
///
/// Take one with `Connection.profile`. It runs the statement, so a
/// statement that writes is refused rather than profiled: a measurement
/// that also inserted two rows changed the thing it was measuring.
#[pyclass(module = "zudb", frozen)]
pub struct Profile {
    /// The stages, in the order they ran.
    #[pyo3(get)]
    stages: Vec<Py<ProfileStage>>,
    /// Every stage added up, in nanoseconds.
    #[pyo3(get)]
    nanos: u64,
    /// The engine's own listing, which is what `print` gives.
    #[pyo3(get)]
    text: String,
}

#[pymethods]
impl Profile {
    fn __str__(&self) -> String {
        self.text.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "<zudb.Profile {} stages, {}>",
            self.stages.len(),
            spelled(self.nanos)
        )
    }

    /// The listing as a notebook shows it, preformatted for the reason
    /// a plan's is.
    fn _repr_html_(&self) -> String {
        html::wrap(&format!(
            "<pre style=\"margin:0\">{}</pre>",
            html::escape(&self.text)
        ))
    }
}

/// A count of nanoseconds in the unit a person would have said it in.
///
/// A `repr` reporting `0 ms` for a statement that took a quarter of a
/// millisecond is a `repr` that reads as though nothing was measured.
fn spelled(nanos: u64) -> String {
    if nanos < 1_000 {
        format!("{nanos} ns")
    } else if nanos < 1_000_000 {
        format!("{:.1} us", nanos as f64 / 1e3)
    } else if nanos < 1_000_000_000 {
        format!("{:.1} ms", nanos as f64 / 1e6)
    } else {
        format!("{:.2} s", nanos as f64 / 1e9)
    }
}

/// A whole plan as the objects a caller reads.
pub fn planned(py: Python<'_>, plan: &QueryPlan) -> PyResult<Plan> {
    let root = match &plan.root {
        Some(node) => Some(Py::new(py, operator(py, node)?)?),
        None => None,
    };
    let mut scalars = Vec::with_capacity(plan.scalars.len());
    for scalar in &plan.scalars {
        scalars.push(Py::new(
            py,
            ScalarPlan {
                reads: scalar.reads.clone(),
                exists: scalar.exists,
                plan: Py::new(py, planned(py, &scalar.plan)?)?,
            },
        )?);
    }
    Ok(Plan {
        root,
        columns: plan.columns.clone(),
        params: plan.params.clone(),
        notes: plan.notes.clone(),
        scalars,
        text: plan.render(),
    })
}

/// One operator and everything it pulls from.
fn operator(py: Python<'_>, node: &zudb::PlanNode) -> PyResult<PlanNode> {
    let mut children = Vec::with_capacity(node.children.len());
    for child in &node.children {
        children.push(Py::new(py, operator(py, child)?)?);
    }
    Ok(PlanNode {
        op: node.op.to_string(),
        name: node.name(),
        bracket: node
            .bracket
            .as_ref()
            .map(|bracket| bracket.prefix().to_string()),
        detail: node.detail.clone(),
        binds: node.binds.clone(),
        tables: node.tables.clone(),
        children,
    })
}

/// A whole profile as the objects a caller reads.
pub fn profiled(py: Python<'_>, profile: &EngineProfile) -> PyResult<Profile> {
    let mut stages = Vec::with_capacity(profile.stages.len());
    for stage in &profile.stages {
        stages.push(Py::new(py, staged(py, stage)?)?);
    }
    Ok(Profile {
        stages,
        nanos: profile.stages.iter().map(|stage| stage.nanos).sum(),
        text: profile.render(),
    })
}

fn staged(py: Python<'_>, stage: &StageProfile) -> PyResult<ProfileStage> {
    let mut ops = Vec::with_capacity(stage.ops.len());
    for op in &stage.ops {
        ops.push(Py::new(py, counted(op))?);
    }
    Ok(ProfileStage {
        sink: stage.sink.clone(),
        rows: stage.out_rows,
        nanos: stage.nanos,
        ops,
    })
}

fn counted(op: &OpProfile) -> ProfileOp {
    ProfileOp {
        op: op.kind.to_string(),
        detail: op.detail.clone(),
        pulls: op.pulls,
        rows: op.rows,
        flat: op.flat,
        estimate: op.est,
        bound: op.bnd,
        nanos: op.nanos,
        qerror: op.qerror(),
    }
}
