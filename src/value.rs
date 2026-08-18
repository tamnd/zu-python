//! Values, both ways across the boundary.
//!
//! A row that arrives as Python objects is the slow path and the one
//! every notebook starts with, so it is the one that has to be
//! obvious: nulls are `None`, integers are `int`, and a graph value is
//! a class with named fields rather than a tuple whose third element
//! is the ordinal if you remember the order. The fast path is Arrow,
//! and nothing here is on it.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDate, PyDateTime, PyDelta, PyDict, PyList, PyTime, PyTuple, PyTzInfo};
use zu_common::temporal::{NANOS_PER_DAY, NANOS_PER_MINUTE, civil_from_days, days_from_civil};
use zu_common::{DurationKind, Temporal};
use zudb::query::Value;
use zudb::zu1::catalog::Catalog;

use crate::html;

/// Microseconds in a day, which is the unit `datetime.timedelta`
/// normalizes to and the one this has to split a duration across.
const MICROS_PER_DAY: i64 = 86_400 * 1_000_000;

/// What the tables in a result are called.
///
/// A node value carries the id of the table it came from and nothing
/// else, because that is what a row holds. A person reading a result
/// wants the name, so the names are taken off the catalog once when a
/// statement runs and carried with the rows. A catalog holds tens of
/// tables, so this is a copy of a few short strings and not a
/// structure worth sharing.
#[derive(Default)]
pub struct Names {
    nodes: HashMap<u32, String>,
    rels: HashMap<u32, String>,
}

impl Names {
    pub fn of(catalog: &Catalog) -> Names {
        Names {
            nodes: catalog
                .node_tables()
                .iter()
                .map(|table| (table.id, table.name.clone()))
                .collect(),
            rels: catalog
                .rel_tables()
                .iter()
                .map(|table| (table.id, table.name.clone()))
                .collect(),
        }
    }

    /// The table's name, or its id written out for a table the catalog
    /// no longer has. A result outlives nothing here, but a name is
    /// for reading and an unreadable one should still print.
    pub fn node(&self, id: u32) -> String {
        self.node_name(id)
            .map(str::to_owned)
            .unwrap_or_else(|| format!("#{id}"))
    }

    pub fn rel(&self, id: u32) -> String {
        self.rel_name(id)
            .map(str::to_owned)
            .unwrap_or_else(|| format!("#{id}"))
    }

    /// The same names, borrowed. A column of a hundred million nodes
    /// holds a handful of distinct table names, and copying one of them
    /// per row is the difference between building a string column and
    /// allocating one.
    pub fn node_name(&self, id: u32) -> Option<&str> {
        self.nodes.get(&id).map(String::as_str)
    }

    pub fn rel_name(&self, id: u32) -> Option<&str> {
        self.rels.get(&id).map(String::as_str)
    }
}

/// One node of the graph.
///
/// The table is the name a person wrote in the schema, and the offset
/// is the row it sits at in that table, which together are what
/// identifies a node in zu. Properties and the full label set arrive
/// on a later release; what is here is what a row carries.
#[pyclass(module = "zudb", frozen, eq, hash, skip_from_py_object)]
#[derive(PartialEq, Eq, Hash, Clone)]
pub struct Node {
    #[pyo3(get)]
    pub table: String,
    #[pyo3(get)]
    pub offset: u64,
}

#[pymethods]
impl Node {
    #[new]
    fn new(table: String, offset: u64) -> Node {
        Node { table, offset }
    }

    fn __repr__(&self) -> String {
        format!("Node({}, {})", self.table, self.offset)
    }

    /// The node as a notebook draws it, which is the pair that names
    /// it: the table it is in and the row it sits at.
    fn _repr_html_(&self) -> String {
        html::wrap(&html::node(self))
    }
}

/// One edge of the graph.
///
/// `ord` is where the edge's properties sit, which is its place in the
/// order the table was loaded in. That is what names an edge: a pair of
/// endpoints does not, since the same pair may run more than once and
/// each of those edges carries its own values. It is the field a caller
/// usually ignores and the one nothing else can replace.
#[pyclass(module = "zudb", frozen, eq, hash, skip_from_py_object)]
#[derive(PartialEq, Eq, Hash, Clone)]
pub struct Rel {
    #[pyo3(get)]
    pub table: String,
    #[pyo3(get)]
    pub src: u64,
    #[pyo3(get)]
    pub dst: u64,
    #[pyo3(get)]
    pub ord: u64,
}

#[pymethods]
impl Rel {
    #[new]
    fn new(table: String, src: u64, dst: u64, ord: u64) -> Rel {
        Rel {
            table,
            src,
            dst,
            ord,
        }
    }

    fn __repr__(&self) -> String {
        format!("Rel({}, {} -> {})", self.table, self.src, self.dst)
    }

    /// The edge as a notebook draws it, with the rows it joins on
    /// either side of the arrow, since nothing beside it says which
    /// they are.
    fn _repr_html_(&self) -> String {
        html::wrap(&html::rel(self))
    }
}

/// A walk: nodes and edges alternating, a node at each end.
///
/// `len` is the number of edges, which is the length of a path
/// everywhere else it is spoken about, so a path of one node has
/// length zero and is the shortest path there is.
#[pyclass(module = "zudb", frozen, skip_from_py_object)]
pub struct Path {
    #[pyo3(get)]
    pub elements: Py<PyList>,
}

#[pymethods]
impl Path {
    #[new]
    fn new(elements: Py<PyList>) -> Path {
        Path { elements }
    }

    /// The nodes of the walk, in the order it visits them.
    #[getter]
    fn nodes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        self.every(py, 0)
    }

    /// The edges of the walk, in the order it crosses them.
    #[getter]
    fn rels<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        self.every(py, 1)
    }

    fn __len__(&self, py: Python<'_>) -> usize {
        self.elements.bind(py).len() / 2
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        format!("Path({} hops)", self.__len__(py))
    }

    /// The walk as a notebook draws it, nodes and arrows alternating
    /// the way a statement writes one.
    fn _repr_html_(&self, py: Python<'_>) -> PyResult<String> {
        Ok(html::wrap(&html::path(py, self)?))
    }
}

impl Path {
    /// Every other element starting at `from`, which is the nodes when
    /// that is zero and the edges when it is one. Outside `pymethods`
    /// because it is how the two getters are written and not a third
    /// thing to call from Python.
    fn every<'py>(&self, py: Python<'py>, from: usize) -> PyResult<Bound<'py, PyList>> {
        let all = self.elements.bind(py);
        let picked = (from..all.len())
            .step_by(2)
            .map(|ix| all.get_item(ix))
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, picked)
    }
}

/// A duration, which Python has no type for.
///
/// `datetime.timedelta` is a day-time duration rounded to
/// microseconds, so it can hold neither of the two things zu stores: a
/// year-month duration is a count of months and no number of days is a
/// month, and a day-time duration is counted in nanoseconds. The
/// conversion is offered rather than done, so a caller who wants a
/// `timedelta` asks for one and knows what they gave up.
#[pyclass(module = "zudb", frozen, eq, hash, from_py_object)]
#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub struct Duration {
    /// Months, for a year-month duration. Zero for a day-time one.
    #[pyo3(get)]
    pub months: i64,
    /// Nanoseconds, for a day-time duration. Zero for a year-month
    /// one.
    #[pyo3(get)]
    pub nanoseconds: i64,
}

#[pymethods]
impl Duration {
    #[new]
    #[pyo3(signature = (months = 0, nanoseconds = 0))]
    fn new(months: i64, nanoseconds: i64) -> PyResult<Duration> {
        if months != 0 && nanoseconds != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "a duration counts months or nanoseconds and never both, since no number of days is a month",
            ));
        }
        Ok(Duration {
            months,
            nanoseconds,
        })
    }

    /// `"year_month"` or `"day_time"`, which is the distinction the
    /// standard draws and the reason the two counts do not mix.
    #[getter]
    fn kind(&self) -> &'static str {
        if self.months == 0 {
            "day_time"
        } else {
            "year_month"
        }
    }

    /// The same duration as a `datetime.timedelta`, rounded towards
    /// zero to the microsecond it can hold. A year-month duration has
    /// no answer and says so.
    #[allow(clippy::wrong_self_convention)] // `to_timedelta` is the name Python readers expect
    fn to_timedelta<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDelta>> {
        if self.months != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "a year-month duration is not a number of days, so there is no timedelta for it",
            ));
        }
        // Split across the three counts the constructor takes rather
        // than handed over as microseconds, because each of them is a
        // C int and an hour is already more microseconds than one
        // holds. The remainders keep their sign, which is what makes
        // a negative duration round towards zero the way a positive
        // one does.
        let micros = self.nanoseconds / 1_000;
        let days = micros / MICROS_PER_DAY;
        let rest = micros % MICROS_PER_DAY;
        let seconds = rest / 1_000_000;
        let micros = rest % 1_000_000;
        let count = |value: i64| {
            i32::try_from(value).map_err(|_| {
                pyo3::exceptions::PyOverflowError::new_err(
                    "this duration is longer than a timedelta can hold",
                )
            })
        };
        PyDelta::new(py, count(days)?, count(seconds)?, count(micros)?, true)
    }

    fn __repr__(&self) -> String {
        if self.months == 0 {
            format!("Duration(nanoseconds={})", self.nanoseconds)
        } else {
            format!("Duration(months={})", self.months)
        }
    }
}

/// One engine value as the Python object it is.
pub fn to_py<'py>(py: Python<'py>, value: &Value, names: &Names) -> PyResult<Bound<'py, PyAny>> {
    Ok(match value {
        Value::Null => py.None().into_bound(py),
        Value::Bool(b) => PyBool::new(py, *b).to_owned().into_any(),
        Value::Int(n) => n.into_pyobject(py)?.into_any(),
        Value::Float(f) => f.into_pyobject(py)?.into_any(),
        Value::Str(s) => s.into_pyobject(py)?.into_any(),
        Value::Node { table, offset } => Node {
            table: names.node(*table),
            offset: *offset,
        }
        .into_pyobject(py)?
        .into_any(),
        Value::Rel {
            table,
            src,
            dst,
            ord,
        } => Rel {
            table: names.rel(*table),
            src: *src,
            dst: *dst,
            ord: *ord,
        }
        .into_pyobject(py)?
        .into_any(),
        Value::List(items) => {
            let out = PyList::empty(py);
            for item in items {
                out.append(to_py(py, item, names)?)?;
            }
            out.into_any()
        }
        Value::Record(fields) => {
            let out = PyDict::new(py);
            for (name, item) in fields {
                out.set_item(name, to_py(py, item, names)?)?;
            }
            out.into_any()
        }
        Value::Path(walk) => {
            let out = PyList::empty(py);
            for step in walk {
                out.append(to_py(py, step, names)?)?;
            }
            Path {
                elements: out.unbind(),
            }
            .into_pyobject(py)?
            .into_any()
        }
        Value::Temporal(t) => temporal_to_py(py, *t)?,
        // GV60 and GV61. A reference goes across as the string that
        // names it, `GRAPH /social` or `BINDING TABLE #3 (2 columns, 7
        // rows)`, which is what the shell prints and what the ABI's
        // JSON carries. A handle is a reference on purpose: the graph
        // is in the file and the table is behind the handle, so
        // building a Python object that held either would copy the
        // thing the value was passed by reference to avoid.
        Value::Graph(handle) => handle.label().into_pyobject(py)?.into_any(),
        Value::BindingTable(table) => table.label().into_pyobject(py)?.into_any(),
        // The executor settles a chain into its edge list before any
        // value leaves the pipeline, so a result never holds one and
        // seeing one here is a bug in the engine rather than something
        // a caller did.
        Value::Chain(_) => {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "an unsettled path chain reached a result, which is a bug: please report it at https://github.com/tamnd/zu/issues",
            ));
        }
    })
}

fn temporal_to_py<'py>(py: Python<'py>, t: Temporal) -> PyResult<Bound<'py, PyAny>> {
    Ok(match t {
        Temporal::Date(days) => {
            let (y, m, d) = civil_from_days(days);
            PyDate::new(py, y, m as u8, d as u8)?.into_any()
        }
        Temporal::LocalTime(nanos) => time_of(py, nanos, None)?.into_any(),
        Temporal::ZonedTime { nanos, offset } => {
            let zone = zone_of(py, offset)?;
            time_of(py, nanos, Some(&zone))?.into_any()
        }
        Temporal::LocalDatetime(nanos) => datetime_of(py, nanos, None)?.into_any(),
        Temporal::ZonedDatetime { nanos, offset } => {
            // Stored as the instant in UTC with the offset it was
            // written in, so the local reading is the instant moved
            // back into that offset. Handing back the instant with a
            // zone attached is what makes `.astimezone` and equality
            // both answer what the writer meant.
            let zone = zone_of(py, offset)?;
            let local = nanos.saturating_add(i64::from(offset) * NANOS_PER_MINUTE);
            datetime_of(py, local, Some(&zone))?.into_any()
        }
        Temporal::Duration(DurationKind::YearMonth, months) => Duration {
            months,
            nanoseconds: 0,
        }
        .into_pyobject(py)?
        .into_any(),
        Temporal::Duration(DurationKind::DayTime, nanos) => Duration {
            months: 0,
            nanoseconds: nanos,
        }
        .into_pyobject(py)?
        .into_any(),
    })
}

/// A nanosecond count since midnight as a `datetime.time`.
///
/// Sub-microsecond digits are dropped, because `datetime` has no place
/// to put them. The Arrow path keeps every one of them, which is the
/// answer for a caller who cannot lose them.
fn time_of<'py>(
    py: Python<'py>,
    nanos: i64,
    zone: Option<&Bound<'py, PyTzInfo>>,
) -> PyResult<Bound<'py, PyTime>> {
    let nanos = nanos.rem_euclid(NANOS_PER_DAY);
    let micros = (nanos / 1_000) % 1_000_000;
    let secs = nanos / 1_000_000_000;
    PyTime::new(
        py,
        (secs / 3_600) as u8,
        ((secs / 60) % 60) as u8,
        (secs % 60) as u8,
        micros as u32,
        zone,
    )
}

/// A nanosecond count since the epoch as a `datetime.datetime`.
fn datetime_of<'py>(
    py: Python<'py>,
    nanos: i64,
    zone: Option<&Bound<'py, PyTzInfo>>,
) -> PyResult<Bound<'py, PyDateTime>> {
    let days = nanos.div_euclid(NANOS_PER_DAY);
    let rest = nanos.rem_euclid(NANOS_PER_DAY);
    let days = i32::try_from(days).map_err(|_| {
        pyo3::exceptions::PyOverflowError::new_err("this datetime is outside the calendar")
    })?;
    let (y, m, d) = civil_from_days(days);
    let micros = (rest / 1_000) % 1_000_000;
    let secs = rest / 1_000_000_000;
    PyDateTime::new(
        py,
        y,
        m as u8,
        d as u8,
        (secs / 3_600) as u8,
        ((secs / 60) % 60) as u8,
        (secs % 60) as u8,
        micros as u32,
        zone,
    )
}

/// An offset in minutes as a `datetime.timezone`.
fn zone_of(py: Python<'_>, offset: i16) -> PyResult<Bound<'_, PyTzInfo>> {
    PyTzInfo::fixed_offset(py, PyDelta::new(py, 0, i32::from(offset) * 60, 0, true)?)
}

/// One Python object as the engine value it is, for a parameter.
///
/// The refusals are as much of the surface as the conversions. A
/// parameter zu cannot hold has to fail at the call rather than
/// arriving as a string, because a query that silently compared a
/// number against its own spelling would answer nothing and say
/// nothing.
pub fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    if value.is_none() {
        return Ok(Value::Null);
    }
    // Before the integer arm, because in Python every bool is an int
    // and a parameter of `True` is not the parameter `1`.
    if let Ok(b) = value.cast::<PyBool>() {
        return Ok(Value::Bool(b.is_true()));
    }
    if let Ok(s) = value.extract::<String>() {
        return Ok(Value::Str(s));
    }
    if let Ok(n) = value.extract::<i64>() {
        return Ok(Value::Int(n));
    }
    if let Ok(f) = value.extract::<f64>() {
        return Ok(Value::Float(f));
    }
    if let Ok(d) = value.extract::<Duration>() {
        return Ok(Value::Temporal(if d.months == 0 {
            Temporal::Duration(DurationKind::DayTime, d.nanoseconds)
        } else {
            Temporal::Duration(DurationKind::YearMonth, d.months)
        }));
    }
    // Datetime before date, since `datetime.datetime` is a subclass of
    // `datetime.date` and reading one as the other would silently
    // throw the time away.
    if let Ok(dt) = value.cast::<PyDateTime>() {
        return datetime_from_py(dt);
    }
    if let Ok(d) = value.cast::<PyDate>() {
        return Ok(Value::Temporal(Temporal::Date(days_from_civil(
            d.getattr("year")?.extract()?,
            d.getattr("month")?.extract()?,
            d.getattr("day")?.extract()?,
        ))));
    }
    if let Ok(t) = value.cast::<PyTime>() {
        return time_from_py(t);
    }
    if let Ok(delta) = value.cast::<PyDelta>() {
        let days: i64 = delta.getattr("days")?.extract()?;
        let secs: i64 = delta.getattr("seconds")?.extract()?;
        let micros: i64 = delta.getattr("microseconds")?.extract()?;
        return Ok(Value::Temporal(Temporal::Duration(
            DurationKind::DayTime,
            days * NANOS_PER_DAY + secs * 1_000_000_000 + micros * 1_000,
        )));
    }
    if let Ok(items) = value.cast::<PyList>() {
        return Ok(Value::List(
            items
                .iter()
                .map(|item| from_py(&item))
                .collect::<PyResult<_>>()?,
        ));
    }
    if let Ok(items) = value.cast::<PyTuple>() {
        return Ok(Value::List(
            items
                .iter()
                .map(|item| from_py(&item))
                .collect::<PyResult<_>>()?,
        ));
    }
    if let Ok(fields) = value.cast::<PyDict>() {
        let mut out = Vec::with_capacity(fields.len());
        for (name, item) in fields.iter() {
            out.push((name.extract::<String>()?, from_py(&item)?));
        }
        return Ok(Value::record(out));
    }
    Err(refused(value))
}

fn datetime_from_py(dt: &Bound<'_, PyDateTime>) -> PyResult<Value> {
    let days = days_from_civil(
        dt.getattr("year")?.extract()?,
        dt.getattr("month")?.extract()?,
        dt.getattr("day")?.extract()?,
    );
    let nanos = i64::from(days) * NANOS_PER_DAY + clock_nanos(dt.as_any())?;
    Ok(Value::Temporal(match offset_of(dt.as_any())? {
        // Stored as the instant, which is the local reading with the
        // offset taken back off it, and the offset kept beside it so
        // the value still prints in the zone it was written in.
        Some(offset) => Temporal::ZonedDatetime {
            nanos: nanos - i64::from(offset) * NANOS_PER_MINUTE,
            offset,
        },
        None => Temporal::LocalDatetime(nanos),
    }))
}

fn time_from_py(t: &Bound<'_, PyTime>) -> PyResult<Value> {
    let nanos = clock_nanos(t.as_any())?;
    Ok(Value::Temporal(match offset_of(t.as_any())? {
        Some(offset) => Temporal::ZonedTime { nanos, offset },
        None => Temporal::LocalTime(nanos),
    }))
}

/// The clock reading of a `time` or a `datetime`, in nanoseconds since
/// midnight.
pub fn clock_nanos(value: &Bound<'_, PyAny>) -> PyResult<i64> {
    let hour: i64 = value.getattr("hour")?.extract()?;
    let minute: i64 = value.getattr("minute")?.extract()?;
    let second: i64 = value.getattr("second")?.extract()?;
    let micro: i64 = value.getattr("microsecond")?.extract()?;
    Ok(((hour * 60 + minute) * 60 + second) * 1_000_000_000 + micro * 1_000)
}

/// The offset from UTC in minutes, for a value that carries one.
///
/// A zone that is a rule rather than an offset, `ZoneInfo` above all,
/// is asked what its offset is for this value and stored as that. The
/// name is not stored, on purpose: a name is a rule that changes when
/// the zone database is updated, and a value that means something
/// different tomorrow is not a value.
fn offset_of(value: &Bound<'_, PyAny>) -> PyResult<Option<i16>> {
    let offset = value.call_method0("utcoffset")?;
    if offset.is_none() {
        return Ok(None);
    }
    let seconds: f64 = offset.call_method0("total_seconds")?.extract()?;
    Ok(Some((seconds / 60.0) as i16))
}

fn refused(value: &Bound<'_, PyAny>) -> PyErr {
    let name = value
        .get_type()
        .getattr("__name__")
        .and_then(|name| name.extract::<String>())
        .unwrap_or_else(|_| "that".to_string());
    pyo3::exceptions::PyTypeError::new_err(format!(
        "a parameter cannot be a {name}: zu holds nulls, booleans, integers, floats, strings, lists, records, dates, times, datetimes and durations"
    ))
}
