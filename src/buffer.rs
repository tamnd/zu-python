//! Python values, buffered as the columns the engine stores.
//!
//! Two callers want the same thing here and would otherwise want it
//! twice. A load reads whole columns and writes them once; an appender
//! reads whole rows and writes them a batch at a time. Both of them end
//! up holding a vector per column in the shape the property store
//! keeps it in, and both of them have to decide what a Python object
//! is on the way in.
//!
//! What a column holds is settled by the table it is being written to
//! when there is one, which is the appender's case, and by its first
//! value when there is not, which is the loader's: a load writes a
//! database that does not exist yet, so nothing but the values can say
//! what the columns are. There is no null either way. A column that
//! holds one cannot be loaded or appended to, so a null here could only
//! ever be refused, and refusing it where it is named is better than
//! refusing it at the end of a million rows.

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDate, PyDateTime, PyDelta, PyTime};
use zu_common::temporal::days_from_civil;
use zu_common::{DurationKind, FloatBits, IntBits, LogicalType, Temporal};
use zudb::Field;
use zudb::query::Value;

use crate::value::{Duration, clock_nanos};

/// One column's values, in the shape the property store wants them.
///
/// Owned rather than borrowed from the caller's lists, because a
/// Python list holds objects and the store holds numbers: there is
/// nothing here to borrow. The arms are the storage arms, so a flush
/// or a load hands a buffer straight over with no pass to convert it.
pub enum Column {
    Int(Vec<u64>),
    Float(Vec<f64>),
    Bool(Vec<bool>),
    /// Kept as strings rather than as bytes, because the store wants
    /// the bytes and the appender wants the `&str`, and a `String`
    /// lends out either without a copy.
    Str(Vec<String>),
    Bytes(Vec<Vec<u8>>),
    Date(Vec<i32>),
    LocalTime(Vec<i64>),
    LocalDatetime(Vec<i64>),
    Duration(DurationKind, Vec<i64>),
}

/// Why a value did not go in.
///
/// A column that wanted something else says what it holds and lets the
/// caller word the rest, because a load knows the column's name and a
/// row of an appender knows its position, and neither message reads
/// well in the other's place. Anything else is Python's own error and
/// is passed along as it is.
pub enum Mismatch {
    Wanted(&'static str),
    Python(PyErr),
}

impl From<PyErr> for Mismatch {
    fn from(err: PyErr) -> Mismatch {
        Mismatch::Python(err)
    }
}

impl Column {
    /// The buffer a column of this declared type appends into, or
    /// `None` for a type the ingest path cannot carry.
    ///
    /// The match is on the exact declared type and not on its family,
    /// because that is what the ingest checks: it compares the stored
    /// column's type against the type its values claim, so an `INT32`
    /// column or a `VARCHAR(20)` one has no buffer here even though its
    /// bits would fit the same lane. This is the engine appender's own
    /// table, kept in step with it, because a buffer it would refuse is
    /// better refused before a million rows go into it.
    pub fn for_type(ty: &LogicalType) -> Option<Column> {
        Some(match ty {
            LogicalType::Int {
                signed: true,
                bits: IntBits::B64,
                precision: None,
            } => Column::Int(Vec::new()),
            LogicalType::Bool => Column::Bool(Vec::new()),
            LogicalType::Float {
                bits: FloatBits::B64,
                precision: None,
            } => Column::Float(Vec::new()),
            LogicalType::Date => Column::Date(Vec::new()),
            LogicalType::LocalTime => Column::LocalTime(Vec::new()),
            LogicalType::LocalDatetime => Column::LocalDatetime(Vec::new()),
            LogicalType::Duration(kind) => Column::Duration(*kind, Vec::new()),
            LogicalType::Str {
                min: None,
                max: None,
                fixed: false,
            } => Column::Str(Vec::new()),
            LogicalType::Bytes {
                min: None,
                max: None,
                fixed: false,
            } => Column::Bytes(Vec::new()),
            _ => return None,
        })
    }

    /// The column a first value starts, with that value already in it,
    /// or `None` for a value no column here holds.
    ///
    /// The order matters twice. Bool before int, since in Python every
    /// bool is an int and a column of `True` is not a column of ones.
    /// Datetime before date, since a datetime is a date and reading one
    /// as the other would throw the time away.
    pub fn start(value: &Bound<'_, PyAny>) -> PyResult<Option<Column>> {
        if let Ok(b) = value.cast::<PyBool>() {
            return Ok(Some(Column::Bool(vec![b.is_true()])));
        }
        if let Ok(s) = value.extract::<String>() {
            return Ok(Some(Column::Str(vec![s])));
        }
        if let Ok(b) = value.cast::<PyBytes>() {
            return Ok(Some(Column::Bytes(vec![b.as_bytes().to_vec()])));
        }
        if let Ok(n) = value.extract::<i64>() {
            return Ok(Some(Column::Int(vec![n as u64])));
        }
        if let Ok(f) = value.extract::<f64>() {
            return Ok(Some(Column::Float(vec![f])));
        }
        if let Ok(d) = value.extract::<Duration>() {
            return Ok(Some(if d.months == 0 {
                Column::Duration(DurationKind::DayTime, vec![d.nanoseconds])
            } else {
                Column::Duration(DurationKind::YearMonth, vec![d.months])
            }));
        }
        if let Ok(delta) = value.cast::<PyDelta>() {
            return Ok(Some(Column::Duration(
                DurationKind::DayTime,
                vec![delta_nanos(delta)?],
            )));
        }
        if let Ok(dt) = value.cast::<PyDateTime>() {
            return Ok(Some(Column::LocalDatetime(vec![datetime_nanos(dt)?])));
        }
        if let Ok(date) = value.cast::<PyDate>() {
            return Ok(Some(Column::Date(vec![date_days(date)?])));
        }
        if let Ok(time) = value.cast::<PyTime>() {
            return Ok(Some(Column::LocalTime(vec![clock_nanos(time.as_any())?])));
        }
        Ok(None)
    }

    /// Takes one more value, or says what this column holds instead.
    ///
    /// The order of the arms is the order of [`Column::start`] and for
    /// the same two reasons: a bool is an int in Python and a datetime
    /// is a date, so each of those pairs has to be told apart before it
    /// is read.
    pub fn push(&mut self, value: &Bound<'_, PyAny>) -> Result<(), Mismatch> {
        let holds = self.holds();
        let wanted = || Mismatch::Wanted(holds);
        match self {
            Column::Bool(v) => v.push(value.cast::<PyBool>().map_err(|_| wanted())?.is_true()),
            Column::Int(v) => {
                if value.cast::<PyBool>().is_ok() {
                    return Err(wanted());
                }
                v.push(value.extract::<i64>().map_err(|_| wanted())? as u64);
            }
            Column::Float(v) => v.push(value.extract::<f64>().map_err(|_| wanted())?),
            Column::Str(v) => v.push(value.extract::<String>().map_err(|_| wanted())?),
            Column::Bytes(v) => v.push(
                value
                    .cast::<PyBytes>()
                    .map_err(|_| wanted())?
                    .as_bytes()
                    .to_vec(),
            ),
            Column::LocalDatetime(v) => {
                let dt = value.cast::<PyDateTime>().map_err(|_| wanted())?;
                v.push(datetime_nanos(dt)?);
            }
            Column::Date(v) => {
                if value.cast::<PyDateTime>().is_ok() {
                    return Err(wanted());
                }
                let date = value.cast::<PyDate>().map_err(|_| wanted())?;
                v.push(date_days(date)?);
            }
            Column::LocalTime(v) => {
                let time = value.cast::<PyTime>().map_err(|_| wanted())?;
                v.push(clock_nanos(time.as_any())?);
            }
            Column::Duration(kind, v) => {
                let kind = *kind;
                v.push(duration_count(kind, value).ok_or_else(wanted)??);
            }
        }
        Ok(())
    }

    /// Takes one more value, and turns a column of integers into a
    /// column of floats rather than refusing a float.
    ///
    /// The one mixture worth widening, and only where the column's type
    /// is not already settled by a table: a list written by hand as
    /// `[1, 2.5]` is a column of floats and refusing it would be
    /// pedantry. The values already in it are exactly what they were,
    /// up to the point where an integer stops fitting in a float, which
    /// is a different problem and a rarer one.
    pub fn widening_push(&mut self, value: &Bound<'_, PyAny>) -> Result<(), Mismatch> {
        if let Column::Int(v) = self
            && value.cast::<PyBool>().is_err()
            && value.extract::<i64>().is_err()
            && let Ok(f) = value.extract::<f64>()
        {
            let mut widened: Vec<f64> = v.iter().map(|&n| n as i64 as f64).collect();
            widened.push(f);
            *self = Column::Float(widened);
            return Ok(());
        }
        self.push(value)
    }

    /// What this column holds, for the message when it was handed
    /// something else. Plural, because it is the column that is being
    /// described and not the value.
    pub fn holds(&self) -> &'static str {
        match self {
            Column::Int(_) => "integers",
            Column::Float(_) => "floats",
            Column::Bool(_) => "booleans",
            Column::Str(_) => "strings",
            Column::Bytes(_) => "byte strings",
            Column::Date(_) => "dates",
            Column::LocalTime(_) => "times",
            Column::LocalDatetime(_) => "datetimes",
            Column::Duration(DurationKind::YearMonth, _) => "year-month durations",
            Column::Duration(DurationKind::DayTime, _) => "day-time durations",
        }
    }

    /// How many values are in it, which is how many rows the table has
    /// if this is the first column and a refusal if it is not.
    pub fn len(&self) -> usize {
        match self {
            Column::Int(v) => v.len(),
            Column::Float(v) => v.len(),
            Column::Bool(v) => v.len(),
            Column::Str(v) => v.len(),
            Column::Bytes(v) => v.len(),
            Column::Date(v) => v.len(),
            Column::LocalTime(v) | Column::LocalDatetime(v) => v.len(),
            Column::Duration(_, v) => v.len(),
        }
    }

    /// Drops the value written last, which is how a refused row takes
    /// back the fields it managed to write before the one that failed.
    pub fn pop(&mut self) {
        match self {
            Column::Int(v) => drop(v.pop()),
            Column::Float(v) => drop(v.pop()),
            Column::Bool(v) => drop(v.pop()),
            Column::Str(v) => drop(v.pop()),
            Column::Bytes(v) => drop(v.pop()),
            Column::Date(v) => drop(v.pop()),
            Column::LocalTime(v) | Column::LocalDatetime(v) => drop(v.pop()),
            Column::Duration(_, v) => drop(v.pop()),
        }
    }

    pub fn clear(&mut self) {
        match self {
            Column::Int(v) => v.clear(),
            Column::Float(v) => v.clear(),
            Column::Bool(v) => v.clear(),
            Column::Str(v) => v.clear(),
            Column::Bytes(v) => v.clear(),
            Column::Date(v) => v.clear(),
            Column::LocalTime(v) | Column::LocalDatetime(v) => v.clear(),
            Column::Duration(_, v) => v.clear(),
        }
    }

    /// One value of this column as a statement's parameter takes it.
    ///
    /// Owned rather than borrowed, unlike `field`, because a parameter
    /// is bound for the length of a statement and the statement is
    /// where the copy was always going to happen. `None` for a column
    /// of byte strings, which no statement carries a literal or a
    /// parameter for.
    pub fn value(&self, row: usize) -> Option<Value> {
        Some(match self {
            Column::Int(v) => Value::Int(v[row] as i64),
            Column::Float(v) => Value::Float(v[row]),
            Column::Bool(v) => Value::Bool(v[row]),
            Column::Str(v) => Value::Str(v[row].clone()),
            Column::Bytes(_) => return None,
            Column::Date(v) => Value::Temporal(Temporal::Date(v[row])),
            Column::LocalTime(v) => Value::Temporal(Temporal::LocalTime(v[row])),
            Column::LocalDatetime(v) => Value::Temporal(Temporal::LocalDatetime(v[row])),
            Column::Duration(kind, v) => Value::Temporal(Temporal::Duration(*kind, v[row])),
        })
    }

    /// One value of this column as the engine's appender takes it.
    ///
    /// A field borrows rather than owning, which is the point of it on
    /// a string column: the buffer already holds the bytes and the
    /// appender is about to copy them into its own, so lending them is
    /// the difference between one copy per row and two.
    pub fn field(&self, row: usize) -> Field<'_> {
        match self {
            Column::Int(v) => Field::Int(v[row] as i64),
            Column::Float(v) => Field::Float(v[row]),
            Column::Bool(v) => Field::Bool(v[row]),
            Column::Str(v) => Field::Str(&v[row]),
            Column::Bytes(v) => Field::Bytes(&v[row]),
            Column::Date(v) => Field::Temporal(Temporal::Date(v[row])),
            Column::LocalTime(v) => Field::Temporal(Temporal::LocalTime(v[row])),
            Column::LocalDatetime(v) => Field::Temporal(Temporal::LocalDatetime(v[row])),
            Column::Duration(kind, v) => Field::Temporal(Temporal::Duration(*kind, v[row])),
        }
    }
}

/// A duration for a column that is already one of the two kinds, or
/// `None` for a duration of the other kind, which is the one place the
/// two do not mix: a column of months has no room for a count of
/// nanoseconds and the other way about.
fn duration_count(kind: DurationKind, value: &Bound<'_, PyAny>) -> Option<PyResult<i64>> {
    if let Ok(d) = value.extract::<Duration>() {
        return match kind {
            DurationKind::YearMonth if d.months != 0 || d.nanoseconds == 0 => Some(Ok(d.months)),
            DurationKind::DayTime if d.months == 0 => Some(Ok(d.nanoseconds)),
            _ => None,
        };
    }
    match (kind, value.cast::<PyDelta>()) {
        (DurationKind::DayTime, Ok(delta)) => Some(delta_nanos(delta)),
        _ => None,
    }
}

/// The name a Python type goes by, for a message about a value that was
/// the wrong one.
pub fn type_name(value: &Bound<'_, PyAny>) -> String {
    value
        .get_type()
        .getattr("__name__")
        .and_then(|name| name.extract::<String>())
        .unwrap_or_else(|_| "unknown".to_string())
}

pub fn date_days(date: &Bound<'_, PyDate>) -> PyResult<i32> {
    Ok(days_from_civil(
        date.getattr("year")?.extract()?,
        date.getattr("month")?.extract()?,
        date.getattr("day")?.extract()?,
    ))
}

pub fn datetime_nanos(dt: &Bound<'_, PyDateTime>) -> PyResult<i64> {
    const NANOS_PER_DAY: i64 = 86_400 * 1_000_000_000;
    let days = date_days(dt.as_any().cast::<PyDate>()?)?;
    Ok(i64::from(days) * NANOS_PER_DAY + clock_nanos(dt.as_any())?)
}

pub fn delta_nanos(delta: &Bound<'_, PyDelta>) -> PyResult<i64> {
    let days: i64 = delta.getattr("days")?.extract()?;
    let seconds: i64 = delta.getattr("seconds")?.extract()?;
    let micros: i64 = delta.getattr("microseconds")?.extract()?;
    Ok(days * 86_400 * 1_000_000_000 + seconds * 1_000_000_000 + micros * 1_000)
}
