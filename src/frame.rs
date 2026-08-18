//! A frame of columns, read out of whatever the caller is holding.
//!
//! The way in for data that is already in columns. pandas, polars and
//! pyarrow all hand out an Arrow stream through the PyCapsule protocol,
//! and reading one costs a memcpy per column and no Python objects at
//! all, which is the same trade the way out makes and for the same
//! reason.
//!
//! What comes back is the same [`Column`] buffers a load and an
//! appender already use, so nothing downstream of here knows whether
//! the rows arrived as Arrow or as Python lists.
//!
//! There is no null anywhere in it. A property that is null is one no
//! row of this engine can hold, so a column with a gap in it can only
//! ever be refused, and refusing it by name and row number is the
//! difference between a caller who knows which cell to fix and one who
//! knows only that something somewhere was empty.

use std::ffi::CStr;

use arrow::array::{
    Array, ArrayRef, BooleanArray, LargeStringArray, PrimitiveArray, StringArray, StringViewArray,
};
use arrow::datatypes::{
    DataType, Date32Type, DurationMicrosecondType, DurationMillisecondType, DurationNanosecondType,
    DurationSecondType, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type,
    IntervalUnit, IntervalYearMonthType, Time32MillisecondType, Time32SecondType,
    Time64MicrosecondType, Time64NanosecondType, TimeUnit, TimestampMicrosecondType,
    TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType, UInt8Type, UInt16Type,
    UInt32Type, UInt64Type,
};
use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow::record_batch::{RecordBatch, RecordBatchReader};
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict};
use zu_common::DurationKind;

use crate::buffer::{Column, type_name};
use crate::columns::Snag;
use crate::load;

/// What a capsule holding an Arrow stream is called, which a consumer
/// checks before it reads the pointer.
const STREAM: &CStr = c"arrow_array_stream";

/// Columns, in the order the frame holds them, and how long they are.
pub struct Frame {
    pub columns: Vec<(String, Column)>,
    pub rows: usize,
}

/// Reads whatever the caller handed over into columns.
///
/// Two shapes, and the first of them is the one that matters: anything
/// that speaks Arrow, which is every frame library worth naming, and a
/// dictionary of lists for a caller with none of them installed.
/// Anything else is refused here rather than iterated hopefully,
/// because the message a caller wants is the list of what would have
/// worked.
pub fn read(py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<Frame> {
    if data.hasattr("__arrow_c_stream__")? {
        return from_arrow(py, data);
    }
    if let Ok(dict) = data.cast::<PyDict>() {
        let columns = load::build(Some(dict))?;
        let rows = columns.first().map(|(_, column)| column.len()).unwrap_or(0);
        return Ok(Frame { columns, rows });
    }
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "a frame is anything with `__arrow_c_stream__`, which a pandas, polars or pyarrow table \
         has, or a dictionary of column name to values, and this is a '{}'",
        type_name(data)
    )))
}

/// Reads the Arrow stream a frame hands out through the capsule
/// protocol.
///
/// The GIL is held to pull each batch, because the producer on the far
/// side of the stream may be Python and calling into an interpreter
/// nobody is holding is how a process ends. It goes down again for the
/// conversion, which is the part that touches every value and is pure
/// Rust: a ten million row frame is a memcpy per column rather than ten
/// million reference counts.
fn from_arrow(py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<Frame> {
    // No requested schema. Asking for one would mean casting on the
    // producer's side, and a producer that cannot cast is entitled to
    // refuse: what is wanted here is whatever it already has.
    let capsule = data.call_method1("__arrow_c_stream__", (py.None(),))?;
    let capsule = capsule.cast_into::<PyCapsule>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "`__arrow_c_stream__` gave back something that is not a capsule, so this frame does \
             not speak the protocol it says it speaks",
        )
    })?;
    // Checked by name, which is what tells a stream from every other
    // capsule a library might hand out, and the pointer comes back
    // from the same call so there is no way to read one without the
    // other.
    let held = capsule.pointer_checked(Some(STREAM)).map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "`__arrow_c_stream__` gave back a capsule of some other kind, which is not a stream to \
             read",
        )
    })?;
    // Moved out and replaced with an empty one, which is the protocol's
    // own rule: the consumer owns the stream from here and the
    // capsule's destructor has to find nothing left to release.
    let stream = unsafe {
        let held = held.as_ptr() as *mut FFI_ArrowArrayStream;
        std::ptr::replace(held, FFI_ArrowArrayStream::empty())
    };
    let reader =
        ArrowArrayStreamReader::try_new(stream).map_err(|err| Snag::Arrow(err).raise(py))?;

    let schema = reader.schema();
    let mut columns: Vec<(String, Column)> = Vec::with_capacity(schema.fields().len());
    for field in schema.fields() {
        columns.push((
            field.name().clone(),
            start(field.name(), field.data_type())?,
        ));
    }
    let mut rows = 0usize;
    for batch in reader {
        let batch = batch.map_err(|err| Snag::Arrow(err).raise(py))?;
        let base = rows;
        py.detach(|| take(&mut columns, &batch, base))
            .map_err(|snag| snag.raise(py))?;
        rows += batch.num_rows();
    }
    Ok(Frame { columns, rows })
}

/// The buffer a column of this Arrow type fills, or the reason there
/// is none.
///
/// The type decides it and not the first value, which is the one thing
/// a frame gives that a list of Python objects does not: an empty frame
/// still knows what its columns are, and a column of floats that
/// happens to start with a whole number is still a column of floats.
fn start(name: &str, ty: &DataType) -> PyResult<Column> {
    let refused = |instead: &str| {
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "column '{name}' is {ty}, and {instead}"
        )))
    };
    Ok(match ty {
        DataType::Boolean => Column::Bool(Vec::new()),
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64 => Column::Int(Vec::new()),
        DataType::Float32 | DataType::Float64 => Column::Float(Vec::new()),
        // The three string layouts of Arrow, which are three ways of
        // holding the same characters: offsets into one buffer, wider
        // offsets into one buffer, and views into several. polars hands
        // out the third by default and pandas the first, and a column of
        // this table holds strings either way.
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => Column::Str(Vec::new()),
        DataType::Date32 => Column::Date(Vec::new()),
        DataType::Time32(_) | DataType::Time64(_) => Column::LocalTime(Vec::new()),
        DataType::Timestamp(_, None) => Column::LocalDatetime(Vec::new()),
        DataType::Timestamp(_, Some(zone)) => {
            return refused(&format!(
                "a column of this table has nowhere to keep '{zone}', so drop the zone once the \
                 values are in the zone you want them in, or write it as a string"
            ));
        }
        DataType::Duration(_) => Column::Duration(DurationKind::DayTime, Vec::new()),
        DataType::Interval(IntervalUnit::YearMonth) => {
            Column::Duration(DurationKind::YearMonth, Vec::new())
        }
        DataType::Dictionary(_, _) => {
            return refused(
                "a dictionary is a layout rather than a type here, so cast it to what it holds \
                 first",
            );
        }
        DataType::Binary | DataType::LargeBinary | DataType::BinaryView => {
            return refused(
                "no statement can read a column of bytes back yet, so writing one would be \
                 writing data the caller cannot get at",
            );
        }
        _ => {
            return refused(
                "a column holds booleans, integers, floats, strings, dates, times, datetimes or \
                 durations",
            );
        }
    })
}

/// Every column of one batch, appended to the buffers it belongs in.
///
/// Runs with the GIL released and so cannot raise: what goes wrong
/// comes back as a [`Snag`] and is raised by the caller.
fn take(columns: &mut [(String, Column)], batch: &RecordBatch, base: usize) -> Result<(), Snag> {
    for (at, (name, column)) in columns.iter_mut().enumerate() {
        let array = batch.column(at);
        if array.null_count() > 0 {
            let row = (0..array.len())
                .find(|&row| array.is_null(row))
                .unwrap_or(0);
            return Err(Snag::Value(format!(
                "column '{name}' has no value at row {}, and every column of a row holds one",
                base + row
            )));
        }
        one(name, column, array, base)?;
    }
    Ok(())
}

/// Every value of a primitive array, converted and pushed.
///
/// A macro rather than a function because the conversion differs per
/// arm in both directions: a `Time32` in seconds and an `Int32` are
/// the same bits and not the same value, and an integer column keeps
/// its bits where a temporal one keeps a count. Twenty arms of the same
/// three lines is what this saves.
macro_rules! pushed {
    ($array:expr, $ty:ty, $out:expr, $changed:expr, $convert:expr) => {{
        let values = $array
            .as_any()
            .downcast_ref::<PrimitiveArray<$ty>>()
            .ok_or_else($changed)?;
        $out.extend(values.values().iter().copied().map($convert));
    }};
}

/// One column of one batch.
///
/// The pair is matched rather than the array alone, so a stream that
/// changed a column's type between batches is caught here instead of
/// filling one buffer with values of two kinds. It cannot happen
/// through a producer that keeps to its own schema, which is why the
/// arm says so rather than explaining itself.
fn one(name: &str, column: &mut Column, array: &ArrayRef, base: usize) -> Result<(), Snag> {
    let changed = || {
        Snag::Type(format!(
            "column '{name}' arrived as {} in a later batch than the one that declared it, which \
             is a producer that broke its own schema",
            array.data_type()
        ))
    };
    match (array.data_type(), column) {
        (DataType::Boolean, Column::Bool(out)) => {
            let values = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(changed)?;
            out.extend(values.values().iter());
        }
        (DataType::Int8, Column::Int(out)) => {
            pushed!(array, Int8Type, out, changed, |n| n as i64 as u64)
        }
        (DataType::Int16, Column::Int(out)) => {
            pushed!(array, Int16Type, out, changed, |n| n as i64 as u64)
        }
        (DataType::Int32, Column::Int(out)) => {
            pushed!(array, Int32Type, out, changed, |n| n as i64 as u64)
        }
        (DataType::Int64, Column::Int(out)) => {
            pushed!(array, Int64Type, out, changed, |n| n as u64)
        }
        (DataType::UInt8, Column::Int(out)) => {
            pushed!(array, UInt8Type, out, changed, u64::from)
        }
        (DataType::UInt16, Column::Int(out)) => {
            pushed!(array, UInt16Type, out, changed, u64::from)
        }
        (DataType::UInt32, Column::Int(out)) => {
            pushed!(array, UInt32Type, out, changed, u64::from)
        }
        (DataType::UInt64, Column::Int(out)) => {
            // The one integer width that does not fit. A value above
            // what an INT64 holds is refused where it sits rather than
            // written as a negative number, which is the kind of thing
            // a caller finds out about a year later.
            let values = array
                .as_any()
                .downcast_ref::<PrimitiveArray<UInt64Type>>()
                .ok_or_else(changed)?;
            for (row, &value) in values.values().iter().enumerate() {
                if value > i64::MAX as u64 {
                    return Err(Snag::Value(format!(
                        "column '{name}' holds {value} at row {}, which is larger than the largest \
                         integer a column of a table holds",
                        base + row
                    )));
                }
                out.push(value);
            }
        }
        (DataType::Float32, Column::Float(out)) => {
            pushed!(array, Float32Type, out, changed, f64::from)
        }
        (DataType::Float64, Column::Float(out)) => {
            pushed!(array, Float64Type, out, changed, |f| f)
        }
        (DataType::Utf8, Column::Str(out)) => {
            let values = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(changed)?;
            out.extend(values.iter().map(|s| s.unwrap_or_default().to_string()));
        }
        (DataType::LargeUtf8, Column::Str(out)) => {
            let values = array
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .ok_or_else(changed)?;
            out.extend(values.iter().map(|s| s.unwrap_or_default().to_string()));
        }
        (DataType::Utf8View, Column::Str(out)) => {
            let values = array
                .as_any()
                .downcast_ref::<StringViewArray>()
                .ok_or_else(changed)?;
            out.extend(values.iter().map(|s| s.unwrap_or_default().to_string()));
        }
        (DataType::Date32, Column::Date(out)) => {
            pushed!(array, Date32Type, out, changed, |days| days)
        }
        (DataType::Time32(TimeUnit::Second), Column::LocalTime(out)) => {
            pushed!(array, Time32SecondType, out, changed, |n| i64::from(n)
                * 1_000_000_000)
        }
        (DataType::Time32(TimeUnit::Millisecond), Column::LocalTime(out)) => {
            pushed!(array, Time32MillisecondType, out, changed, |n| i64::from(n)
                * 1_000_000)
        }
        (DataType::Time64(TimeUnit::Microsecond), Column::LocalTime(out)) => {
            pushed!(array, Time64MicrosecondType, out, changed, |n| n * 1_000)
        }
        (DataType::Time64(TimeUnit::Nanosecond), Column::LocalTime(out)) => {
            pushed!(array, Time64NanosecondType, out, changed, |n| n)
        }
        (DataType::Timestamp(TimeUnit::Second, None), Column::LocalDatetime(out)) => {
            pushed!(array, TimestampSecondType, out, changed, |n| n
                * 1_000_000_000)
        }
        (DataType::Timestamp(TimeUnit::Millisecond, None), Column::LocalDatetime(out)) => {
            pushed!(array, TimestampMillisecondType, out, changed, |n| n
                * 1_000_000)
        }
        (DataType::Timestamp(TimeUnit::Microsecond, None), Column::LocalDatetime(out)) => {
            pushed!(array, TimestampMicrosecondType, out, changed, |n| n * 1_000)
        }
        (DataType::Timestamp(TimeUnit::Nanosecond, None), Column::LocalDatetime(out)) => {
            pushed!(array, TimestampNanosecondType, out, changed, |n| n)
        }
        (DataType::Duration(TimeUnit::Second), Column::Duration(DurationKind::DayTime, out)) => {
            pushed!(array, DurationSecondType, out, changed, |n| n
                * 1_000_000_000)
        }
        (
            DataType::Duration(TimeUnit::Millisecond),
            Column::Duration(DurationKind::DayTime, out),
        ) => {
            pushed!(array, DurationMillisecondType, out, changed, |n| n
                * 1_000_000)
        }
        (
            DataType::Duration(TimeUnit::Microsecond),
            Column::Duration(DurationKind::DayTime, out),
        ) => {
            pushed!(array, DurationMicrosecondType, out, changed, |n| n * 1_000)
        }
        (
            DataType::Duration(TimeUnit::Nanosecond),
            Column::Duration(DurationKind::DayTime, out),
        ) => {
            pushed!(array, DurationNanosecondType, out, changed, |n| n)
        }
        (
            DataType::Interval(IntervalUnit::YearMonth),
            Column::Duration(DurationKind::YearMonth, out),
        ) => {
            pushed!(array, IntervalYearMonthType, out, changed, i64::from)
        }
        _ => return Err(changed()),
    }
    Ok(())
}
