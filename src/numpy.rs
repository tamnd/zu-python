//! A result as numpy arrays, one per column.
//!
//! The third way out of a result, beside rows and Arrow. It exists
//! because numpy is what a lot of code already holds: a model that
//! takes arrays, a plotting call, a loop somebody wrote before
//! DataFrames. Handing that code an Arrow table means it has to convert
//! one, and converting one means pyarrow has to be installed to do
//! work numpy could have done with the same bytes.
//!
//! The same bytes is what this is. `zudb::query::column` hands back one
//! owned buffer per column, values end to end in the layout every
//! columnar format uses, and numpy is a pointer, a length and a dtype
//! over exactly that. So an integer column, a float column, a datetime
//! column and a duration column all become arrays by moving the `Vec`
//! into numpy and naming the type: no pass over the values, no
//! allocation, and no second copy of the result in memory. The two that
//! cost something are the ones where the layouts differ rather than the
//! names: a boolean column is a bit per row here and a byte per row
//! there, and a date is 32 bits here and 64 bits there, so each takes
//! one widening pass.
//!
//! Nulls are the other half of the shape. numpy has no missing value
//! for an integer, so a column with one comes back as a
//! `numpy.ma.masked_array`, which is a data array and a boolean mask
//! beside it and is what a masked column has meant in numpy since long
//! before any of the alternatives. The mask is built from the validity
//! bitmap the engine already filled, and the data array underneath is
//! still the engine's own buffer: masking is a wrapper and not a copy.
//! Columns that become object arrays carry `None` in the cell instead,
//! because an object array has somewhere to put it.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use zudb::query::QueryResult;
use zudb::query::column::{Column, ColumnData, ColumnType, Offsets, Validity};

// The crate, not this module. Both are called numpy and a plain path
// would be ambiguous, which is the price of naming a module after the
// thing it is about.
use ::numpy::IntoPyArray;

use crate::value::{Names, to_py};

/// Every column of a result, as a dict of numpy arrays.
///
/// The order is the order the statement projected them, which a dict
/// keeps, so `list(result.fetchnumpy())` is `result.columns`.
pub fn arrays<'py>(
    py: Python<'py>,
    result: &QueryResult,
    names: &Names,
) -> PyResult<Bound<'py, PyDict>> {
    let ma = PyModule::import(py, "numpy.ma")?;
    // The read down the columns touches no Python object, and it is
    // two passes over every row of the result, so it happens with the
    // GIL down like every other pass this client makes over a result.
    let columns = py
        .detach(|| result.columnar())
        .map_err(|mixed| PyTypeError::new_err(mixed.to_string()))?;

    let out = PyDict::new(py);
    for held in columns.columns {
        let name = held.name;
        if out.contains(name)? {
            return Err(PyValueError::new_err(format!(
                "the result has two columns called '{name}', and a dict holds one of each name: \
                 give them different names with AS"
            )));
        }
        let array = column(py, &ma, held, names)?;
        out.set_item(name, array)?;
    }
    Ok(out)
}

/// One column, as the array and the mask that go with it.
fn column<'py>(
    py: Python<'py>,
    ma: &Bound<'py, PyModule>,
    held: Column<'_>,
    names: &Names,
) -> PyResult<Bound<'py, PyAny>> {
    let name = held.name;
    let len = held.len;
    let valid = held.validity;

    // The object arms answer for themselves: an object array has a cell
    // for `None`, so a mask beside it would say the same thing twice.
    let flat = match held.data {
        ColumnData::Null => return objects(py, (0..len).map(|_| py.None()).collect()),
        ColumnData::Str(strings) => {
            let mut out = Vec::with_capacity(len);
            let bytes = &strings.bytes;
            let mut spans = spans(&strings.offsets);
            for at in 0..len {
                let (from, upto) = spans.next().unwrap_or((0, 0));
                out.push(match missing(&valid, at) {
                    true => py.None(),
                    false => text(py, name, at, &bytes[from..upto])?,
                });
            }
            return objects(py, out);
        }
        ColumnData::Complex(values) => {
            let mut out = Vec::with_capacity(values.len());
            for value in values {
                out.push(to_py(py, value, names)?.unbind());
            }
            return objects(py, out);
        }

        // A bit per row here, a byte per row in numpy, so this one is
        // unpacked rather than moved.
        ColumnData::Bool { bits } => {
            let mut out = Vec::with_capacity(len);
            for at in 0..len {
                out.push(bits[at / 8] & (1u8 << (at % 8)) != 0);
            }
            out.into_pyarray(py).into_any()
        }
        ColumnData::Int(values) => values.into_pyarray(py).into_any(),
        ColumnData::Float(values) => values.into_pyarray(py).into_any(),
        // Days are 32 bits in the engine and 64 in a numpy
        // `datetime64`, which is the one widening pass here.
        ColumnData::Days(values) => {
            let wide: Vec<i64> = values.into_iter().map(i64::from).collect();
            seen(wide.into_pyarray(py).into_any(), "datetime64[D]")?
        }
        ColumnData::Months(values) => seen(values.into_pyarray(py).into_any(), "timedelta64[M]")?,
        ColumnData::Nanos(values) => {
            let array = values.into_pyarray(py).into_any();
            match held.ty {
                // A time of day is nanoseconds since midnight, which is
                // what numpy calls a `timedelta64`. It has no type for a
                // clock reading, and this is the reading itself rather
                // than a rounded stand-in for it.
                ColumnType::LocalTime => seen(array, "timedelta64[ns]")?,
                ColumnType::DayTime => seen(array, "timedelta64[ns]")?,
                ColumnType::LocalDatetime => seen(array, "datetime64[ns]")?,
                // The instant, in UTC. numpy has no zone to carry the
                // offset in, and the instant is what the buffer holds.
                ColumnType::ZonedDatetime { .. } => seen(array, "datetime64[ns]")?,
                ColumnType::ZonedTime { .. } => {
                    return Err(PyTypeError::new_err(format!(
                        "column '{name}' holds a time with an offset, which numpy has no type for"
                    )));
                }
                ty => return Err(mismatch(name, &ty)),
            }
        }
    };

    match valid {
        None => Ok(flat),
        Some(held) => masked(ma, flat, mask(&held, len).into_pyarray(py).into_any()),
    }
}

/// The bytes of one string, as a Python one.
///
/// The engine wrote the buffer and every string in it came in as a
/// Python string, so this cannot fail in practice; it is checked
/// anyway, because the alternative is a wrong answer rather than an
/// exception.
fn text<'py>(py: Python<'py>, name: &str, at: usize, bytes: &[u8]) -> PyResult<Py<PyAny>> {
    match std::str::from_utf8(bytes) {
        Ok(text) => Ok(pyo3::types::PyString::new(py, text).into_any().unbind()),
        Err(_) => Err(PyValueError::new_err(format!(
            "the string at row {at} of column '{name}' is not valid UTF-8"
        ))),
    }
}

/// Where each string sits in the bytes, whichever width the offsets are.
fn spans(offsets: &Offsets) -> Box<dyn Iterator<Item = (usize, usize)> + '_> {
    match offsets {
        Offsets::I32(held) => Box::new(
            held.windows(2)
                .map(|pair| (pair[0] as usize, pair[1] as usize)),
        ),
        Offsets::I64(held) => Box::new(
            held.windows(2)
                .map(|pair| (pair[0] as usize, pair[1] as usize)),
        ),
    }
}

/// Whether row `at` is null, when there is a bitmap to ask.
fn missing(valid: &Option<Validity>, at: usize) -> bool {
    valid.as_ref().is_some_and(|held| !held.is_valid(at))
}

/// The mask numpy wants, which is the bitmap the other way up: set
/// means missing there, and set means present in the engine's.
fn mask(valid: &Validity, len: usize) -> Vec<bool> {
    (0..len).map(|at| !valid.is_valid(at)).collect()
}

/// A list of Python objects as an object array.
fn objects<'py>(py: Python<'py>, values: Vec<Py<PyAny>>) -> PyResult<Bound<'py, PyAny>> {
    Ok(values.into_pyarray(py).into_any())
}

/// The same buffer read as another type of the same width.
///
/// `view` is numpy's own word for it and copies nothing: a
/// `datetime64[ns]` array and the `int64` array under it are the same
/// bytes, which is exactly the relationship the engine's buffer already
/// has with both of them.
fn seen<'py>(array: Bound<'py, PyAny>, dtype: &str) -> PyResult<Bound<'py, PyAny>> {
    array.call_method1("view", (dtype,))
}

/// The array and its mask, as the one object numpy has for the pair.
fn masked<'py>(
    ma: &Bound<'py, PyModule>,
    data: Bound<'py, PyAny>,
    mask: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let how = PyDict::new(ma.py());
    // Neither array is copied: the data is the engine's buffer and the
    // mask was built for this call, so there is nothing to protect
    // either of them from.
    how.set_item("copy", false)?;
    ma.call_method("masked_array", (data, mask), Some(&how))
}

/// A buffer holding something other than what the column's type says,
/// which is this module reading its own input wrong rather than
/// anything the caller did.
fn mismatch(name: &str, ty: &ColumnType) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!(
        "column '{name}' came back as {} in a buffer that does not hold one",
        ty.name()
    ))
}
