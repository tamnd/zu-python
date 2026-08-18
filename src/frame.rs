//! A frame of columns, described where the caller keeps them.
//!
//! The way in for data that is already in columns. pandas, polars and
//! pyarrow all hand out an Arrow stream through the PyCapsule protocol,
//! and what comes back from one is buffers: eight-byte words back to
//! back, one bit a row for a boolean, characters end to end with
//! offsets cutting them up. That is how this engine lays a column out
//! too, so what this module produces is not a copy of any of it but a
//! description: where each column is, how wide its values are, and what
//! they mean.
//!
//! Two things do copy, and both are said rather than hidden. A stream
//! of several batches is concatenated into one, because a column of a
//! table is one run of bytes and two batches are two of them; that is a
//! memcpy per column and it happens once. A dictionary of Python lists
//! is read into buffers of this module's own, because a list holds
//! objects and a column holds numbers, so there is nothing there to
//! point at.
//!
//! What keeps the bytes alive is [`Held`], which the frame holds and
//! which the engine drops when the last table naming those bytes goes.
//! Dropping it releases the Arrow arrays back to whoever exported them,
//! and that release may reach into an interpreter, so it takes the GIL
//! first: the engine drops a frame on whichever thread finished with
//! it, and that is not always a thread of Python's.
//!
//! There is no null anywhere in it. A property that is null is one no
//! row of this engine can hold, so a column with a gap in it can only
//! ever be refused, and refusing it by name and row number is the
//! difference between a caller who knows which cell to fix and one who
//! knows only that something somewhere was empty.

use std::ffi::CStr;
use std::ptr::NonNull;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, LargeStringArray, StringArray};
use arrow::compute::{concat, concat_batches};
use arrow::datatypes::{DataType, IntervalUnit, TimeUnit};
use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow::record_batch::{RecordBatch, RecordBatchReader};
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict};
use zu_common::{DurationKind, FloatBits, IntBits, LogicalType};
use zudb::{Column, Layout};

use crate::buffer::{self, type_name};
use crate::columns::Snag;
use crate::load;

/// What a capsule holding an Arrow stream is called, which a consumer
/// checks before it reads the pointer.
const STREAM: &CStr = c"arrow_array_stream";

/// What a registered frame's bytes are, and the thing whose life is
/// their life.
///
/// One of the two fields is filled. The batch is the arrays a producer
/// handed over, held so the buffers underneath them stay where they
/// are; the owned vectors are what this client built out of Python
/// objects, for a caller with no frame library installed.
pub struct Held {
    /// An `Option` only so that [`Drop`] can take it, which is where
    /// the GIL has to be held.
    batch: Option<RecordBatch>,
    owned: Vec<Bytes>,
}

/// Releasing an imported Arrow array calls the callback the producer
/// gave with it, and a producer that is pyarrow drops Python objects in
/// that callback. The engine drops a frame when the last table naming
/// it goes, which may be inside a statement on a thread holding no GIL,
/// so this takes one. Attaching on a thread that already has it costs a
/// check.
impl Drop for Held {
    fn drop(&mut self) {
        if self.batch.is_some() {
            Python::attach(|_| drop(self.batch.take()));
        }
    }
}

/// One column this client built, because a list of Python objects is
/// not a column and something has to hold the bytes.
///
/// The variants are the layouts the engine reads rather than the types
/// a value has: a date and a count of days are one variant, and what
/// tells them apart is the logical type recorded beside the pointer.
enum Bytes {
    /// Signed 64-bit values, whatever they count.
    Counts(Vec<u64>),
    /// Signed 32-bit values, which is what a date is.
    Days(Vec<i32>),
    Floats(Vec<f64>),
    /// One bit a row, low bit of the first byte first.
    Bits(Vec<u8>),
    /// Arrow's `Utf8`: characters end to end and `rows + 1` offsets.
    Text {
        data: Vec<u8>,
        offsets: Vec<i32>,
    },
}

/// A frame as the engine is about to be told about it.
///
/// The columns hold raw pointers into what `held` keeps alive, which is
/// why the two travel together and why neither is any use without the
/// other.
pub struct Described {
    pub columns: Vec<Column>,
    pub rows: u64,
    held: Arc<Held>,
}

// A pointer is not `Send`, because Rust cannot know what it addresses.
// These address buffers the `Arc` in the same struct keeps alive, and
// that `Arc` is `Send` and `Sync`, so a description travels wherever
// the thing it describes does. Nothing writes through them.
unsafe impl Send for Described {}

impl Described {
    /// Registers this as a table named `name` on `engine`.
    ///
    /// Every pointer in it addresses a buffer the `Arc` handed over
    /// with them keeps alive, which is what building one of these
    /// promises and what nothing between there and here undoes, so the
    /// `unsafe` that [`zudb::Frame::new`] asks for is discharged where
    /// the buffers were described rather than here.
    pub fn register(self, engine: &mut zudb::Connection, name: &str) -> zudb::Result<()> {
        let Described {
            columns,
            rows,
            held,
        } = self;
        let frame = unsafe { zudb::Frame::new(name, rows, columns, held) }?;
        engine.register(frame)
    }
}

/// Reads whatever the caller handed over into a description of it.
///
/// Two shapes, and the first of them is the one that matters: anything
/// that speaks Arrow, which is every frame library worth naming, and a
/// dictionary of lists for a caller with none of them installed.
/// Anything else is refused here rather than iterated hopefully,
/// because the message a caller wants is the list of what would have
/// worked.
pub fn read(py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<Described> {
    if data.hasattr("__arrow_c_stream__")? {
        return from_arrow(py, data);
    }
    if let Ok(dict) = data.cast::<PyDict>() {
        return from_lists(dict);
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
/// nobody is holding is how a process ends. It goes down for everything
/// after that, which is pure Rust and, on the path that matters, walks
/// the pointers rather than the rows.
fn from_arrow(py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<Described> {
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
    let mut batches = Vec::new();
    for batch in reader {
        batches.push(batch.map_err(|err| Snag::Arrow(err).raise(py))?);
    }

    let batch = py
        .detach(|| -> Result<RecordBatch, Snag> {
            // One batch is the frame where it lies. Several are one
            // memcpy per column, because a column of a table is one run
            // of bytes and a table of two batches is two of them. None
            // is an empty frame, which the caller refuses under the
            // name it was asked to register.
            let batch = match batches.len() {
                1 => batches.pop().expect("the one batch"),
                _ => concat_batches(&schema, &batches)?,
            };
            settled(batch)
        })
        .map_err(|snag| snag.raise(py))?;

    let rows = batch.num_rows() as u64;
    let mut columns = Vec::with_capacity(batch.num_columns());
    for (at, array) in batch.columns().iter().enumerate() {
        let name = batch.schema_ref().field(at).name();
        let (ty, layout) = described(name, array)?;
        columns.push(Column {
            name: name.clone(),
            ty,
            layout,
        });
    }
    Ok(Described {
        columns,
        rows,
        held: Arc::new(Held {
            batch: Some(batch),
            owned: Vec::new(),
        }),
    })
}

/// A batch with everything about it that a pointer cannot express taken
/// out of it.
///
/// A producer may hand over a slice of a longer array, and a slice does
/// not start where its buffers do: a bitmap that begins partway into a
/// byte and offsets that begin partway into their data are both things
/// a bare pointer does not say. A skewed column is copied down to
/// itself, which costs the rows it actually holds and no more, and
/// every other column is left where it is.
///
/// Runs with the GIL down, so what is wrong comes back rather than
/// being raised.
fn settled(batch: RecordBatch) -> Result<RecordBatch, Snag> {
    let mut columns = batch.columns().to_vec();
    let mut skewed = false;
    for column in &mut columns {
        if offset(column) {
            // An empty slice of it goes in front, because `concat` of
            // one array hands that array straight back: the right
            // answer for a concatenation and the wrong one here, where
            // the copy is the whole point.
            let nothing = column.slice(0, 0);
            *column = concat(&[nothing.as_ref(), column.as_ref()])?;
            skewed = true;
        }
    }
    let batch = match skewed {
        true => RecordBatch::try_new(batch.schema(), columns)?,
        false => batch,
    };
    for (at, column) in batch.columns().iter().enumerate() {
        whole(batch.schema_ref().field(at).name(), column)?;
    }
    Ok(batch)
}

/// Whether this column starts somewhere other than where its buffers
/// do.
///
/// Two ways it can, because a slice reaches this by two roads. An array
/// may carry the row offset itself, which is what `offset` is; or the
/// producer may have handed the offset over already applied to the
/// buffers, which is what an Arrow import does to a string column, and
/// then the array counts from zero and its first offset does not.
fn offset(array: &ArrayRef) -> bool {
    if array.offset() != 0 {
        return true;
    }
    let starts_at = |from: Option<i64>| from.is_some_and(|from| from != 0);
    match array.data_type() {
        DataType::Utf8 => starts_at(
            array
                .as_any()
                .downcast_ref::<StringArray>()
                .and_then(|text| text.value_offsets().first().map(|&from| from as i64)),
        ),
        DataType::LargeUtf8 => starts_at(
            array
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .and_then(|text| text.value_offsets().first().copied()),
        ),
        _ => false,
    }
}

/// That a column has a value in every row of it.
///
/// Names the first row rather than the count, because a caller with a
/// gap in a column wants to go and look at it.
fn whole(name: &str, array: &ArrayRef) -> Result<(), Snag> {
    if array.null_count() == 0 {
        return Ok(());
    }
    let row = (0..array.len())
        .find(|&row| array.is_null(row))
        .unwrap_or(0);
    Err(Snag::Value(format!(
        "column '{name}' has no value at row {row}, and every column of a row holds one"
    )))
}

/// Reads a dictionary of Python lists into buffers of this module's
/// own.
///
/// The one path that copies, and it copies because there is nothing to
/// point at: a Python list holds objects and a column holds numbers.
/// What decides a column's type is its first value, which is
/// [`buffer::Column`]'s rule and the loader's, so a dictionary and a
/// load read the same way and refuse the same things.
fn from_lists(dict: &Bound<'_, PyDict>) -> PyResult<Described> {
    let built = load::build(Some(dict))?;
    let rows = built.first().map(|(_, column)| column.len()).unwrap_or(0) as u64;
    let mut named = Vec::with_capacity(built.len());
    let mut owned = Vec::with_capacity(built.len());
    for (name, column) in built {
        let (ty, bytes) = packed(&name, column)?;
        named.push((name, ty));
        owned.push(bytes);
    }
    let held = Arc::new(Held { batch: None, owned });
    let columns = named
        .into_iter()
        .zip(&held.owned)
        .map(|((name, ty), bytes)| Column {
            name,
            ty,
            layout: lent(bytes),
        })
        .collect();
    Ok(Described {
        columns,
        rows,
        held,
    })
}

/// One column of Python values, as the bytes a frame reads and what
/// they mean.
fn packed(name: &str, column: buffer::Column) -> PyResult<(LogicalType, Bytes)> {
    let counts = LogicalType::Int {
        signed: true,
        bits: IntBits::B64,
        precision: None,
    };
    let characters = LogicalType::Str {
        min: None,
        max: None,
        fixed: false,
    };
    // The temporal buffers count in `i64` and the integer one holds the
    // bits of one in a `u64`, and both of them are the eight-byte lane
    // the engine reads through, so this cast is the identity on the
    // bytes and the logical type beside them is what says how to read
    // them.
    let same = |v: Vec<i64>| Bytes::Counts(v.into_iter().map(|n| n as u64).collect());
    Ok(match column {
        buffer::Column::Int(v) => (counts, Bytes::Counts(v)),
        buffer::Column::Float(v) => (
            LogicalType::Float {
                bits: FloatBits::B64,
                precision: None,
            },
            Bytes::Floats(v),
        ),
        buffer::Column::Bool(v) => {
            // At least one byte, so the pointer is an allocation and
            // not the dangling address an empty vector lends out. A
            // frame of no rows never reaches a read, but it does reach
            // the check that the pointer is not null.
            let mut bits = vec![0u8; v.len().div_ceil(8).max(1)];
            for (row, &yes) in v.iter().enumerate() {
                if yes {
                    bits[row / 8] |= 1 << (row % 8);
                }
            }
            (LogicalType::Bool, Bytes::Bits(bits))
        }
        buffer::Column::Str(v) => {
            let mut data = Vec::with_capacity(v.iter().map(String::len).sum::<usize>().max(1));
            let mut offsets = Vec::with_capacity(v.len() + 1);
            offsets.push(0i32);
            for word in &v {
                data.extend_from_slice(word.as_bytes());
                let end = i32::try_from(data.len()).map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "column '{name}' holds more than two gigabytes of characters, which is \
                         further than the offsets of a frame reach"
                    ))
                })?;
                offsets.push(end);
            }
            (characters, Bytes::Text { data, offsets })
        }
        // Refused by the loader that built the column, so this arm is
        // here to be exhaustive rather than to be reached.
        buffer::Column::Bytes(_) => {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "column '{name}' holds byte strings, and no statement can read a column of bytes \
                 back yet"
            )));
        }
        buffer::Column::Date(v) => (LogicalType::Date, Bytes::Days(v)),
        buffer::Column::LocalTime(v) => (LogicalType::LocalTime, same(v)),
        buffer::Column::LocalDatetime(v) => (LogicalType::LocalDatetime, same(v)),
        buffer::Column::Duration(kind, v) => (LogicalType::Duration(kind), same(v)),
    })
}

/// Where one of this module's own buffers is, as a layout.
fn lent(bytes: &Bytes) -> Layout {
    match bytes {
        Bytes::Counts(v) => Layout::Int {
            ptr: at(v.as_ptr()),
            bits: IntBits::B64,
            signed: true,
            scale: 1,
        },
        Bytes::Days(v) => Layout::Int {
            ptr: at(v.as_ptr()),
            bits: IntBits::B32,
            signed: true,
            scale: 1,
        },
        Bytes::Floats(v) => Layout::Float {
            ptr: at(v.as_ptr()),
            bits: FloatBits::B64,
        },
        Bytes::Bits(v) => Layout::Bool {
            ptr: at(v.as_ptr()),
        },
        Bytes::Text { data, offsets } => Layout::Str {
            offsets: at(offsets.as_ptr()),
            wide: false,
            data: at(data.as_ptr()),
            data_len: data.len(),
        },
    }
}

/// A pointer to the start of something this process is holding.
///
/// Never null: a vector's pointer is an allocation or a dangling
/// aligned address, and an Arrow buffer's is an allocation. Neither of
/// them is address zero.
fn at<T>(ptr: *const T) -> NonNull<u8> {
    NonNull::new(ptr as *mut u8).expect("a buffer of this process is never at address zero")
}

/// Where one Arrow column is and what it means.
///
/// The buffers come off the array data rather than off a downcast per
/// type, because what is wanted is the same thing every time: the run
/// of bytes Arrow put the values in. The row offset was taken out of
/// the array before this, so buffer zero starts at row zero.
///
/// The scale is what one value is multiplied by to reach the unit its
/// meaning counts in, which is where Arrow's microseconds meet this
/// engine's nanoseconds. Nothing is converted here: the multiplication
/// happens per scanned chunk, on the rows a statement actually reads.
fn described(name: &str, array: &ArrayRef) -> PyResult<(LogicalType, Layout)> {
    let ty = array.data_type();
    let refused = |instead: &str| {
        pyo3::exceptions::PyTypeError::new_err(format!("column '{name}' is {ty}, and {instead}"))
    };
    let data = array.to_data();
    let bufs = data.buffers();
    let word = |bits: IntBits, signed: bool, scale: i64, means: LogicalType| {
        (
            means,
            Layout::Int {
                ptr: at(bufs[0].as_ptr()),
                bits,
                signed,
                scale,
            },
        )
    };
    let plain = |signed: bool, bits: IntBits| LogicalType::Int {
        signed,
        bits,
        precision: None,
    };
    let float = |bits: FloatBits| {
        (
            LogicalType::Float {
                bits,
                precision: None,
            },
            Layout::Float {
                ptr: at(bufs[0].as_ptr()),
                bits,
            },
        )
    };
    let characters = || LogicalType::Str {
        min: None,
        max: None,
        fixed: false,
    };
    let text = |wide: bool| {
        (
            characters(),
            Layout::Str {
                offsets: at(bufs[0].as_ptr()),
                wide,
                data: at(bufs[1].as_ptr()),
                data_len: bufs[1].len(),
            },
        )
    };
    Ok(match ty {
        DataType::Boolean => (
            LogicalType::Bool,
            Layout::Bool {
                ptr: at(bufs[0].as_ptr()),
            },
        ),
        DataType::Int8 => word(IntBits::B8, true, 1, plain(true, IntBits::B8)),
        DataType::Int16 => word(IntBits::B16, true, 1, plain(true, IntBits::B16)),
        DataType::Int32 => word(IntBits::B32, true, 1, plain(true, IntBits::B32)),
        DataType::Int64 => word(IntBits::B64, true, 1, plain(true, IntBits::B64)),
        DataType::UInt8 => word(IntBits::B8, false, 1, plain(false, IntBits::B8)),
        DataType::UInt16 => word(IntBits::B16, false, 1, plain(false, IntBits::B16)),
        DataType::UInt32 => word(IntBits::B32, false, 1, plain(false, IntBits::B32)),
        DataType::UInt64 => word(IntBits::B64, false, 1, plain(false, IntBits::B64)),
        DataType::Float32 => float(FloatBits::B32),
        DataType::Float64 => float(FloatBits::B64),
        // The three string layouts of Arrow, which are three ways of
        // holding the same characters: offsets into one buffer, wider
        // offsets into one buffer, and views into several. polars hands
        // out the third by default and pandas the first, and a column
        // of this table holds strings either way. A short view is
        // already this engine's own, byte for byte.
        DataType::Utf8 => text(false),
        DataType::LargeUtf8 => text(true),
        DataType::Utf8View => (
            characters(),
            Layout::View {
                views: at(bufs[0].as_ptr()),
                data: bufs[1..]
                    .iter()
                    .map(|buf| (at(buf.as_ptr()), buf.len()))
                    .collect(),
            },
        ),
        DataType::Date32 => word(IntBits::B32, true, 1, LogicalType::Date),
        DataType::Time32(TimeUnit::Second) => {
            word(IntBits::B32, true, 1_000_000_000, LogicalType::LocalTime)
        }
        DataType::Time32(TimeUnit::Millisecond) => {
            word(IntBits::B32, true, 1_000_000, LogicalType::LocalTime)
        }
        DataType::Time64(TimeUnit::Microsecond) => {
            word(IntBits::B64, true, 1_000, LogicalType::LocalTime)
        }
        DataType::Time64(TimeUnit::Nanosecond) => {
            word(IntBits::B64, true, 1, LogicalType::LocalTime)
        }
        DataType::Timestamp(unit, None) => {
            word(IntBits::B64, true, nanos(unit), LogicalType::LocalDatetime)
        }
        DataType::Timestamp(_, Some(zone)) => {
            return Err(refused(&format!(
                "a column of this table has nowhere to keep '{zone}', so drop the zone once the \
                 values are in the zone you want them in, or write it as a string"
            )));
        }
        DataType::Duration(unit) => word(
            IntBits::B64,
            true,
            nanos(unit),
            LogicalType::Duration(DurationKind::DayTime),
        ),
        DataType::Interval(IntervalUnit::YearMonth) => word(
            IntBits::B32,
            true,
            1,
            LogicalType::Duration(DurationKind::YearMonth),
        ),
        DataType::Dictionary(_, _) => {
            return Err(refused(
                "a dictionary is a layout rather than a type here, so cast it to what it holds \
                 first",
            ));
        }
        DataType::Binary | DataType::LargeBinary | DataType::BinaryView => {
            return Err(refused(
                "no statement can read a column of bytes back yet, so registering one would be \
                 naming data the caller cannot get at",
            ));
        }
        _ => {
            return Err(refused(
                "a column holds booleans, integers, floats, strings, dates, times, datetimes or \
                 durations",
            ));
        }
    })
}

/// How many nanoseconds one count of this unit is, which is the scale a
/// temporal column is read through.
fn nanos(unit: &TimeUnit) -> i64 {
    match unit {
        TimeUnit::Second => 1_000_000_000,
        TimeUnit::Millisecond => 1_000_000,
        TimeUnit::Microsecond => 1_000,
        TimeUnit::Nanosecond => 1,
    }
}
