//! What a notebook shows.
//!
//! Jupyter asks an object for `_repr_html_` before it falls back to
//! `repr`, so a result printed in a cell can be a table instead of a
//! line saying how many rows it has. That is most of what a person
//! does with this library in a notebook, and a line of text is a
//! strictly worse answer than the rows themselves.
//!
//! Nothing here is a dependency on anything. The markup is a table, a
//! stylesheet and no script, so it survives `nbconvert`, an exported
//! HTML file and a notebook diff, and there is nothing to install for
//! it to work.
//!
//! Colours are the notebook's own, through `currentColor` and opacity,
//! because a light theme and a dark one are both in the room and a
//! guess about which would be wrong half the time.

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt};

use crate::value::{Node, Path, Rel};

/// Rows a table shows before it stops and says how many there were.
///
/// A result is already in memory, so this is not about the cost of
/// reading it: a hundred rows is more than a person reads and about a
/// screen and a half of scrolling, and a million rows of markup is a
/// notebook file nobody can open.
pub const SHOWN: usize = 100;

/// The longest a cell is shown at, in characters.
///
/// A column holding a document would otherwise be a page holding a
/// document, and the rest of the row would be somewhere below it.
pub const WIDEST: usize = 200;

/// The stylesheet, written beside every table.
///
/// Classes rather than an attribute on every cell, because a hundred
/// rows of ten columns is a thousand cells and the style repeated on
/// each of them would be most of what the notebook stores. Repeating
/// the block itself costs a few hundred bytes per result and defines
/// exactly the same rules every time.
const STYLE: &str = "<style>\
.zu-wrap{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;line-height:1.5}\
.zu-table{border-collapse:collapse}\
.zu-table th{text-align:left;font-weight:600;padding:2px 12px 2px 0;\
border-bottom:1px solid currentColor;opacity:.75;white-space:pre}\
.zu-table td{padding:2px 12px 2px 0;vertical-align:top;white-space:pre}\
.zu-num{text-align:right;font-variant-numeric:tabular-nums}\
.zu-null{opacity:.45;font-style:italic}\
.zu-note{opacity:.6;padding-top:4px}\
.zu-at{opacity:.6}\
</style>";

/// The markup for one thing, with the stylesheet and the wrapper that
/// the classes hang off.
pub fn wrap(inner: &str) -> String {
    format!("{STYLE}<div class=\"zu-wrap\">{inner}</div>")
}

/// Text that cannot be markup.
///
/// A string column holding `<script>` is a string column, and a client
/// that pasted it into a page unescaped would be a client that runs a
/// caller's data as code in their notebook.
pub fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for letter in text.chars() {
        match letter {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(letter),
        }
    }
    out
}

/// Text cut to `WIDEST` characters, with a mark where it was cut.
///
/// Characters and not bytes, so a column of Japanese is cut where a
/// column of English is and nothing is cut through the middle of a
/// letter.
fn clipped(text: &str) -> String {
    let mut out = String::new();
    for (seen, letter) in text.chars().enumerate() {
        if seen == WIDEST {
            out.push('…');
            break;
        }
        out.push(letter);
    }
    out
}

/// A count with its thousands marked, because six digits are read by
/// counting them and `100,000` is read by looking at it.
pub fn counted(number: usize) -> String {
    let digits = number.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (seen, digit) in digits.chars().enumerate() {
        if seen > 0 && (digits.len() - seen).is_multiple_of(3) {
            out.push(',');
        }
        out.push(digit);
    }
    out
}

/// `1 row` and `2 rows`, which is worth the four lines it takes.
fn many(count: usize, one: &str, more: &str) -> String {
    format!("{} {}", counted(count), if count == 1 { one } else { more })
}

/// What a table says underneath itself.
pub fn note(rows: usize, columns: usize) -> String {
    let mut text = format!(
        "{}, {}",
        many(rows, "row", "rows"),
        many(columns, "column", "columns")
    );
    if rows > SHOWN {
        text.push_str(&format!(", first {} shown", counted(SHOWN)));
    }
    text
}

/// One node, as the pair that names it.
pub fn node(node: &Node) -> String {
    format!(
        "<span class=\"zu-node\">({} <span class=\"zu-at\">#{}</span>)</span>",
        escape(&node.table),
        node.offset
    )
}

/// One edge on its own, which has to say what it joins because there
/// is nothing beside it saying so.
pub fn rel(rel: &Rel) -> String {
    format!(
        "<span class=\"zu-rel\"><span class=\"zu-at\">#{}</span> -[{}]-&gt; \
         <span class=\"zu-at\">#{}</span></span>",
        rel.src,
        escape(&rel.table),
        rel.dst
    )
}

/// One edge inside a walk, where the nodes on either side already say
/// which rows it joins and repeating them would be noise.
fn hop(rel: &Rel) -> String {
    format!(
        "<span class=\"zu-rel\">-[{}]-&gt;</span>",
        escape(&rel.table)
    )
}

/// A walk, drawn the way a statement writes one.
pub fn path(py: Python<'_>, path: &Path) -> PyResult<String> {
    let mut out = String::from("<span class=\"zu-path\">");
    for (place, element) in path.elements.bind(py).iter().enumerate() {
        if place > 0 {
            out.push(' ');
        }
        out.push_str(&piece(&element)?);
    }
    out.push_str("</span>");
    Ok(out)
}

/// One element of a walk, or anything else that turned up in a list
/// that is supposed to hold nodes and edges.
fn piece(element: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(one) = element.cast::<Node>() {
        return Ok(node(&one.borrow()));
    }
    if let Ok(one) = element.cast::<Rel>() {
        return Ok(hop(&one.borrow()));
    }
    Ok(escape(&clipped(&element.str()?.to_string_lossy())))
}

/// One cell of a table, tag and all.
///
/// Values are shown the way `str` shows them rather than the way
/// `repr` does, so a string is its text and not its quotes, and the
/// three graph classes are drawn instead. Numbers line up on the
/// right, which is the one piece of formatting a table of numbers
/// cannot do without, and booleans are left alone there: Python says a
/// `bool` is an `int` and a column of `True` is not a column to read
/// digit by digit.
pub fn cell(object: &Bound<'_, PyAny>) -> PyResult<String> {
    if object.is_none() {
        return Ok("<td class=\"zu-null\">null</td>".to_string());
    }
    if let Ok(one) = object.cast::<Node>() {
        return Ok(format!("<td>{}</td>", node(&one.borrow())));
    }
    if let Ok(one) = object.cast::<Rel>() {
        return Ok(format!("<td>{}</td>", rel(&one.borrow())));
    }
    if let Ok(one) = object.cast::<Path>() {
        return Ok(format!("<td>{}</td>", path(object.py(), &one.borrow())?));
    }
    let text = escape(&clipped(&object.str()?.to_string_lossy()));
    let counts = (object.is_instance_of::<PyInt>() || object.is_instance_of::<PyFloat>())
        && !object.is_instance_of::<PyBool>();
    if counts {
        Ok(format!("<td class=\"zu-num\">{text}</td>"))
    } else {
        Ok(format!("<td>{text}</td>"))
    }
}
