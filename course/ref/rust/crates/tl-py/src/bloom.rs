//! `tinyllm_rs.Bloom` (ds.08): `tl_ds::bloom::Bloom` for Python, byte for
//! byte the same filter (formats/bloom.md), so the corpus pipeline (data.03)
//! can build a screen in one process and load it in another.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use tl_ds::bloom::{Bloom as Inner, BloomError};

fn value_err(e: BloomError) -> PyErr {
    // SOLUTION-BEGIN ds.08
    PyValueError::new_err(e.to_string())
    // SOLUTION-END
}

#[pyclass(module = "tinyllm_rs")]
pub struct Bloom {
    inner: Inner,
}

#[pymethods]
impl Bloom {
    /// `n` arrives as a Python int of any size: a negative or zero `n` is a
    /// ValueError, not an OverflowError.
    #[staticmethod]
    fn with_rate(n: i64, p: f64) -> PyResult<Bloom> {
        // SOLUTION-BEGIN ds.08
        if n < 1 {
            return Err(PyValueError::new_err(format!("n must be >= 1, got {n}")));
        }
        Inner::with_rate(n as u64, p).map(|inner| Bloom { inner }).map_err(value_err)
        // SOLUTION-END
    }

    fn insert(&mut self, item: &[u8]) {
        // SOLUTION-BEGIN ds.08
        self.inner.insert(item)
        // SOLUTION-END
    }

    fn contains(&self, item: &[u8]) -> bool {
        // SOLUTION-BEGIN ds.08
        self.inner.contains(item)
        // SOLUTION-END
    }

    /// `other` may be this very filter (`b.union(b)`): its bits are read
    /// before any are written.
    fn union(slf: &Bound<'_, Self>, other: &Bound<'_, Self>) -> PyResult<()> {
        // SOLUTION-BEGIN ds.08
        let theirs = other.borrow().inner.clone();
        slf.borrow_mut().inner.union(&theirs).map_err(value_err)
        // SOLUTION-END
    }

    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        // SOLUTION-BEGIN ds.08
        PyBytes::new(py, &self.inner.to_bytes())
        // SOLUTION-END
    }

    #[staticmethod]
    fn from_bytes(b: &[u8]) -> PyResult<Bloom> {
        // SOLUTION-BEGIN ds.08
        Inner::from_bytes(b).map(|inner| Bloom { inner }).map_err(value_err)
        // SOLUTION-END
    }
}
