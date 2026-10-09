//! tl-py: the Python module `tinyllm_rs` (contracts/py/tinyllm_rs.pyi, D8).
//!
//! Python reaches Rust only through this module. L1.5 owns this crate root
//! and `tok` (the byte-level BPE as `tinyllm_rs.Bpe`); ds.08 owns `bloom`
//! (`tinyllm_rs.Bloom`). Errors cross as Python exceptions (`ValueError`,
//! `OSError`), never as a panic: PyO3 would turn one into
//! `pyo3_runtime.PanicException`, which every course test treats as a failure.

use pyo3::prelude::*;

pub mod bloom;
pub mod tok;

/// The module object: one class per wrapped type.
#[pymodule]
fn tinyllm_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // SOLUTION-BEGIN L1.5
    m.add_class::<tok::Bpe>()?;
    m.add_class::<bloom::Bloom>()?;
    Ok(())
    // SOLUTION-END
}
