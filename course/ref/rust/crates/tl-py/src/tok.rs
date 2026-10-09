//! `tinyllm_rs.Bpe` (L1.5): `tl_tok::ByteBpe` for Python.
//!
//! `encode_batch` releases the GIL (`Python::detach`) while the Rust threads
//! run, so other Python threads keep going and the data pipeline (data.07)
//! tokenizes shards on every core.

use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;

use tl_tok::{ByteBpe, LoadError, Tokenizer};

/// A load error as the Python exception the contract names: `OSError` when
/// the file cannot be read, `ValueError` for everything else.
pub fn load_err(e: LoadError) -> PyErr {
    // SOLUTION-BEGIN L1.5
    match e {
        LoadError::Io { .. } => PyOSError::new_err(e.to_string()),
        other => PyValueError::new_err(other.to_string()),
    }
    // SOLUTION-END
}

#[pyclass(module = "tinyllm_rs", frozen)]
pub struct Bpe {
    inner: ByteBpe,
}

#[pymethods]
impl Bpe {
    #[staticmethod]
    fn from_hf_json(path: &str) -> PyResult<Bpe> {
        // SOLUTION-BEGIN L1.5
        ByteBpe::from_hf_json(std::path::Path::new(path)).map(|inner| Bpe { inner }).map_err(load_err)
        // SOLUTION-END
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        // SOLUTION-BEGIN L1.5
        self.inner.encode(text)
        // SOLUTION-END
    }

    fn encode_batch(&self, py: Python<'_>, texts: Vec<String>, threads: i64) -> PyResult<Vec<Vec<u32>>> {
        // SOLUTION-BEGIN L1.5
        if threads < 0 {
            return Err(PyValueError::new_err(format!("threads must be >= 0, got {threads}")));
        }
        let inner = &self.inner;
        Ok(py.detach(|| {
            let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
            inner.encode_batch(&refs, threads as usize)
        }))
        // SOLUTION-END
    }

    fn decode(&self, ids: Vec<i64>) -> PyResult<String> {
        // SOLUTION-BEGIN L1.5
        let n = self.inner.vocab_size() as i64;
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            if id < 0 || id >= n {
                return Err(PyValueError::new_err(format!("id {id} is outside the vocabulary (size {n})")));
            }
            out.push(id as u32);
        }
        self.inner.decode(&out).map_err(|e| PyValueError::new_err(e.to_string()))
        // SOLUTION-END
    }

    fn vocab_size(&self) -> u32 {
        // SOLUTION-BEGIN L1.5
        self.inner.vocab_size()
        // SOLUTION-END
    }
}
