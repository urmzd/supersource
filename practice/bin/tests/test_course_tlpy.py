"""The Rust farm's tl-py build (DESIGN 2.5, 5.4): a cdylib built in the shared
farm with PYO3_PYTHON and the macOS link args, copied to tinyllm_rs.so in
TINYLLM_PYEXT_DIR, which goes first on PYTHONPATH; the contract pre-check
enforces the build contract. The fixture extension uses the CPython C API
directly (a local `pyo3` stand-in carries the manifest features), so the
build runs offline."""

from pathlib import Path


def test_python_imports_the_learners_rust_extension(ss):
    ss.add_extras("tlpy")
    ss.init()
    ss("start", "ds.90", rc=0)
    for u in ("rust/crates/tl-demo/src/lib.rs", "rust/crates/tl-demo/src/acc.rs"):
        ss.implement(u)
    ss("check", "ds.90", rc=0, timeout=600)
    out = ss("start", "ds.92", rc=0).out
    assert "wrote     rust/crates/tl-py/Cargo.toml" in out
    unit = ss.learner / "rust/crates/tl-py/src/lib.rs"
    assert 'todo!("ds.92")' in unit.read_text()
    out = ss(
        "check", "ds.92", rc=1, timeout=600
    ).out  # the stub panics inside the extension
    assert "FAIL" in out
    ss.implement("rust/crates/tl-py/src/lib.rs")
    out = ss("check", "ds.92", rc=0, timeout=600).out
    assert "PASS ds.92" in out
    ext = ss.learner / ".ss/overlay/ds.92/pyext/tinyllm_rs.so"
    assert ext.is_file() and ext.stat().st_size > 0

    # The build contract: the pyo3 features and the macOS link args.
    man = ss.learner / "rust/crates/tl-py/Cargo.toml"
    man.write_text(man.read_text().replace(', "extension-module"', ""))
    out = ss("check", "ds.92", rc=4).out
    assert "the pyo3 dependency needs feature 'extension-module'" in out
    man.write_text(
        man.read_text().replace('"abi3-py311"]', '"abi3-py311", "extension-module"]')
    )
    cfg = ss.learner / "rust/.cargo/config.toml"
    cfg.unlink()
    out = ss("check", "ds.92", rc=4).out
    assert "dynamic_lookup" in out
    assert Path(ext).is_file()
