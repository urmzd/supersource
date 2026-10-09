//! Links libtinyllm.a into every crate that depends on tl-sys.
//!
//! Where the library comes from, in order:
//!   1. $TINYLLM_C_LIB_DIR (the harness sets it to its own build of your C
//!      units, .ss/overlay/<ID>/build);
//!   2. c/build/ at the root of your repo (what `make -C c` produces).
//!
//! Without a library this script only warns: crates that never call into C
//! still build, and a binary that does call it fails to link, naming the
//! missing `tl_` symbol.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=TINYLLM_C_LIB_DIR");
    let dir = match env::var_os("TINYLLM_C_LIB_DIR") {
        Some(d) => PathBuf::from(d),
        None => PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("set by cargo")).join("../../../c/build"),
    };
    let lib = dir.join("libtinyllm.a");
    // Rerun when the library is rebuilt, so a changed C unit is relinked.
    println!("cargo:rerun-if-changed={}", lib.display());
    if lib.is_file() {
        println!("cargo:rustc-link-search=native={}", dir.display());
        println!("cargo:rustc-link-lib=static=tinyllm");
    } else {
        println!(
            "cargo:warning=tl-sys: {} not found; build the C library first (make -C c) or set TINYLLM_C_LIB_DIR",
            lib.display()
        );
    }
}
