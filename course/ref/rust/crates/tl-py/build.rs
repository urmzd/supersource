// The Python interpreter provides the C API symbols an extension module
// calls, so on macOS the cdylib links with `-undefined dynamic_lookup`
// (what maturin would pass). Linux needs nothing: unresolved symbols are
// allowed in shared objects there.
fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-cdylib-link-arg=-undefined");
        println!("cargo:rustc-cdylib-link-arg=dynamic_lookup");
    }
}
