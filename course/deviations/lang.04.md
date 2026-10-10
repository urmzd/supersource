## lang.04: Rust byte lengths without FFI

The original primer called C `strlen` from Rust with `extern "C"` and
`CString`. The no-FFI decision removes that call. `LEN` now teaches the
difference between UTF-8 byte length (`str.len()`) and Unicode scalar count
(`chars().count()`); NUL is valid Rust string content and counts as one byte.
The TCP echo server and Cargo workspace exercise remain. Forward references
now describe the candle engine and file-based tokenizer parity.
