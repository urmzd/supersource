## data.07: tokenizer mutants target file-based Python loading

The generation-config precedence and byte-tokenizer hash mutants now target
`BPETokenizer` and `_Bytes` in the Python corpus tokenizer. Rust tokenizer
parity is covered separately through frozen fixtures and the standalone Rust
driver.
