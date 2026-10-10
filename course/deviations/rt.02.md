## rt.02: C support and arena ownership

The former `rt.01` shared-library and language-loader lesson is retired under
the no-FFI course design. `abi.c` remains as standalone C support for status
codes, error messages, and allocator hooks, and is now owned by optional
`rt.02` alongside `arena.c`. Consumers declare `rt.02` as a C dependency and
test binaries link the needed C units directly. Python and Rust implementations
communicate with optional C exercises through fixture files or process
protocols.
