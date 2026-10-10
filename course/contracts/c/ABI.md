# Standalone C module interfaces

The headers in `include/tinyllm/` define small C interfaces used by the
optional kernel and data-structure exercises. They are ordinary C headers
for independently compiled test binaries. Python and Rust do not load or
link these interfaces. Cross-language exchange uses files and process
protocols.

## Shared support

`tinyllm/abi.h` contains the shared status enum, error reporting functions,
and allocator hooks used by C units. These helpers are compiled into the
standalone C test process when needed. They do not define a dynamically
loaded library ABI, and the course does not promise binary compatibility
between releases.

## Module headers

Each exercise owns its contract header next to the corresponding C unit:

| Header | Purpose |
| --- | --- |
| `tinyllm/matmul.h` | Matrix multiplication kernels |
| `tinyllm/softmax.h` | Stable row-wise normalization |
| `tinyllm/attention.h` | Attention kernels |
| `tinyllm/kv_pool.h` | Paged KV block storage |
| `tinyllm/numerics.h` | Low precision and scalar numerical routines |

The header documents argument sizes, strides, ownership, and status behavior.
Tests include the module header directly and link only the C units under
test plus any shared C support those units use.

## Build and safety

`c/Makefile` may produce a static archive for local C experiments. The
verified course path builds standalone test executables and exchanges results
with other language implementations through files and process protocols. Use
`SANITIZE=1` for AddressSanitizer and UndefinedBehaviorSanitizer builds. Threaded
pool tests may additionally run under ThreadSanitizer.
