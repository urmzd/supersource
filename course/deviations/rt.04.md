## rt.04: local block-hash implementation

The paged KV pool previously linked `tl_fnv1a64` from C `M06.3`. The core
M06.3 module is now Python-only, so rt.04 implements the same FNV-1a 64-bit
byte update locally and uses it for the chained block hash. This keeps the
optional C pool self-contained while retaining M06.3 as reading for the hash
definition. The existing block-hash fixture tests verify the byte contract.
