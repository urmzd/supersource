# C 10: HTTP request parser

**Concepts:** state machine, buffer management, incremental input
**Difficulty:** ⭐⭐⭐⭐

An HTTP/1.1 request-header parser that accepts input in arbitrary chunks. This
is the exercise where "just split on `\r\n`" stops working, because the split
you were given may land in the middle of a header name.

## The contract

`http.h` declares it, `main.c` tests it, you write `http.c`.

| Function | Does |
|----------|------|
| `http_init(p)` | Reset for a new request |
| `http_feed(p, data, len, &consumed)` | Feed bytes; `INCOMPLETE`, `DONE`, or `ERROR` |
| `http_header(p, name)` | Case-insensitive lookup |

Fixed-size buffers throughout, no allocation. `consumed` must stop at the end of
the header block so the caller can find the body.

## What to notice

**Incremental parsing is the whole exercise.** A socket hands you whatever the
network felt like: 3 bytes, then 1400, then 12. There is no relationship between
those boundaries and the message's. So the parser cannot hold a pointer into the
caller's buffer across calls, and must carry enough state to resume mid-line.
`test_every_possible_split` feeds the same request at every one of its ~90 split
points and then one byte at a time, which is the test that fails for anything
built around `strtok` or a single pass over a complete buffer.

**Refuse over-long input, never truncate it.** Truncating a 3,000-byte path into
a 512-byte buffer does not produce an error, it produces a *different request*
that the application then trusts. That is a security bug, not a robustness one,
and the same reasoning applies to header names, values, and the header count.

**Strictness here is a security property, not pedantry.** Every rejected case in
`test_malformed_requests_are_rejected` is a real parser-differential class. When
a proxy and a backend disagree about whether `Foo : bar` is a header, or about
which of two `Content-Length` values wins, an attacker can hide a second request
inside the first. That is request smuggling, and the defence is that a parser
which is unsure must refuse rather than guess.

**`strtol` is the wrong tool for `Content-Length`.** It accepts `"12abc"`,
`"+5"`, leading whitespace, and negatives, and reports success for all of them.
A body length the parser and the server disagree about is precisely the
smuggling primitive above. Validate that the field is digits and nothing else.

**Accept bare LF while the spec says CRLF.** Robustness against real clients
requires it, and stripping at most one trailing `\r` costs one line. Note that
this is a place where being liberal is *safe*, because both readings produce the
same message. Contrast with `Foo : bar`, where being liberal changes what the
message means, and is therefore not safe. The distinction is the entire art of
the robustness principle, and the reason it has fallen out of favour.

**The error state has to be sticky.** Once a request is malformed, there is no
resynchronisation point: you cannot know where the next valid line starts, and
guessing lets an attacker choose. The connection is finished. The tests check
that feeding more bytes after an error still reports an error.

## Extending it

Add chunked transfer encoding, which is a second state machine layered on the
first and where a great many CVEs live, particularly around the interaction
between `Transfer-Encoding` and `Content-Length` when both are present. The
correct behaviour is to reject the request outright, and working out why is more
instructive than any amount of reading about it.
