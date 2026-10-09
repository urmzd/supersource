// Package testkit is the course's fault and determinism kit for Go tests
// (course/DESIGN.md 4.4, 5.9). Course tests and the learner's graded tests
// import its packages; nothing here is learner code.
//
//	clock       a fake Clock (Now, After, NewTimer, Sleep) driven by Advance
//	failpoint   named failpoints enabled by TL_FAILPOINTS="name=crash;other=error(msg)"
//	effects     an HTTP side-effect sink that records idempotency keys; AssertExactlyOnce
//	proc        KillLoop: start a learner binary, SIGKILL it at seeded offsets, restart, check
//	chaosproxy  a TCP proxy: latency, drop, reset mid-stream, half-open, bandwidth limit
//	otlpsink    an in-test OTLP/HTTP receiver (JSON and protobuf, gzip) with span-tree checks
//	promscrape  a Prometheus text-format scraper with metric and histogram lookups
//	faketool    a rule-based fake OpenAI-compatible provider keyed on the last message
package testkit
