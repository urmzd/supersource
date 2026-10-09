# Durable WAL: the segmented event log

<!-- modules: dur.01 (owns it: go/durable/log), dur.02 and dur.03 (write histories and queue transitions), dur.10 (Raft replicates its records), ops.02 and ops.11 (drills)
     conformance: durable/ -->

The durable server keeps every history event and every queue transition in one append-only log, so a SIGKILL at any instant loses nothing that was acknowledged. Directory: `[durable].wal_dir` (`/var/lib/durable/wal` on its own PVC in Kubernetes). All integers little-endian.

## Segments

Files `<first_seq>.log`, where `first_seq` is the sequence number of the segment's first record as 20 decimal digits (`00000000000000000001.log` is the first segment). A segment holds whole records only; the writer starts a new segment before a record that would take the current one past 64 MiB. Segments are never rewritten, only appended to (the last one) or, on recovery, truncated (the last one).

## Record

```
offset  size  field
0       4     u32 len       payload bytes
4       4     u32 crc32c    CRC-32C of the 8 seq bytes followed by the payload
8       8     u64 seq       1 for the first record ever, then +1 per record, with no gap across segments
16      len   payload       a tl.durable.v1.WalRecord (proto/tl/durable/v1/durable.proto), protobuf binary
```

CRC-32C is the Castagnoli CRC of [kv-block.md](kv-block.md). `WalRecord` carries the `workflow_id`, the `run_id`, and either one `HistoryEvent` or one `TaskTransition` (lease, retry, dead letter, redrive, ack). Replaying every record in `seq` order rebuilds every history and every queue exactly; that is the whole recovery procedure.

## Durability and acknowledgement

An rpc that appends (StartWorkflow, CompleteWorkflowTask, CompleteActivityTask, ...) answers only after its records are written **and** `fsync`ed. Group commit is allowed: several rpcs may share one `fsync`. An append that fails (I/O error, quota) leaves the log as it was and the rpc fails; nothing acknowledged is ever changed.

**Quota.** The total size of all segments is capped by `[durable].wal_max_bytes` (`TL_DURABLE__WAL_MAX_BYTES`). An append that would exceed it fails with gRPC `RESOURCE_EXHAUSTED` and appends nothing; reads keep working; raising the cap and restarting resumes writes (drill ops.11).

## Recovery

On start the server scans every segment in order:

1. In every segment but the last, a short header, a `len` past the end of the file, a CRC mismatch, or a `seq` that is not the previous plus one is **corruption**: the server refuses to start and names the segment and offset. It never guesses.
2. In the last segment, the first record that fails any of those checks is a **torn tail** (a crash mid-append, never acknowledged): the file is truncated at that record's offset, and the next append gets `seq` = last valid + 1.

The failpoint `dur/log/after-write-before-fsync` (testkit) crashes between the write and the `fsync`, which is exactly the torn tail case.

## Raft (dur.10, optional)

With Raft on, each `Entry.data` of [`tl.raft.v1`](../proto/tl/raft/v1/raft.proto) is one complete framed record (header and payload). A follower appends the bytes as received after checking the CRC; the leader acknowledges an rpc once the entry is committed on a majority.

## Worked example

`WalRecord{workflow_id: "w", run_id: "r", event: HistoryEvent{event_id: 1, started: {workflow_type: "Echo"}}}` encodes to 18 payload bytes:

```
0a 01 77                field 1 (workflow_id), length 1, "w"
12 01 72                field 2 (run_id), length 1, "r"
1a 0a                   field 3 (event), length 10:
   08 01                  field 1 (event_id) = 1
   52 06                  field 10 (started), length 6:
      0a 04 45 63 68 6f      field 1 (workflow_type), "Echo"
```

As record `seq = 1` it is 34 bytes:

```
12 00 00 00             len 18
68 1a dc e6             crc32c 0xE6DC1A68 over the seq bytes and the payload
01 00 00 00 00 00 00 00  seq 1
0a 01 77 12 01 72 1a 0a 08 01 52 06 0a 04 45 63 68 6f
```
