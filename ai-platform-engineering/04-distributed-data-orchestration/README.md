# Distributed Data & Orchestration

## Overview

- **Primary references**: [Temporal docs](https://docs.temporal.io/) (free), [DBOS docs](https://docs.dbos.dev/) (free), [Apache Cassandra docs](https://cassandra.apache.org/doc/latest/) (free), DDIA Ch 5-9 (replication, partitioning, transactions, consistency)
- **Supplementary**: [Apache Cadence](https://cadenceworkflow.io/docs/) (free), [Apache AGE](https://age.apache.org/) (graph extension for Postgres, free), [pgvector](https://github.com/pgvector/pgvector), [Neo4j docs](https://neo4j.com/docs/), [Citus](https://www.citusdata.com/) (Postgres sharding), [Vitess](https://vitess.io/) (MySQL sharding)
- **Prerequisites**: [Data Engineering](../../data-engineering/) (OLTP vs OLAP, storage engines), [System Design](../../systems/01-system-design/) (replication, CAP, consistency), [RPC & Protocols](../02-rpc-and-protocols/) (idempotency, partial failure)
- **Estimated time**: 3-4 weeks at 8-10 hrs/week

## Key Takeaways

- **OLTP and OLAP are different machines.** Transactional systems are row-oriented, index-heavy, latency-sensitive (the app database); analytical systems are column-oriented, scan-heavy, throughput-sensitive (the warehouse). Storage layout follows access pattern.
- **Sharding is horizontal partitioning by a key.** When one node can't hold the data or the load, split rows across nodes by a partition key. The hard parts are choosing the key (avoid hot shards), routing, rebalancing, and cross-shard queries.
- **Durable execution makes workflows crash-proof.** Temporal, Cadence, and DBOS persist every step of a long-running workflow so that, after any crash, it resumes exactly where it left off — replacing hand-rolled retry/cron/state-machine glue.
- **The engine encodes the question.** Cassandra for write-heavy, partition-keyed, highly-available data; Postgres for transactions (and graphs via AGE, vectors via pgvector); a knowledge graph when relationships *are* the data.
- **Temporal data is a modeling discipline.** "What did we know, and when?" needs bitemporal thinking (valid time vs transaction time) — distinct from the Temporal *workflow* engine that happens to share the name.

## How to Study

- Shard a table by a key in Citus or Vitess; write a query that hits one shard and one that fans out to all of them — feel the difference.
- Build a three-step pipeline (download → embed → index) as a Temporal/DBOS workflow; kill the worker mid-run and watch it resume without redoing completed steps.
- Model the same domain three ways: relational (Postgres), wide-column (Cassandra), and graph (Apache AGE / Cypher). Run the queries each is good at.
- Take an OLTP schema and design its OLAP counterpart (star schema) — see why you don't run analytics on the production database.

---

# Concepts & Techniques

## Core Insight

At scale, data outgrows a single box along three independent axes: it gets too *big* (partition it — sharding), too *important to lose* (replicate it), and too *interconnected to query naively* (model it — relational, wide-column, or graph). Meanwhile the *processes* over that data — train a model, run a RAG pipeline, fulfill an order — are multi-step and long-running, so they outgrow a single request and need to survive crashes (durable orchestration). The platform engineer matches each kind of data to an engine whose physical layout fits the query, and each long-running process to an execution model that persists its progress.

## 1. OLTP vs OLAP

**The foundational storage dichotomy**

| | OLTP | OLAP |
|---|------|------|
| Workload | Many small reads/writes | Few huge scans/aggregations |
| Layout | **Row-oriented** | **Column-oriented** |
| Optimized for | Point lookups, updates, transactions | Bulk scans, group-by, joins |
| Latency | Milliseconds, per-request | Seconds-minutes, per-query |
| Examples | Postgres, MySQL, Cassandra | Snowflake, BigQuery, ClickHouse, DuckDB |
| Freshness | Live, authoritative | Loaded via ELT, slightly stale |

**Key ideas**:
- **Why columnar wins analytics**: an aggregation touches a few columns over millions of rows; storing each column contiguously means you read only what you need, and it compresses brilliantly (similar values adjacent) and vectorizes.
- **Why row wins transactions**: an app reads/writes a whole entity at once; keeping a row's fields together makes that one disk/page touch.
- **Don't run analytics on the OLTP database**: heavy scans contend with live transactions and lock the app. You **ELT** into a warehouse (see [Data Engineering](../../data-engineering/)). HTAP systems try to do both; mostly you separate them.

## 2. Sharding (Horizontal Partitioning)

**Splitting data across nodes by a key**

**Key ideas**:
- **What and why**: partition rows across N nodes so no single node holds all the data or serves all the load. Scales writes and storage beyond one machine (replication scales reads/availability — they're orthogonal).
- **Partition strategies**:
  - **Hash sharding**: `hash(key) → shard`. Even distribution, but kills range scans.
  - **Range sharding**: contiguous key ranges per shard. Great for range queries, risks hot ranges.
  - **Consistent hashing**: place nodes and keys on a ring so adding/removing a node moves minimal data — the basis of Cassandra/Dynamo.
- **The hard parts**:
  - **Hot shards / skew**: a bad key (e.g. `country` when 90% are one country, or a celebrity user) overloads one shard. Choose a high-cardinality, evenly-distributed key; sometimes add a salt.
  - **Cross-shard queries & joins**: a query spanning shards must scatter-gather and merge — slow and complex. Co-locate data that's queried together.
  - **Rebalancing**: adding nodes must move data without downtime (virtual nodes / hash slots make this incremental).
  - **Distributed transactions** across shards need 2PC/saga — expensive; avoid spanning shards in one transaction when you can.
- **Tooling**: Citus (Postgres), Vitess (MySQL), and native sharding in Cassandra/MongoDB/CockroachDB/Spanner.

## 3. Replication & Consistency (Brief)

**Key ideas**:
- **Replication** = copies of the same partition on multiple nodes, for availability and read-scaling. Leader-follower (Postgres) vs leaderless quorum (Cassandra/Dynamo: read R + write W replicas, tune for consistency vs availability).
- **CAP / PACELC**: under a partition you choose consistency *or* availability; even without one, you trade latency vs consistency. Cassandra leans AP (tunable); Spanner/CockroachDB lean CP.
- **Quorum**: `R + W > N` gives strong-ish consistency across replicas; lower it for speed. (Depth in [System Design](../../systems/01-system-design/) and DDIA.)

## 4. Apache Cassandra

**Wide-column, write-optimized, highly available**

**Key ideas**:
- **Architecture**: masterless ring, consistent hashing, every node equal — no single point of failure, linear horizontal scaling, multi-datacenter replication.
- **Storage engine**: **LSM-tree** (memtable → SSTables → compaction) — write-optimized, sequential writes, great ingest throughput. (Same engine family discussed in [Data Engineering storage](../../data-engineering/02-storage-warehousing/).)
- **Data model**: **query-first** — you design tables *around the queries* you'll run, because there are no ad-hoc joins. The **partition key** decides which node holds the row (and is your sharding key); clustering keys order within a partition.
- **Tunable consistency**: per-query `ONE`/`QUORUM`/`ALL` lets you trade latency for consistency. AP by default.
- **Use it for**: time-series, event logs, feature stores, write-heavy workloads, anything needing always-on multi-region writes. **Don't** use it when you need ad-hoc analytical queries or multi-row transactions.
- **Relatives**: ScyllaDB (C++ rewrite), DynamoDB (managed, same lineage), HBase/Bigtable (wide-column cousins).

## 5. Durable Orchestration & Execution

**Long-running, crash-proof workflows**

**The problem**: a real process — ingest a document, chunk it, embed it, index it, notify the user; or run a multi-day training job with checkpoints — spans minutes to days, calls flaky services, and *must not* lose its place when a worker crashes, deploys, or scales. Hand-rolling this with cron + a state table + retry logic is where bugs live.

**Durable execution** persists the *progress* of a workflow so it can resume exactly where it stopped:

- **Temporal** (and its predecessor **Cadence**, both originally from Uber): you write workflow code as ordinary functions; the engine records every step's result in an **event-sourced history** and **replays** that history to reconstruct state after a crash. **Activities** (the side-effecting steps) get automatic retries, timeouts, and heartbeats. Workflows can run for months, sleep, wait for signals, and survive any process restart. Cadence is the open-source ancestor; Temporal is the actively-developed fork.
- **DBOS**: durable execution backed by **Postgres** — workflow and step state are written transactionally to the database, so recovery is "read the last committed step and continue." Lighter-weight, library-first, leans on the database you already run rather than a separate cluster.
- **The shared idea — durable execution / the saga pattern**: decompose a long process into idempotent steps whose outcomes are persisted; on failure, resume from the last completed step (and run **compensations** to undo partial work when you can't go forward). This is the reliable backbone for AI pipelines, agent loops, and data workflows.
- **Where the storage comes in**: the engine needs a durable store for history — Temporal/Cadence use Cassandra (or Postgres/MySQL) underneath; DBOS uses Postgres directly. The orchestration layer is itself a sharded, replicated database problem.
- **vs workflow schedulers**: Airflow/Dagster (from [Data Engineering](../../data-engineering/04-orchestration-modeling/)) orchestrate *batch DAGs on a schedule*; Temporal/DBOS provide *durable execution for arbitrary application code* with fine-grained state and long-lived, event-driven workflows. Overlapping but distinct tools.

## 6. Knowledge Graphs & Graph Databases

**When the relationships *are* the data**

**Key ideas**:
- **Graph model**: nodes (entities) + edges (typed relationships) + properties. Traversals ("friends-of-friends who bought X", "what depends on this service") that would be many self-joins in SQL are first-class, index-free adjacency walks.
- **Knowledge graph**: a graph of entities and facts (often triples: subject-predicate-object, RDF/SPARQL) encoding a domain's semantics — powers entity resolution, recommendations, and **GraphRAG** (retrieve a subgraph of related facts instead of flat text chunks, for grounded multi-hop answers).
- **Engines**:
  - **Neo4j**: the dominant native graph database; **Cypher** query language; tuned for deep traversals.
  - **Apache AGE** (*A Graph Extension*): adds **graph capabilities to Postgres** — run Cypher inside Postgres, alongside relational tables (and pgvector embeddings). Lets one database serve relational, vector, *and* graph queries — attractive for an AI platform that wants graph + RAG without a separate system.
  - **RDF / triple stores** (Blazegraph, GraphDB) for formal ontologies and SPARQL.
- **Why it matters for AI platforms**: GraphRAG and entity-centric retrieval combine a knowledge graph with embeddings — the graph gives structure and multi-hop reasoning, the vectors give fuzzy semantic recall.

## 7. Postgres as the AI-Platform Swiss-Army Database

**Key ideas**:
- One reason Postgres dominates platform stacks: extensions turn it into many databases — **pgvector** (embeddings + ANN for RAG), **Apache AGE** (graphs/Cypher), **Citus** (sharding/distributed), **TimescaleDB** (time-series), plus rock-solid OLTP transactions and **DBOS** durable execution on top.
- The pragmatic default: start with Postgres + extensions; reach for Cassandra (write-scale/HA), a dedicated vector DB (Qdrant/Milvus at huge scale), or Neo4j (deep graph) only when Postgres's limits actually bite.

## 8. Temporal Data Modeling

**Modeling time itself — distinct from the Temporal engine**

**Key ideas**:
- **Bitemporal data**: track two independent time axes — **valid time** (when a fact was true in the real world) and **transaction time** (when the system recorded it). Lets you answer "what did we *believe* the price was on March 1, as of what we knew on Feb 28?"
- **Slowly Changing Dimensions (SCD Type 2)**: keep history by adding rows with validity ranges rather than overwriting — the warehouse pattern (see [dimensional modeling](../../data-engineering/04-orchestration-modeling/)).
- **Event sourcing**: store the immutable log of *events*; current state is a fold over the log. This is exactly how Temporal/Cadence reconstruct workflow state — and how you get audit, replay, and time-travel for free.
- **Why platforms care**: reproducibility (which data/version produced this model?), audit, point-in-time correctness, and debugging "what did the system know when it made this decision?"

---

## Decision Cheat Sheet

| Need | Reach for |
|------|-----------|
| App transactions, mixed workload | Postgres (OLTP) |
| Analytics, big scans, aggregations | Columnar warehouse (OLAP) — Snowflake/ClickHouse/DuckDB |
| Write-heavy, always-on, multi-region | Cassandra / ScyllaDB |
| Outgrew one node's storage/writes | Shard (Citus / Vitess / native) |
| Crash-proof long-running workflow | Temporal / Cadence (or DBOS on Postgres) |
| Scheduled batch DAGs | Airflow / Dagster ([Data Eng](../../data-engineering/04-orchestration-modeling/)) |
| Relationships / multi-hop reasoning | Neo4j, or Apache AGE on Postgres (GraphRAG) |
| Embeddings + ANN retrieval | pgvector / Qdrant / Milvus |
| "What did we know, and when?" | Bitemporal modeling / event sourcing |

## Connections to Other Tracks

| Concept | Connected Track | Application |
|---------|-----------------|-------------|
| OLTP/OLAP, storage engines, ELT, SCDs | [Data Engineering](../../data-engineering/) | Where this storage knowledge is rooted |
| Replication, CAP, partitioning, consensus | [System Design](../../systems/01-system-design/) | The distributed-systems foundations |
| Idempotency, retries, partial failure | [RPC & Protocols](../02-rpc-and-protocols/) | Why durable execution exists |
| Vector stores, embeddings, RAG retrieval | [Training & Frameworks](../01-training-and-frameworks/) | Where embeddings are stored and queried |
| Checkpointed, resumable pipelines | [Training & Frameworks](../01-training-and-frameworks/) | Durable orchestration of training jobs |
| Event sourcing, replay, exactly-once | [Data Engineering](../../data-engineering/03-batch-streaming/) | The log as source of truth |
| Graph algorithms (BFS/DFS, shortest path) | [Algorithms — Graphs](../../algorithms/06-graphs/) | What graph databases execute |

## Company Relevance

| Company | How This Appears | Difficulty |
|---------|-----------------|------------|
| Uber | Built Cadence → Temporal; Cassandra at scale | Expert |
| Netflix / Datadog | Cassandra, durable workflows, sharding | Expert |
| Stripe | Sharded Postgres/Mongo, idempotent durable workflows | Expert |
| Temporal / DBOS | Durable execution as a product | Expert |
| Palantir | Knowledge graphs, entity resolution, bitemporal | Expert |
| Anthropic / OpenAI | Durable pipelines for training/eval, vector + graph retrieval | Expert |
</content>
