# hybrid-retrieval

Hybrid semantic + keyword retrieval, scoped memory, a bounded knowledge graph and an auditable token-budgeted context assembler, written as small Rust engines that share one contract and each come with an in-memory reference implementation.

## Why this exists

An AI tutor answers from a mix of things: the learner's uploaded material, what it remembers about the learner, how concepts connect, and the conversation so far. All of that has to fit into a fixed model window, and when an answer is wrong someone has to be able to ask "why did the model see *this* and not *that*?". This code was extracted from the context layer of a production AI learning workspace. It is the part that decides what gets retrieved, what gets remembered, and what actually makes it into the prompt, with the reason for every inclusion and exclusion recorded.

## Architecture

```mermaid
flowchart LR
    subgraph core["engine-core"]
        C[Engine trait · metadata · health<br/>EngineError with stable codes<br/>EngineRequestContext: workspace + deadline]
    end

    Q[query] --> RET
    DOCS[documents] --> IDX[pipeline::index_documents]
    IDX --> EMB[engine-embedding<br/>embed_all: batching +<br/>response validation]
    IDX --> BE[(RetrievalBackend<br/>in-memory / your store)]

    subgraph RET[engine-retrieval · HybridRetrievalEngine]
        direction TB
        E2[embed query] --> SEM[semantic arm<br/>cosine]
        BE2[ ] -.-> LEX[keyword arm<br/>token overlap]
        SEM --> RRF[reciprocal-rank fusion<br/>k = 60, reasons kept]
        LEX --> RRF
        RRF --> RR[optional RerankingEngine]
    end

    MEM[engine-memory<br/>scoped, versioned,<br/>conflict-aware] --> MAP
    KG[engine-knowledge-graph<br/>bounded BFS, shortest path] --> MAP
    RR --> MAP[pipeline::context_candidates_*]
    MAP --> CTX[engine-context<br/>BudgetingContextEngine]
    CTX --> OUT[ContextAssembly<br/>items in reading order<br/>+ every exclusion with a reason]

    VEC[engine-vector<br/>in-memory / pgvector] -.optional durable store.-> BE
    PG[engine-postgres<br/>pgvector adapter +<br/>embedding spend cap] -.feature = postgres.-> VEC
```

The repository is a Cargo workspace. Each engine is its own crate that depends only on `engine-core`, so they compile, test and get replaced independently. The root crate `hybrid-retrieval` re-exports them and adds `pipeline` — the one place that maps retrieval hits, memories and graph paths onto context candidates.

| crate | what it does |
| --- | --- |
| `engine-core` | `Engine` trait, metadata/health, `EngineError { code, kind, retryable }`, request context with workspace id and deadline |
| `engine-embedding` | `EmbeddingEngine` contract and `embed_all`, which batches and rejects wrong cardinality, wrong width, NaN and all-zero vectors from a provider |
| `engine-vector` | `VectorEngine` contract (upsert / filtered cosine search / delete), workspace-isolated in-memory store |
| `engine-retrieval` | `HybridRetrievalEngine`: semantic and keyword arms run concurrently, fused with RRF, optional reranker |
| `engine-memory` | user/workspace memory with explicit scope, provenance, confidence, expiry, optimistic versions and explicit conflict reporting |
| `engine-knowledge-graph` | nodes and edges with provenance; traversal bounded by depth and node count; shortest path |
| `engine-context` | deterministic context assembly under a token budget with per-source quotas, dedupe and an exclusion audit |
| `engine-postgres` *(feature `postgres`)* | pgvector-backed `VectorEngine` and `BudgetedEmbedder`, a spend ceiling that survives restarts |

### Design decisions and tradeoffs

- **Reciprocal-rank fusion instead of score blending.** Cosine similarity and keyword scores live on unrelated scales; normalizing and weighting them is fragile and needs retuning whenever either arm changes. RRF only looks at rank (`1 / (60 + rank + 1)`), so a document both arms agree on wins without any calibration. The cost is that a dominant hit in one arm cannot win by a landslide. Raw `semantic_score` and `lexical_score` stay on the candidate alongside `reasons` (`semantic_match`, `lexical_match`, `reranked`), so "why did this come back" can be answered after the fact.
- **The reranker gets the last word, but sits behind a trait.** `RerankingEngine` receives the fused list and returns it in its own order. A cross-encoder or LLM judge can be added without touching fusion, and the system still works without one.
- **The context assembler is an auditor, not just a packer.** Every candidate ends up either in `items` or in `excluded` with a reason: `Duplicate`, `TokenBudget`, `SourceBudget` or `ItemLimit`. Required items (system prompt, current turn) are never displaced. If they cannot fit, assembly fails loudly instead of silently dropping them. Selection is greedy by priority then relevance, but the output is re-sorted into a stable *reading* order, so the model never sees the order the selection loop happened to pick things in.
- **Per-source quotas.** Retrieval is the source most likely to flood a window, so it can be capped (`source_limits`) without starving memory or tool output. Truncation only applies to optional, truncatable items with at least 16 tokens of room, and cuts at a word boundary.
- **Workspace isolation is part of the storage key in every reference store**, not a filter callers have to remember. Personal (`Global`) memory deliberately crosses workspaces but not users.
- **Optimistic concurrency for memory.** `UpdateMemory { expected_version }` makes two concurrent edits a `Conflict` error rather than a lost write. Contradictions between memories are recorded explicitly and surfaced by `conflicts()` instead of being resolved by whichever was written last.
- **Bounded graph work.** Traversal requires a depth of 1–12 and a node cap of at most 10k, and reports `truncated` when it stops early. A request can never turn into an unbounded walk of the graph.
- **Errors are data.** `EngineError` carries a stable snake_case `code`, a kind and a `retryable` flag, so callers branch on `code == "memory_version_conflict"` instead of parsing messages.
- **pgvector at 3072 dimensions.** pgvector's HNSW index caps `vector` at 2000 dimensions, but `halfvec` goes to 4000. The Postgres adapter keeps full-precision `vector` storage and builds the index on the `embedding::halfvec(3072)` expression, ordering by the same cast so the planner uses it. Vectors live in a per-model side table (`engine_vector_embeddings`), and searches join on the asking model, because two models' vectors of the same width are not comparable and mixing them fails silently. The migration comments explain this in more detail.
- **A spend ceiling, not a rate limiter.** `BudgetedEmbedder` wraps any embedder and refuses to start a request once persisted spend reaches the budget. It uses the provider's billed token count when one is reported and a deliberately pessimistic estimate when not.

## Quick start

```bash
git clone <this repo> && cd hybrid-retrieval
cargo run --example demo
```

The demo needs no network access and no API keys. It indexes 15 synthetic coffee-brewing notes with a feature-hashing embedder, runs a hybrid query through a toy reranker, adds two memories and a knowledge-graph path, then assembles a 600-token window with 256 tokens reserved for output and a 120-token cap on retrieved passages:

```text
Query: Why does my espresso taste bitter and what should I change?
Semantic arm: 15 hits, keyword arm: 14 hits, fused to top 6

  rank   fused semantic keyword  why                                    source
  1     0.0356    0.321    0.27  semantic_match+lexical_match+reranked  Home Brewing Handbook p.7
       Grind size controls extraction speed. A finer grind exposes more surface area, so water extract…
  2     0.0351    0.282    0.27  semantic_match+lexical_match+reranked  Home Brewing Handbook p.19
  ...

Context window: 600 tokens total, 256 reserved for output, 320 available, 249 used

INCLUDED (in the order the model reads them):
  [System      ]  19 tok              You are a coffee brewing assistant. Cite the handbook page for every claim.
  [Memory      ]  17 tok              The user prefers answers that name one variable to change at a time.
  [Memory      ]  15 tok              The user brews espresso at home on a single boiler machine.
  [Tool        ]  21 tok              Related concepts: finer grind --causes--> faster extraction --raises--> bittern…
  [UserPinned  ]  42 tok              Grind size controls extraction speed. A finer grind exposes more surface area, …
  [Retrieval   ]  46 tok              Channeling happens when water finds a weak path through the espresso puck. Part…
  [Retrieval   ]  39 tok              Backflush an espresso machine weekly and descale it according to water hardness…
  [Retrieval   ]  35 tok (truncated)  Under-extracted coffee tastes sour and thin because the brew stopped before sug…
  [Conversation]  15 tok              Why does my espresso taste bitter and what should I change?

EXCLUDED:
  [Retrieval   ] Duplicate     Grind size controls extraction speed. A finer grind exposes more surface area, …
  [Retrieval   ] SourceBudget  Water hardness matters. Some minerals help extraction, but very hard water scal…
  [Retrieval   ] SourceBudget  Light roasts keep more origin acidity and need finer grinds or hotter water to …
```

The user had pinned the top passage, so the retrieved copy is excluded as a `Duplicate`. The last two hits did not fit the retrieval quota and are reported as `SourceBudget`, not silently dropped.

Using the engines from your own code:

```rust
use hybrid_retrieval::{retrieval::*, hashing_embedder::HashingEmbeddingEngine, pipeline};
use std::sync::Arc;

let embedder = Arc::new(HashingEmbeddingEngine::new(256)?);   // or your provider's EmbeddingEngine
let backend = Arc::new(InMemoryRetrievalBackend::default());  // or your RetrievalBackend
pipeline::index_documents(embedder.as_ref(), &backend, &ctx, documents).await?;
let engine = HybridRetrievalEngine::new(embedder, backend)?;
let hits = engine.retrieve(&ctx, RetrievalQuery::new("why is my espresso bitter")).await?;
let candidates = pipeline::context_candidates_from_retrieval(&hits.candidates);
```

> **About the embedders.** `engine_embedding::DeterministicEmbeddingEngine` is a contract-test double. It deliberately mixes the purpose (query vs. document) into its hashes, so its vectors are not meant to be compared semantically. The demo uses `HashingEmbeddingEngine`, which hashes words and character trigrams the same way for every purpose. Related wording really does score higher with it, so the semantic arm has something to contribute. Neither replaces a real embedding model.

## Tests

```bash
cargo test --workspace --all-features
```

- **29 unit tests** inside the crates: 27 carried over from production, plus 2 for the demo embedder. They cover: RRF rewarding agreement between the arms; workspace isolation in every store; vectors with the wrong width, all zeros or NaN rejected with no partial writes; optimistic memory versions stopping lost writes; expired and forgotten memories never returned; personal memory crossing workspaces but not users; graph traversal and paths respecting their bounds; undirected edges; invalid edges rejected before any write; the context budget never exceeded; required-context overflow failing explicitly; dedupe keeping the stronger candidate; embedding spend arithmetic and the token estimate never undercounting.
- **8 integration tests** (`tests/end_to_end.rs`) across crates: the indexed corpus comes back with provenance (page numbers) intact; source filters; workspace isolation end to end; the reranker deciding the final order; the retrieval→context path respecting source quotas, with every candidate either included or explained; a pinned passage winning dedupe over its retrieved copy; memory priority and version carried into context; assembly deterministic for the same inputs.
- **3 Postgres contract tests** (`crates/postgres/tests/postgres_contract.rs`), ignored by default: ranking, metadata filters, workspace isolation and delete against real pgvector; searches seeing only the asking model's vectors; the spend ceiling surviving a fresh `BudgetedEmbedder` over the same database.

To run the Postgres tests (needs Docker, or any Postgres with pgvector >= 0.7 for `halfvec`):

```bash
docker compose up -d
TEST_DATABASE_URL=postgres://postgres:postgres@localhost:54329/retrieval \
  cargo test -p engine-postgres -- --ignored --test-threads=1
```

CI runs fmt, clippy (`-D warnings`), the full test suite and the demo, plus a second job that runs the Postgres tests against a `pgvector/pgvector:pg17` service container.

## What was left out

The production system wires these engines to hosted embedding providers (with retries and per-provider pricing), a Postgres-backed retrieval backend over its own document/chunk schema using `tsvector` for the keyword arm, and Postgres implementations of the memory and graph engines. Those adapters depend on the product's schema (users, workspaces, sources), auth and job queue, so they are not included. What remains is the vendor-neutral core, a reference implementation for every contract, and the pgvector vector store plus spend ceiling, which stand on their own. Document extraction and chunking live in a separate repository.
