//! Hybrid retrieval and context assembly, as a set of small engines.
//!
//! Every engine remains an independent Cargo package under `crates/`, while
//! consumers need only this one dependency. The namespace stays explicit so a
//! growing engine catalogue does not flatten hundreds of types into one root.

#![forbid(unsafe_code)]

pub use engine_context as context;
pub use engine_core as core;
pub use engine_embedding as embedding;
pub use engine_knowledge_graph as knowledge_graph;
pub use engine_memory as memory;
#[cfg(feature = "postgres")]
pub use engine_postgres as postgres;
pub use engine_retrieval as retrieval;
pub use engine_vector as vector;

pub mod hashing_embedder;
pub mod pipeline;
pub mod sample_corpus;
