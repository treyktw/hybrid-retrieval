//! Glue between the engines: indexing documents for retrieval, and turning
//! retrieval hits, memories and graph paths into context candidates.
//!
//! The engines deliberately do not know about each other — the context
//! assembler accepts candidates from anywhere and the retrieval engine does
//! not know a context window exists. This module is the one place that maps
//! one vocabulary onto the other, so the mapping decisions (what counts as a
//! duplicate, how a fused score becomes a relevance, where an item sits in
//! the final prompt) are visible and testable instead of scattered through
//! callers.

use engine_context::{ContextCandidate, ContextRole, ContextSource};
use engine_core::{EngineError, EngineRequestContext, EngineResult};
use engine_embedding::{EmbeddingEngine, EmbeddingPurpose, embed_all};
use engine_knowledge_graph::GraphPath;
use engine_memory::MemoryMatch;
use engine_retrieval::{InMemoryRetrievalBackend, RetrievalCandidate, RetrievalDocument};
use serde_json::Value;

/// Prompt position bands. The assembler sorts selected items by `order`, so
/// these decide the reading order the model sees: memories about the user,
/// then structured facts, then retrieved passages, then the live turn.
pub const MEMORY_ORDER: i64 = 100;
pub const GRAPH_ORDER: i64 = 200;
pub const RETRIEVAL_ORDER: i64 = 300;
pub const CONVERSATION_ORDER: i64 = 1_000;

/// Embed every document that does not already carry a vector, then store the
/// batch in the in-memory retrieval backend for the request's workspace.
///
/// Goes through [`embed_all`] rather than calling the embedder directly so
/// batch limits, cardinality, dimensions and non-finite values are checked the
/// same way they are at query time.
pub async fn index_documents(
    embedder: &dyn EmbeddingEngine,
    backend: &InMemoryRetrievalBackend,
    context: &EngineRequestContext,
    mut documents: Vec<RetrievalDocument>,
) -> EngineResult<usize> {
    let pending_indexes = documents
        .iter()
        .enumerate()
        .filter(|(_, document)| document.vector.is_empty())
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let pending_texts = pending_indexes
        .iter()
        .map(|index| documents[*index].content.clone())
        .collect::<Vec<_>>();
    let vectors = embed_all(
        embedder,
        context,
        &pending_texts,
        EmbeddingPurpose::Document,
    )
    .await?;
    if vectors.len() != pending_indexes.len() {
        return Err(EngineError::dependency(
            "index_embedding_count_mismatch",
            "embedding engine returned a different number of vectors than documents",
        ));
    }
    for (index, vector) in pending_indexes.into_iter().zip(vectors) {
        documents[index].vector = vector.values;
    }
    for document in &mut documents {
        if document.token_count.is_none() {
            document.token_count = Some(engine_context::estimate_tokens(&document.content));
        }
    }
    backend.replace(context, documents).await
}

/// Map ranked retrieval hits onto context candidates.
///
/// Relevance is the fused score divided by the best fused score in the batch.
/// RRF scores are tiny and only meaningful relative to each other (rank 0 in
/// both arms is ~0.033), so normalizing to the top hit gives the assembler a
/// 0..1 number it can weigh against memories without pretending RRF is a
/// probability.
pub fn context_candidates_from_retrieval(
    candidates: &[RetrievalCandidate],
) -> Vec<ContextCandidate> {
    let best = candidates
        .iter()
        .map(|candidate| candidate.fused_score)
        .fold(0.0_f32, f32::max);
    candidates
        .iter()
        .enumerate()
        .map(|(rank, candidate)| {
            let mut context_candidate = ContextCandidate::new(
                ContextSource::Retrieval,
                ContextRole::Reference,
                candidate.document.content.clone(),
            );
            context_candidate.relevance = if best > 0.0 {
                (candidate.fused_score / best).clamp(0.0, 1.0)
            } else {
                0.0
            };
            context_candidate.priority = 50;
            context_candidate.order = RETRIEVAL_ORDER + rank as i64;
            context_candidate.dedupe_key = Some(document_dedupe_key(&candidate.document));
            let metadata = &mut context_candidate.metadata;
            metadata.insert(
                "document_id".into(),
                candidate.document.id.to_string().into(),
            );
            metadata.insert(
                "source_id".into(),
                candidate.document.provenance.source_id.to_string().into(),
            );
            if let Some(title) = &candidate.document.provenance.source_title {
                metadata.insert("source_title".into(), title.clone().into());
            }
            if let Some(location) = &candidate.document.provenance.location {
                metadata.insert("location".into(), location.clone());
            }
            metadata.insert(
                "fused_score".into(),
                f64::from(candidate.fused_score).into(),
            );
            metadata.insert("reasons".into(), Value::from(candidate.reasons.clone()));
            context_candidate
        })
        .collect()
}

/// The key two candidates must share to count as the same content. Keyed on
/// the retrieval document id, so a passage the user pinned and the same
/// passage found by search collapse into one item.
pub fn document_dedupe_key(document: &RetrievalDocument) -> String {
    format!("document:{}", document.id)
}

/// Map memory matches onto context candidates. A memory keeps its own
/// priority, so a high-priority instruction outranks an incidental fact.
pub fn context_candidates_from_memories(matches: &[MemoryMatch]) -> Vec<ContextCandidate> {
    matches
        .iter()
        .enumerate()
        .map(|(rank, memory_match)| {
            let mut context_candidate = ContextCandidate::new(
                ContextSource::Memory,
                ContextRole::Reference,
                memory_match.memory.content.clone(),
            );
            context_candidate.relevance = memory_match.score.clamp(0.0, 1.0);
            context_candidate.priority = memory_match.memory.priority;
            context_candidate.order = MEMORY_ORDER + rank as i64;
            context_candidate.dedupe_key = Some(format!("memory:{}", memory_match.memory.id.0));
            context_candidate
                .metadata
                .insert("memory_version".into(), memory_match.memory.version.into());
            context_candidate
        })
        .collect()
}

/// Render a graph path as one short reference line, e.g.
/// `grind size --controls--> extraction --raises--> bitterness`.
pub fn render_graph_path(path: &GraphPath) -> String {
    let mut rendered = String::new();
    for (index, node) in path.nodes.iter().enumerate() {
        if index > 0 {
            let relation = path
                .edges
                .get(index - 1)
                .map(|edge| edge.relation.as_str())
                .unwrap_or("related_to");
            rendered.push_str(&format!(" --{relation}--> "));
        }
        rendered.push_str(&node.label);
    }
    rendered
}

/// A graph path as a context candidate. Graph facts are derived rather than
/// quoted, so they enter as tool output rather than as a retrieved passage.
pub fn context_candidate_from_graph_path(path: &GraphPath) -> ContextCandidate {
    let mut context_candidate = ContextCandidate::new(
        ContextSource::Tool,
        ContextRole::Tool,
        format!("Related concepts: {}", render_graph_path(path)),
    );
    context_candidate.priority = 60;
    context_candidate.relevance = 0.7;
    context_candidate.order = GRAPH_ORDER;
    context_candidate.truncatable = false;
    context_candidate
}
