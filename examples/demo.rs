//! End-to-end walk through every engine with no network and no API keys.
//!
//! 1. Index a 15-note synthetic corpus with a deterministic hashing embedder.
//! 2. Run a hybrid (semantic + keyword) query fused with RRF, through a tiny
//!    reranker to show the hook.
//! 3. Pull a scoped memory and a knowledge-graph path for the same question.
//! 4. Assemble a context window under a token budget and print what made it
//!    in, what did not, and why.
//!
//! Run with `cargo run --example demo`.

use async_trait::async_trait;
use chrono::Utc;
use hybrid_retrieval::context::{
    BudgetingContextEngine, ContextAssemblyRequest, ContextBudget, ContextCandidate, ContextEngine,
    ContextRole, ContextSource, SelectionStrategy,
};
use hybrid_retrieval::core::{EngineRequestContext, EngineResult};
use hybrid_retrieval::hashing_embedder::HashingEmbeddingEngine;
use hybrid_retrieval::knowledge_graph::{
    EdgeId, GraphEdge, GraphMutation, GraphNode, InMemoryKnowledgeGraphEngine,
    KnowledgeGraphEngine, NodeId, NodeKind,
};
use hybrid_retrieval::memory::{
    InMemoryMemoryEngine, MemoryEngine, MemoryKind, MemoryProvenance, MemoryScope, MemorySearch,
    RememberMemory,
};
use hybrid_retrieval::pipeline::{
    CONVERSATION_ORDER, context_candidate_from_graph_path, context_candidates_from_memories,
    context_candidates_from_retrieval, document_dedupe_key, index_documents,
};
use hybrid_retrieval::retrieval::{
    HybridRetrievalEngine, InMemoryRetrievalBackend, RerankingEngine, RetrievalCandidate,
    RetrievalEngine, RetrievalQuery,
};
use hybrid_retrieval::sample_corpus::sample_documents;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;
use uuid::Uuid;

const QUESTION: &str = "Why does my espresso taste bitter and what should I change?";

/// A stand-in for a cross-encoder: nudges candidates that mention more of the
/// query's words. Real rerankers are a model call; the hook is the same shape.
struct TermDensityReranker;

#[async_trait]
impl RerankingEngine for TermDensityReranker {
    async fn rerank(
        &self,
        _context: &EngineRequestContext,
        query: &str,
        mut candidates: Vec<RetrievalCandidate>,
    ) -> EngineResult<Vec<RetrievalCandidate>> {
        let query_terms = words(query);
        for candidate in &mut candidates {
            let document_terms = words(&candidate.document.content);
            let overlap = query_terms.intersection(&document_terms).count() as f32;
            let density = overlap / query_terms.len().max(1) as f32;
            candidate.rerank_score = Some(density);
            candidate.fused_score += density * 0.01;
            candidate.reasons.push("reranked".into());
        }
        candidates.sort_by(|left, right| right.fused_score.total_cmp(&left.fused_score));
        Ok(candidates)
    }
}

fn words(text: &str) -> BTreeSet<String> {
    text.split(|character: char| !character.is_alphanumeric())
        .filter(|word| word.len() > 3)
        .map(str::to_lowercase)
        .collect()
}

fn preview(text: &str, width: usize) -> String {
    let flat = text.replace('\n', " ");
    if flat.chars().count() <= width {
        flat
    } else {
        format!("{}…", flat.chars().take(width - 1).collect::<String>())
    }
}

fn graph_node(workspace_id: Uuid, label: &str) -> GraphNode {
    let now = Utc::now();
    GraphNode {
        id: NodeId::new(),
        workspace_id,
        kind: NodeKind::Concept,
        label: label.into(),
        aliases: BTreeSet::new(),
        properties: BTreeMap::new(),
        confidence: 0.9,
        provenance: Vec::new(),
        created_at: now,
        updated_at: now,
    }
}

fn graph_edge(workspace_id: Uuid, from: NodeId, to: NodeId, relation: &str) -> GraphEdge {
    let now = Utc::now();
    GraphEdge {
        id: EdgeId::new(),
        workspace_id,
        from,
        to,
        relation: relation.into(),
        directed: true,
        weight: 1.0,
        confidence: 0.9,
        properties: BTreeMap::new(),
        provenance: Vec::new(),
        created_at: now,
        updated_at: now,
    }
}

#[tokio::main]
async fn main() -> EngineResult<()> {
    let mut context = EngineRequestContext::new(Uuid::new_v4());
    context.actor_id = Some("demo_user".into());

    // ── 1. index ───────────────────────────────────────────────────────────
    let embedder = Arc::new(HashingEmbeddingEngine::new(256)?);
    let backend = Arc::new(InMemoryRetrievalBackend::default());
    let documents = sample_documents();
    let document_count = index_documents(embedder.as_ref(), &backend, &context, documents).await?;
    println!(
        "Indexed {document_count} notes with the feature-hashing embedder (256-d, no API key).\n"
    );

    // ── 2. hybrid retrieval ────────────────────────────────────────────────
    let retrieval = HybridRetrievalEngine::new(embedder.clone(), backend.clone())?
        .with_reranker(Arc::new(TermDensityReranker));
    let mut query = RetrievalQuery::new(QUESTION);
    query.limit = 6;
    query.candidate_limit = 15;
    let response = retrieval.retrieve(&context, query).await?;
    println!("Query: {QUESTION}");
    println!(
        "Semantic arm: {} hits, keyword arm: {} hits, fused to top {}\n",
        response.semantic_hits,
        response.lexical_hits,
        response.candidates.len()
    );
    println!(
        "  {:<4} {:>7} {:>8} {:>7}  {:<38} source",
        "rank", "fused", "semantic", "keyword", "why"
    );
    for (rank, candidate) in response.candidates.iter().enumerate() {
        let provenance = &candidate.document.provenance;
        let page = provenance
            .location
            .as_ref()
            .and_then(|location| location.get("page"))
            .map(|page| format!("p.{page}"))
            .unwrap_or_default();
        println!(
            "  {:<4} {:>7.4} {:>8} {:>7}  {:<38} {} {}",
            rank + 1,
            candidate.fused_score,
            candidate
                .semantic_score
                .map(|score| format!("{score:.3}"))
                .unwrap_or_else(|| "-".into()),
            candidate
                .lexical_score
                .map(|score| format!("{score:.2}"))
                .unwrap_or_else(|| "-".into()),
            candidate.reasons.join("+"),
            provenance.source_title.as_deref().unwrap_or("?"),
            page
        );
        println!("       {}", preview(&candidate.document.content, 96));
    }

    // ── 3. memory + knowledge graph ────────────────────────────────────────
    let memory = InMemoryMemoryEngine::new()?;
    for (content, priority) in [
        (
            "The user brews espresso at home on a single boiler machine.",
            80,
        ),
        (
            "The user prefers answers that name one variable to change at a time.",
            90,
        ),
        ("The user drinks tea in the afternoon.", 30),
    ] {
        memory
            .remember(
                &context,
                RememberMemory {
                    scope: MemoryScope::User("demo_user".into()),
                    kind: MemoryKind::Preference,
                    content: content.into(),
                    summary: None,
                    confidence: 0.9,
                    priority,
                    provenance: MemoryProvenance::user_statement(),
                    tags: BTreeSet::new(),
                    contradicts: BTreeSet::new(),
                    expires_at: None,
                },
            )
            .await?;
    }
    let mut memory_search = MemorySearch::new("espresso change variable");
    memory_search.limit = 5;
    let memories = memory.search(&context, memory_search).await?;

    let graph = InMemoryKnowledgeGraphEngine::new()?;
    let workspace_id = context.workspace_id;
    let grind = graph_node(workspace_id, "finer grind");
    let extraction = graph_node(workspace_id, "faster extraction");
    let bitterness = graph_node(workspace_id, "bitterness");
    graph
        .upsert(
            &context,
            GraphMutation {
                nodes: vec![grind.clone(), extraction.clone(), bitterness.clone()],
                edges: vec![
                    graph_edge(workspace_id, grind.id, extraction.id, "causes"),
                    graph_edge(workspace_id, extraction.id, bitterness.id, "raises"),
                ],
            },
        )
        .await?;
    let path = graph
        .shortest_path(&context, grind.id, bitterness.id, 4)
        .await?
        .expect("seeded path exists");

    // ── 4. context assembly ────────────────────────────────────────────────
    let mut candidates = Vec::new();

    let mut system = ContextCandidate::new(
        ContextSource::System,
        ContextRole::System,
        "You are a coffee brewing assistant. Cite the handbook page for every claim.",
    );
    system.required = true;
    system.order = 0;
    candidates.push(system);

    let mut user_turn =
        ContextCandidate::new(ContextSource::Conversation, ContextRole::User, QUESTION);
    user_turn.required = true;
    user_turn.order = CONVERSATION_ORDER;
    candidates.push(user_turn);

    candidates.extend(context_candidates_from_memories(&memories));
    candidates.push(context_candidate_from_graph_path(&path));
    let retrieval_candidates = context_candidates_from_retrieval(&response.candidates);
    candidates.extend(retrieval_candidates);

    // The user had already pinned the top passage. Same dedupe key, higher
    // priority: the pinned copy should win and the retrieved copy should be
    // reported as a duplicate rather than spending tokens twice.
    if let Some(top) = response.candidates.first() {
        let mut pinned = ContextCandidate::new(
            ContextSource::UserPinned,
            ContextRole::Reference,
            top.document.content.clone(),
        );
        pinned.priority = 95;
        pinned.relevance = 1.0;
        pinned.order = 250;
        pinned.dedupe_key = Some(document_dedupe_key(&top.document));
        candidates.push(pinned);
    }

    let content_by_id = candidates
        .iter()
        .map(|candidate| (candidate.id, candidate.content.clone()))
        .collect::<HashMap<_, _>>();

    let budget = ContextBudget {
        model_window: 600,
        reserve_for_output: 256,
        fixed_overhead: 24,
        max_items: 12,
        // Retrieved passages may use at most 120 of the 320 available tokens.
        source_limits: BTreeMap::from([(ContextSource::Retrieval, 120)]),
    };
    let assembler = BudgetingContextEngine::new()?;
    let assembly = assembler
        .assemble(
            &context,
            ContextAssemblyRequest {
                budget,
                candidates,
                strategy: SelectionStrategy::PriorityThenRelevance,
            },
        )
        .await?;

    println!(
        "\nContext window: {} tokens total, {} reserved for output, {} available, {} used\n",
        assembly.model_window,
        assembly.reserved_output_tokens,
        assembly.available_tokens,
        assembly.used_tokens
    );
    println!("INCLUDED (in the order the model reads them):");
    for item in &assembly.items {
        println!(
            "  [{:<12}] {:>3} tok{}  {}",
            format!("{:?}", item.candidate.source),
            item.selected_tokens,
            if item.truncated {
                " (truncated)"
            } else {
                "            "
            },
            preview(&item.candidate.content, 80)
        );
    }
    println!("\nEXCLUDED:");
    for excluded in &assembly.excluded {
        let content = content_by_id
            .get(&excluded.candidate_id)
            .map(String::as_str)
            .unwrap_or("");
        println!(
            "  [{:<12}] {:<13} {}",
            format!("{:?}", excluded.source),
            format!("{:?}", excluded.reason),
            preview(content, 80)
        );
    }
    Ok(())
}
