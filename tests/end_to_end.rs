//! Integration tests across crates: indexing -> hybrid retrieval -> context
//! assembly, plus the cross-engine invariants the glue in `pipeline` relies on.

use async_trait::async_trait;
use hybrid_retrieval::context::{
    BudgetingContextEngine, ContextAssemblyRequest, ContextBudget, ContextCandidate, ContextEngine,
    ContextRole, ContextSource, ExclusionReason, SelectionStrategy,
};
use hybrid_retrieval::core::{EngineRequestContext, EngineResult};
use hybrid_retrieval::hashing_embedder::HashingEmbeddingEngine;
use hybrid_retrieval::memory::{
    InMemoryMemoryEngine, MemoryEngine, MemoryKind, MemoryProvenance, MemoryScope, MemorySearch,
    RememberMemory, UpdateMemory,
};
use hybrid_retrieval::pipeline::{
    context_candidates_from_memories, context_candidates_from_retrieval, document_dedupe_key,
    index_documents,
};
use hybrid_retrieval::retrieval::{
    HybridRetrievalEngine, InMemoryRetrievalBackend, RerankingEngine, RetrievalCandidate,
    RetrievalEngine, RetrievalQuery,
};
use hybrid_retrieval::sample_corpus::{
    FIELD_GUIDE_SOURCE_ID, HANDBOOK_SOURCE_ID, sample_documents,
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use uuid::Uuid;

async fn indexed_engine(
    context: &EngineRequestContext,
) -> (HybridRetrievalEngine, Arc<InMemoryRetrievalBackend>) {
    let embedder = Arc::new(HashingEmbeddingEngine::new(128).unwrap());
    let backend = Arc::new(InMemoryRetrievalBackend::default());
    let indexed = index_documents(embedder.as_ref(), &backend, context, sample_documents())
        .await
        .unwrap();
    assert_eq!(indexed, 15);
    let engine = HybridRetrievalEngine::new(embedder, backend.clone()).unwrap();
    (engine, backend)
}

fn budget(model_window: usize, retrieval_limit: Option<usize>) -> ContextBudget {
    ContextBudget {
        model_window,
        reserve_for_output: 100,
        fixed_overhead: 0,
        max_items: 20,
        source_limits: retrieval_limit
            .map(|limit| BTreeMap::from([(ContextSource::Retrieval, limit)]))
            .unwrap_or_default(),
    }
}

#[tokio::test]
async fn indexed_corpus_is_retrievable_with_provenance_intact() {
    let context = EngineRequestContext::new(Uuid::new_v4());
    let (engine, _) = indexed_engine(&context).await;
    let response = engine
        .retrieve(&context, RetrievalQuery::new("cold brew for twelve hours"))
        .await
        .unwrap();
    let top = &response.candidates[0];
    assert!(top.document.content.starts_with("Cold brew"));
    assert_eq!(top.document.provenance.source_id, FIELD_GUIDE_SOURCE_ID);
    assert_eq!(
        top.document.provenance.location,
        Some(serde_json::json!({"page": 9}))
    );
    // Found by both arms, and the reasons say so.
    assert!(top.reasons.contains(&"semantic_match".to_owned()));
    assert!(top.reasons.contains(&"lexical_match".to_owned()));
    assert!(top.document.token_count.is_some());
}

#[tokio::test]
async fn retrieval_filters_restrict_to_one_source() {
    let context = EngineRequestContext::new(Uuid::new_v4());
    let (engine, _) = indexed_engine(&context).await;
    let mut query = RetrievalQuery::new("espresso bitter extraction");
    query.filters.source_ids = vec![HANDBOOK_SOURCE_ID];
    let response = engine.retrieve(&context, query).await.unwrap();
    assert!(!response.candidates.is_empty());
    assert!(
        response
            .candidates
            .iter()
            .all(|candidate| candidate.document.provenance.source_id == HANDBOOK_SOURCE_ID)
    );
}

#[tokio::test]
async fn another_workspace_sees_nothing_from_the_indexed_corpus() {
    let context = EngineRequestContext::new(Uuid::new_v4());
    let (engine, _) = indexed_engine(&context).await;
    let stranger = EngineRequestContext::new(Uuid::new_v4());
    let response = engine
        .retrieve(&stranger, RetrievalQuery::new("espresso"))
        .await
        .unwrap();
    assert!(response.candidates.is_empty());
}

/// Reverses whatever order fusion produced, so the test can prove the
/// reranker's order is the one returned.
struct ReversingReranker;

#[async_trait]
impl RerankingEngine for ReversingReranker {
    async fn rerank(
        &self,
        _context: &EngineRequestContext,
        _query: &str,
        mut candidates: Vec<RetrievalCandidate>,
    ) -> EngineResult<Vec<RetrievalCandidate>> {
        candidates.reverse();
        for (rank, candidate) in candidates.iter_mut().enumerate() {
            candidate.rerank_score = Some(1.0 / (rank as f32 + 1.0));
        }
        Ok(candidates)
    }
}

#[tokio::test]
async fn reranker_has_the_final_say_on_order() {
    let context = EngineRequestContext::new(Uuid::new_v4());
    let (plain, backend) = indexed_engine(&context).await;
    let embedder = Arc::new(HashingEmbeddingEngine::new(128).unwrap());
    let reranked = HybridRetrievalEngine::new(embedder, backend)
        .unwrap()
        .with_reranker(Arc::new(ReversingReranker));
    let mut query = RetrievalQuery::new("grind size extraction espresso");
    query.limit = 5;
    let plain_ids = plain
        .retrieve(&context, query.clone())
        .await
        .unwrap()
        .candidates
        .into_iter()
        .map(|candidate| candidate.document.id)
        .collect::<Vec<_>>();
    let reranked_response = reranked.retrieve(&context, query).await.unwrap();
    let mut reranked_ids = reranked_response
        .candidates
        .iter()
        .map(|candidate| candidate.document.id)
        .collect::<Vec<_>>();
    assert!(
        reranked_response
            .candidates
            .iter()
            .all(|candidate| candidate.rerank_score.is_some())
    );
    reranked_ids.reverse();
    assert_eq!(plain_ids, reranked_ids);
}

#[tokio::test]
async fn retrieval_to_context_respects_source_quota_and_reports_every_exclusion() {
    let context = EngineRequestContext::new(Uuid::new_v4());
    let (engine, _) = indexed_engine(&context).await;
    let mut query = RetrievalQuery::new("why is my espresso bitter");
    query.limit = 10;
    let response = engine.retrieve(&context, query).await.unwrap();
    let candidates = context_candidates_from_retrieval(&response.candidates);
    let candidate_count = candidates.len();

    let assembly = BudgetingContextEngine::new()
        .unwrap()
        .assemble(
            &context,
            ContextAssemblyRequest {
                budget: budget(1_000, Some(90)),
                candidates,
                strategy: SelectionStrategy::PriorityThenRelevance,
            },
        )
        .await
        .unwrap();

    let retrieval_tokens = assembly
        .items
        .iter()
        .filter(|item| item.candidate.source == ContextSource::Retrieval)
        .map(|item| item.selected_tokens)
        .sum::<usize>();
    assert!(
        retrieval_tokens <= 90,
        "retrieval used {retrieval_tokens} tokens"
    );
    // Nothing silently disappears: every candidate is either in or explained.
    assert_eq!(
        assembly.items.len() + assembly.excluded.len(),
        candidate_count
    );
    assert!(
        assembly
            .excluded
            .iter()
            .all(|excluded| excluded.reason == ExclusionReason::SourceBudget)
    );
    // Provenance survives the trip into the context package.
    assert!(
        assembly
            .items
            .iter()
            .all(|item| item.candidate.metadata.contains_key("location"))
    );
}

#[tokio::test]
async fn pinned_copy_of_a_retrieved_passage_wins_dedupe() {
    let context = EngineRequestContext::new(Uuid::new_v4());
    let (engine, _) = indexed_engine(&context).await;
    let response = engine
        .retrieve(&context, RetrievalQuery::new("channeling espresso puck"))
        .await
        .unwrap();
    let top = response.candidates[0].clone();
    let mut candidates = context_candidates_from_retrieval(&response.candidates);
    let retrieved_copy_id = candidates[0].id;
    let mut pinned = ContextCandidate::new(
        ContextSource::UserPinned,
        ContextRole::Reference,
        top.document.content.clone(),
    );
    pinned.priority = 95;
    pinned.dedupe_key = Some(document_dedupe_key(&top.document));
    let pinned_id = pinned.id;
    candidates.push(pinned);

    let assembly = BudgetingContextEngine::new()
        .unwrap()
        .assemble(
            &context,
            ContextAssemblyRequest {
                budget: budget(4_000, None),
                candidates,
                strategy: SelectionStrategy::PriorityThenRelevance,
            },
        )
        .await
        .unwrap();
    assert!(
        assembly
            .items
            .iter()
            .any(|item| item.candidate.id == pinned_id)
    );
    assert!(assembly.excluded.iter().any(|excluded| {
        excluded.candidate_id == retrieved_copy_id && excluded.reason == ExclusionReason::Duplicate
    }));
}

#[tokio::test]
async fn memory_candidates_carry_priority_and_version_into_context() {
    let mut context = EngineRequestContext::new(Uuid::new_v4());
    context.actor_id = Some("user_a".into());
    let memory = InMemoryMemoryEngine::new().unwrap();
    let remembered = memory
        .remember(
            &context,
            RememberMemory {
                scope: MemoryScope::User("user_a".into()),
                kind: MemoryKind::Instruction,
                content: "Always give espresso advice one variable at a time".into(),
                summary: None,
                confidence: 0.9,
                priority: 90,
                provenance: MemoryProvenance::user_statement(),
                tags: BTreeSet::new(),
                contradicts: BTreeSet::new(),
                expires_at: None,
            },
        )
        .await
        .unwrap();
    memory
        .update(
            &context,
            remembered.id,
            UpdateMemory {
                expected_version: Some(1),
                confidence: Some(1.0),
                ..UpdateMemory::default()
            },
        )
        .await
        .unwrap();
    let matches = memory
        .search(&context, MemorySearch::new("espresso advice"))
        .await
        .unwrap();
    let candidates = context_candidates_from_memories(&matches);
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].priority, 90);
    assert_eq!(candidates[0].source, ContextSource::Memory);
    assert_eq!(candidates[0].metadata["memory_version"], 2);
}

#[tokio::test]
async fn assembly_is_deterministic_for_the_same_inputs() {
    let context = EngineRequestContext::new(Uuid::new_v4());
    let (engine, _) = indexed_engine(&context).await;
    let response = engine
        .retrieve(
            &context,
            RetrievalQuery::new("water temperature extraction"),
        )
        .await
        .unwrap();
    let candidates = context_candidates_from_retrieval(&response.candidates);
    let assembler = BudgetingContextEngine::new().unwrap();
    let run = |candidates: Vec<ContextCandidate>| {
        let assembler = &assembler;
        let context = &context;
        async move {
            assembler
                .assemble(
                    context,
                    ContextAssemblyRequest {
                        budget: budget(260, None),
                        candidates,
                        strategy: SelectionStrategy::RelevancePerToken,
                    },
                )
                .await
                .unwrap()
        }
    };
    let first = run(candidates.clone()).await;
    let second = run(candidates).await;
    let ids = |assembly: &hybrid_retrieval::context::ContextAssembly| {
        assembly
            .items
            .iter()
            .map(|item| (item.candidate.id, item.selected_tokens))
            .collect::<Vec<_>>()
    };
    assert_eq!(ids(&first), ids(&second));
    assert!(first.used_tokens <= first.available_tokens);
}
