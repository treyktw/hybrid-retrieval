//! Hybrid retrieval.
//!
//! The engine combines semantic and lexical recall using reciprocal-rank
//! fusion, keeps provenance attached, and optionally delegates final ordering
//! to a reranker. Backends remain replaceable.

#![forbid(unsafe_code)]

use async_trait::async_trait;
use engine_core::{
    CostClass, Engine, EngineHealth, EngineId, EngineKind, EngineMetadata, EngineRequestContext,
    EngineResult,
};
use engine_embedding::{EmbeddingEngine, EmbeddingPurpose, embed_all};
use engine_vector::cosine_similarity;
use futures::future::try_join;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use uuid::Uuid;

pub const RRF_K: f32 = 60.0;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RetrievalFilters {
    #[serde(default)]
    pub source_ids: Vec<Uuid>,
    #[serde(default)]
    pub kinds: Vec<String>,
    #[serde(default)]
    pub metadata_equals: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalQuery {
    pub text: String,
    pub limit: usize,
    pub candidate_limit: usize,
    #[serde(default)]
    pub filters: RetrievalFilters,
    pub minimum_score: Option<f32>,
}

impl RetrievalQuery {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            limit: 8,
            candidate_limit: 40,
            filters: RetrievalFilters::default(),
            minimum_score: None,
        }
    }

    fn validate(&self) -> EngineResult<()> {
        if self.text.trim().is_empty() {
            return Err(engine_core::EngineError::invalid_input(
                "retrieval_query_empty",
                "retrieval query must not be empty",
            ));
        }
        if self.limit == 0 || self.limit > 100 {
            return Err(engine_core::EngineError::invalid_input(
                "retrieval_limit_invalid",
                "retrieval limit must be between 1 and 100",
            ));
        }
        if self.candidate_limit < self.limit || self.candidate_limit > 2_000 {
            return Err(engine_core::EngineError::invalid_input(
                "retrieval_candidates_invalid",
                "candidate limit must be at least the result limit and no more than 2000",
            ));
        }
        if self
            .minimum_score
            .is_some_and(|score| !score.is_finite() || !(0.0..=1.0).contains(&score))
        {
            return Err(engine_core::EngineError::invalid_input(
                "retrieval_minimum_score_invalid",
                "minimum score must be finite and between 0 and 1",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Provenance {
    pub source_id: Uuid,
    pub source_title: Option<String>,
    pub source_kind: Option<String>,
    pub uri: Option<String>,
    pub location: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalDocument {
    pub id: Uuid,
    pub content: String,
    pub provenance: Provenance,
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
    #[serde(default)]
    pub vector: Vec<f32>,
    pub token_count: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalCandidate {
    pub document: RetrievalDocument,
    pub semantic_score: Option<f32>,
    pub lexical_score: Option<f32>,
    pub rerank_score: Option<f32>,
    pub fused_score: f32,
    #[serde(default)]
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalResponse {
    pub candidates: Vec<RetrievalCandidate>,
    pub semantic_hits: usize,
    pub lexical_hits: usize,
    pub took_ms: u64,
    pub engine_id: EngineId,
}

#[async_trait]
pub trait RetrievalBackend: Send + Sync {
    async fn semantic_search(
        &self,
        context: &EngineRequestContext,
        query: &RetrievalQuery,
        query_vector: &[f32],
    ) -> EngineResult<Vec<RetrievalCandidate>>;

    async fn lexical_search(
        &self,
        context: &EngineRequestContext,
        query: &RetrievalQuery,
    ) -> EngineResult<Vec<RetrievalCandidate>>;

    async fn health(&self) -> EngineHealth {
        EngineHealth::healthy(0)
    }
}

#[async_trait]
pub trait RerankingEngine: Send + Sync {
    async fn rerank(
        &self,
        context: &EngineRequestContext,
        query: &str,
        candidates: Vec<RetrievalCandidate>,
    ) -> EngineResult<Vec<RetrievalCandidate>>;
}

#[async_trait]
pub trait RetrievalEngine: Engine {
    async fn retrieve(
        &self,
        context: &EngineRequestContext,
        query: RetrievalQuery,
    ) -> EngineResult<RetrievalResponse>;
}

pub struct HybridRetrievalEngine {
    metadata: EngineMetadata,
    embedder: Arc<dyn EmbeddingEngine>,
    backend: Arc<dyn RetrievalBackend>,
    reranker: Option<Arc<dyn RerankingEngine>>,
}

impl HybridRetrievalEngine {
    pub fn new(
        embedder: Arc<dyn EmbeddingEngine>,
        backend: Arc<dyn RetrievalBackend>,
    ) -> EngineResult<Self> {
        let id = EngineId::new("retrieval.hybrid-rrf-v1")?;
        let mut metadata =
            EngineMetadata::new(id, EngineKind::Retrieval, "Hybrid RRF retrieval engine")
                .with_capabilities([
                    "semantic",
                    "lexical",
                    "metadata_filter",
                    "provenance",
                    "rrf",
                ]);
        metadata.cost_class = CostClass::Variable;
        Ok(Self {
            metadata,
            embedder,
            backend,
            reranker: None,
        })
    }

    pub fn with_reranker(mut self, reranker: Arc<dyn RerankingEngine>) -> Self {
        self.metadata.capabilities.insert("reranking".into());
        self.reranker = Some(reranker);
        self
    }
}

#[async_trait]
impl Engine for HybridRetrievalEngine {
    fn metadata(&self) -> EngineMetadata {
        self.metadata.clone()
    }

    async fn health(&self) -> EngineHealth {
        self.backend.health().await
    }
}

#[async_trait]
impl RetrievalEngine for HybridRetrievalEngine {
    async fn retrieve(
        &self,
        context: &EngineRequestContext,
        query: RetrievalQuery,
    ) -> EngineResult<RetrievalResponse> {
        context.ensure_active()?;
        query.validate()?;
        let started = Instant::now();
        let texts = vec![query.text.clone()];
        let query_vector = embed_all(
            self.embedder.as_ref(),
            context,
            &texts,
            EmbeddingPurpose::Query,
        )
        .await?
        .into_iter()
        .next()
        .ok_or_else(|| {
            engine_core::EngineError::dependency(
                "retrieval_query_embedding_missing",
                "embedding engine returned no query vector",
            )
        })?
        .values;

        let semantic = self.backend.semantic_search(context, &query, &query_vector);
        let lexical = self.backend.lexical_search(context, &query);
        let (semantic, lexical) = try_join(semantic, lexical).await?;
        let semantic_hits = semantic.len();
        let lexical_hits = lexical.len();
        let mut candidates = reciprocal_rank_fusion(semantic, lexical, query.limit);

        if let Some(reranker) = &self.reranker {
            candidates = reranker.rerank(context, &query.text, candidates).await?;
            candidates.truncate(query.limit);
        }
        if let Some(minimum) = query.minimum_score {
            candidates.retain(|candidate| candidate.fused_score >= minimum);
        }

        Ok(RetrievalResponse {
            candidates,
            semantic_hits,
            lexical_hits,
            took_ms: started.elapsed().as_millis() as u64,
            engine_id: self.metadata.id.clone(),
        })
    }
}

pub fn reciprocal_rank(rank: usize) -> f32 {
    1.0 / (RRF_K + rank as f32 + 1.0)
}

pub fn reciprocal_rank_fusion(
    semantic: Vec<RetrievalCandidate>,
    lexical: Vec<RetrievalCandidate>,
    limit: usize,
) -> Vec<RetrievalCandidate> {
    let mut merged = HashMap::<Uuid, RetrievalCandidate>::new();
    for (rank, mut candidate) in semantic.into_iter().enumerate() {
        candidate.fused_score = reciprocal_rank(rank);
        candidate.reasons.push("semantic_match".into());
        merged.insert(candidate.document.id, candidate);
    }
    for (rank, mut candidate) in lexical.into_iter().enumerate() {
        let contribution = reciprocal_rank(rank);
        merged
            .entry(candidate.document.id)
            .and_modify(|existing| {
                existing.fused_score += contribution;
                existing.lexical_score = candidate.lexical_score;
                if !existing
                    .reasons
                    .iter()
                    .any(|reason| reason == "lexical_match")
                {
                    existing.reasons.push("lexical_match".into());
                }
            })
            .or_insert_with(|| {
                candidate.fused_score = contribution;
                candidate.reasons.push("lexical_match".into());
                candidate
            });
    }
    let mut output = merged.into_values().collect::<Vec<_>>();
    output.sort_by(|left, right| {
        right
            .fused_score
            .total_cmp(&left.fused_score)
            .then_with(|| left.document.id.cmp(&right.document.id))
    });
    output.truncate(limit);
    output
}

#[derive(Clone, Default)]
pub struct InMemoryRetrievalBackend {
    documents: Arc<RwLock<HashMap<(Uuid, Uuid), RetrievalDocument>>>,
}

impl InMemoryRetrievalBackend {
    pub async fn replace(
        &self,
        context: &EngineRequestContext,
        documents: Vec<RetrievalDocument>,
    ) -> EngineResult<usize> {
        context.ensure_active()?;
        let count = documents.len();
        let mut stored = self.documents.write().await;
        for document in documents {
            stored.insert((context.workspace_id, document.id), document);
        }
        Ok(count)
    }
}

fn matches_filters(document: &RetrievalDocument, filters: &RetrievalFilters) -> bool {
    (filters.source_ids.is_empty() || filters.source_ids.contains(&document.provenance.source_id))
        && (filters.kinds.is_empty()
            || document
                .provenance
                .source_kind
                .as_ref()
                .is_some_and(|kind| filters.kinds.contains(kind)))
        && filters
            .metadata_equals
            .iter()
            .all(|(key, value)| document.metadata.get(key) == Some(value))
}

#[async_trait]
impl RetrievalBackend for InMemoryRetrievalBackend {
    async fn semantic_search(
        &self,
        context: &EngineRequestContext,
        query: &RetrievalQuery,
        query_vector: &[f32],
    ) -> EngineResult<Vec<RetrievalCandidate>> {
        context.ensure_active()?;
        let stored = self.documents.read().await;
        let mut candidates = stored
            .iter()
            .filter(|((workspace_id, _), document)| {
                *workspace_id == context.workspace_id
                    && !document.vector.is_empty()
                    && matches_filters(document, &query.filters)
            })
            .map(|(_, document)| RetrievalCandidate {
                document: document.clone(),
                semantic_score: Some(cosine_similarity(query_vector, &document.vector)),
                lexical_score: None,
                rerank_score: None,
                fused_score: 0.0,
                reasons: Vec::new(),
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| {
            right
                .semantic_score
                .unwrap_or_default()
                .total_cmp(&left.semantic_score.unwrap_or_default())
                // Ties broken by id so equal scores do not come back in
                // HashMap order, which changes from run to run.
                .then_with(|| left.document.id.cmp(&right.document.id))
        });
        candidates.truncate(query.candidate_limit);
        Ok(candidates)
    }

    async fn lexical_search(
        &self,
        context: &EngineRequestContext,
        query: &RetrievalQuery,
    ) -> EngineResult<Vec<RetrievalCandidate>> {
        context.ensure_active()?;
        let query_tokens = tokenize(&query.text);
        let stored = self.documents.read().await;
        let mut candidates = stored
            .iter()
            .filter(|((workspace_id, _), document)| {
                *workspace_id == context.workspace_id && matches_filters(document, &query.filters)
            })
            .filter_map(|(_, document)| {
                let document_tokens = tokenize(&document.content);
                let overlap = query_tokens.intersection(&document_tokens).count();
                (overlap > 0).then(|| RetrievalCandidate {
                    document: document.clone(),
                    semantic_score: None,
                    lexical_score: Some(overlap as f32 / query_tokens.len().max(1) as f32),
                    rerank_score: None,
                    fused_score: 0.0,
                    reasons: Vec::new(),
                })
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| {
            right
                .lexical_score
                .unwrap_or_default()
                .total_cmp(&left.lexical_score.unwrap_or_default())
                // Ties broken by id so equal scores do not come back in
                // HashMap order, which changes from run to run.
                .then_with(|| left.document.id.cmp(&right.document.id))
        });
        candidates.truncate(query.candidate_limit);
        Ok(candidates)
    }

    async fn health(&self) -> EngineHealth {
        let mut health = EngineHealth::healthy(0);
        health
            .details
            .insert("documents".into(), self.documents.read().await.len().into());
        health
    }
}

fn tokenize(text: &str) -> HashSet<String> {
    text.split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_lowercase)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_embedding::DeterministicEmbeddingEngine;

    fn candidate(id: Uuid) -> RetrievalCandidate {
        RetrievalCandidate {
            document: RetrievalDocument {
                id,
                content: String::new(),
                provenance: Provenance {
                    source_id: Uuid::new_v4(),
                    ..Provenance::default()
                },
                metadata: BTreeMap::new(),
                vector: vec![],
                token_count: None,
            },
            semantic_score: Some(0.5),
            lexical_score: None,
            rerank_score: None,
            fused_score: 0.0,
            reasons: Vec::new(),
        }
    }

    #[test]
    fn agreement_between_independent_arms_wins() {
        let shared = Uuid::new_v4();
        let semantic_only = Uuid::new_v4();
        let lexical_only = Uuid::new_v4();
        let semantic = vec![candidate(semantic_only), candidate(shared)];
        let mut lexical_first = candidate(lexical_only);
        lexical_first.semantic_score = None;
        lexical_first.lexical_score = Some(0.8);
        let mut lexical_shared = candidate(shared);
        lexical_shared.semantic_score = None;
        lexical_shared.lexical_score = Some(0.7);
        let fused = reciprocal_rank_fusion(semantic, vec![lexical_first, lexical_shared], 3);
        assert_eq!(fused[0].document.id, shared);
        assert_eq!(fused[0].reasons, vec!["semantic_match", "lexical_match"]);
    }

    #[test]
    fn rejects_non_finite_score_floors() {
        let mut query = RetrievalQuery::new("bounded search");
        query.minimum_score = Some(f32::NAN);
        assert_eq!(
            query.validate().unwrap_err().code,
            "retrieval_minimum_score_invalid"
        );
    }

    #[tokio::test]
    async fn hybrid_engine_preserves_provenance_and_workspace_isolation() {
        let embedder = Arc::new(DeterministicEmbeddingEngine::new(24).unwrap());
        let backend = Arc::new(InMemoryRetrievalBackend::default());
        let engine = HybridRetrievalEngine::new(embedder.clone(), backend.clone()).unwrap();
        let context = EngineRequestContext::new(Uuid::new_v4());
        let other = EngineRequestContext::new(Uuid::new_v4());
        let texts = vec!["binary search halves a sorted array".to_owned()];
        let vector = embed_all(
            embedder.as_ref(),
            &context,
            &texts,
            EmbeddingPurpose::Document,
        )
        .await
        .unwrap()
        .remove(0)
        .values;
        let visible = Uuid::new_v4();
        let make_document = |id| RetrievalDocument {
            id,
            content: texts[0].clone(),
            provenance: Provenance {
                source_id: Uuid::new_v4(),
                source_title: Some("Algorithms".into()),
                source_kind: Some("pdf".into()),
                uri: None,
                location: Some(serde_json::json!({"page": 4})),
            },
            metadata: BTreeMap::new(),
            vector: vector.clone(),
            token_count: Some(8),
        };
        backend
            .replace(&context, vec![make_document(visible)])
            .await
            .unwrap();
        backend
            .replace(&other, vec![make_document(Uuid::new_v4())])
            .await
            .unwrap();
        let result = engine
            .retrieve(&context, RetrievalQuery::new("binary search"))
            .await
            .unwrap();
        assert_eq!(result.candidates.len(), 1);
        assert_eq!(result.candidates[0].document.id, visible);
        assert_eq!(
            result.candidates[0].document.provenance.location,
            Some(serde_json::json!({"page": 4}))
        );
    }
}
