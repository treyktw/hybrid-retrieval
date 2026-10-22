//! Vector storage is separate from embedding generation. This crate defines
//! the storage contract and provides a workspace-isolated in-memory reference
//! backend used for contract tests and development.

#![forbid(unsafe_code)]

use async_trait::async_trait;
use engine_core::{
    Engine, EngineHealth, EngineId, EngineKind, EngineMetadata, EngineRequestContext, EngineResult,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use uuid::Uuid;

type VectorKey = (Uuid, String, Uuid);
type VectorStore = HashMap<VectorKey, VectorPoint>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorPoint {
    pub id: Uuid,
    pub vector: Vec<f32>,
    pub namespace: String,
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetadataFilter {
    #[serde(default)]
    pub equals: BTreeMap<String, Value>,
}

impl MetadataFilter {
    fn matches(&self, metadata: &BTreeMap<String, Value>) -> bool {
        self.equals
            .iter()
            .all(|(key, value)| metadata.get(key) == Some(value))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorQuery {
    pub vector: Vec<f32>,
    pub namespace: String,
    pub limit: usize,
    pub minimum_score: Option<f32>,
    #[serde(default)]
    pub filter: MetadataFilter,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorMatch {
    pub id: Uuid,
    pub score: f32,
    pub namespace: String,
    pub metadata: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchResult {
    pub matches: Vec<VectorMatch>,
    pub examined: usize,
    pub took_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteVectors {
    pub namespace: String,
    #[serde(default)]
    pub ids: Vec<Uuid>,
    pub delete_namespace: bool,
}

#[async_trait]
pub trait VectorEngine: Engine {
    fn dimensions(&self) -> usize;

    async fn upsert(
        &self,
        context: &EngineRequestContext,
        points: Vec<VectorPoint>,
    ) -> EngineResult<usize>;

    async fn search(
        &self,
        context: &EngineRequestContext,
        query: VectorQuery,
    ) -> EngineResult<VectorSearchResult>;

    async fn delete(
        &self,
        context: &EngineRequestContext,
        request: DeleteVectors,
    ) -> EngineResult<usize>;
}

#[derive(Clone)]
pub struct InMemoryVectorEngine {
    metadata: EngineMetadata,
    dimensions: usize,
    // Workspace is part of the key so a caller cannot accidentally search a
    // namespace belonging to another workspace.
    points: Arc<RwLock<VectorStore>>,
}

impl InMemoryVectorEngine {
    pub fn new(dimensions: usize) -> EngineResult<Self> {
        if dimensions == 0 {
            return Err(engine_core::EngineError::invalid_input(
                "vector_dimensions_zero",
                "vector dimensions must be greater than zero",
            ));
        }
        let id = EngineId::new("vector.memory-v1")?;
        let mut metadata = EngineMetadata::new(id, EngineKind::Vector, "In-memory vector engine")
            .with_capabilities([
                "upsert",
                "cosine_search",
                "metadata_filter",
                "namespace_delete",
            ]);
        metadata
            .attributes
            .insert("dimensions".into(), dimensions.into());
        Ok(Self {
            metadata,
            dimensions,
            points: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    fn validate_vector(&self, vector: &[f32]) -> EngineResult<()> {
        if vector.len() != self.dimensions {
            return Err(engine_core::EngineError::invalid_input(
                "vector_dimension_mismatch",
                format!(
                    "vector has {} dimensions; expected {}",
                    vector.len(),
                    self.dimensions
                ),
            ));
        }
        if vector.iter().any(|value| !value.is_finite()) {
            return Err(engine_core::EngineError::invalid_input(
                "vector_non_finite",
                "vector contains NaN or infinity",
            ));
        }
        if vector.iter().all(|value| *value == 0.0) {
            return Err(engine_core::EngineError::invalid_input(
                "vector_zero_norm",
                "cosine similarity is undefined for an all-zero vector",
            ));
        }
        Ok(())
    }
}

#[async_trait]
impl Engine for InMemoryVectorEngine {
    fn metadata(&self) -> EngineMetadata {
        self.metadata.clone()
    }

    async fn health(&self) -> EngineHealth {
        let started = Instant::now();
        let count = self.points.read().await.len();
        let mut health = EngineHealth::healthy(started.elapsed().as_millis() as u64);
        health.details.insert("points".into(), count.into());
        health
    }
}

#[async_trait]
impl VectorEngine for InMemoryVectorEngine {
    fn dimensions(&self) -> usize {
        self.dimensions
    }

    async fn upsert(
        &self,
        context: &EngineRequestContext,
        points: Vec<VectorPoint>,
    ) -> EngineResult<usize> {
        context.ensure_active()?;
        if points.len() > 5_000 {
            return Err(engine_core::EngineError::invalid_input(
                "vector_batch_too_large",
                "vector upsert accepts at most 5000 points per transaction",
            ));
        }
        for point in &points {
            self.validate_vector(&point.vector)?;
            if point.namespace.trim().is_empty() {
                return Err(engine_core::EngineError::invalid_input(
                    "vector_namespace_empty",
                    "vector namespace must not be empty",
                ));
            }
        }
        let count = points.len();
        let mut stored = self.points.write().await;
        for point in points {
            stored.insert(
                (context.workspace_id, point.namespace.clone(), point.id),
                point,
            );
        }
        Ok(count)
    }

    async fn search(
        &self,
        context: &EngineRequestContext,
        query: VectorQuery,
    ) -> EngineResult<VectorSearchResult> {
        context.ensure_active()?;
        self.validate_vector(&query.vector)?;
        if query.namespace.trim().is_empty() {
            return Err(engine_core::EngineError::invalid_input(
                "vector_namespace_empty",
                "vector namespace must not be empty",
            ));
        }
        if query.limit == 0 || query.limit > 1_000 {
            return Err(engine_core::EngineError::invalid_input(
                "vector_limit_invalid",
                "vector search limit must be between 1 and 1000",
            ));
        }
        if query
            .minimum_score
            .is_some_and(|score| !score.is_finite() || !(-1.0..=1.0).contains(&score))
        {
            return Err(engine_core::EngineError::invalid_input(
                "vector_minimum_score_invalid",
                "minimum score must be finite and between -1 and 1",
            ));
        }
        let started = Instant::now();
        let stored = self.points.read().await;
        let mut examined = 0;
        let mut matches = stored
            .iter()
            .filter(|((workspace_id, namespace, _), point)| {
                *workspace_id == context.workspace_id
                    && namespace == &query.namespace
                    && query.filter.matches(&point.metadata)
            })
            .filter_map(|(_, point)| {
                examined += 1;
                let score = cosine_similarity(&query.vector, &point.vector);
                (query.minimum_score.is_none_or(|minimum| score >= minimum)).then(|| VectorMatch {
                    id: point.id,
                    score,
                    namespace: point.namespace.clone(),
                    metadata: point.metadata.clone(),
                })
            })
            .collect::<Vec<_>>();
        matches.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.id.cmp(&right.id))
        });
        matches.truncate(query.limit);
        Ok(VectorSearchResult {
            matches,
            examined,
            took_ms: started.elapsed().as_millis() as u64,
        })
    }

    async fn delete(
        &self,
        context: &EngineRequestContext,
        request: DeleteVectors,
    ) -> EngineResult<usize> {
        context.ensure_active()?;
        if request.namespace.trim().is_empty() {
            return Err(engine_core::EngineError::invalid_input(
                "vector_namespace_empty",
                "vector namespace must not be empty",
            ));
        }
        if !request.delete_namespace && request.ids.is_empty() {
            return Ok(0);
        }
        let before = self.points.read().await.len();
        let mut stored = self.points.write().await;
        stored.retain(|(workspace_id, namespace, id), _| {
            !(*workspace_id == context.workspace_id
                && namespace == &request.namespace
                && (request.delete_namespace || request.ids.contains(id)))
        });
        Ok(before - stored.len())
    }
}

pub fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    let dot = left
        .iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum::<f32>();
    let left_norm = left.iter().map(|value| value * value).sum::<f32>().sqrt();
    let right_norm = right.iter().map(|value| value * value).sum::<f32>().sqrt();
    if left_norm == 0.0 || right_norm == 0.0 {
        0.0
    } else {
        (dot / (left_norm * right_norm)).clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(id: Uuid, vector: Vec<f32>, course: &str) -> VectorPoint {
        VectorPoint {
            id,
            vector,
            namespace: "knowledge".into(),
            metadata: BTreeMap::from([("course".into(), course.into())]),
        }
    }

    #[tokio::test]
    async fn search_is_ranked_filtered_and_workspace_isolated() {
        let engine = InMemoryVectorEngine::new(3).unwrap();
        let first_workspace = EngineRequestContext::new(Uuid::new_v4());
        let second_workspace = EngineRequestContext::new(Uuid::new_v4());
        let close = Uuid::new_v4();
        let hidden = Uuid::new_v4();
        engine
            .upsert(
                &first_workspace,
                vec![
                    point(close, vec![1.0, 0.0, 0.0], "math"),
                    point(Uuid::new_v4(), vec![0.0, 1.0, 0.0], "history"),
                ],
            )
            .await
            .unwrap();
        engine
            .upsert(
                &second_workspace,
                vec![point(hidden, vec![1.0, 0.0, 0.0], "math")],
            )
            .await
            .unwrap();

        let result = engine
            .search(
                &first_workspace,
                VectorQuery {
                    vector: vec![1.0, 0.0, 0.0],
                    namespace: "knowledge".into(),
                    limit: 10,
                    minimum_score: None,
                    filter: MetadataFilter {
                        equals: BTreeMap::from([("course".into(), "math".into())]),
                    },
                },
            )
            .await
            .unwrap();
        assert_eq!(result.matches.len(), 1);
        assert_eq!(result.matches[0].id, close);
        assert_ne!(result.matches[0].id, hidden);
    }

    #[tokio::test]
    async fn rejects_wrong_dimensions_without_partial_writes() {
        let engine = InMemoryVectorEngine::new(2).unwrap();
        let context = EngineRequestContext::new(Uuid::new_v4());
        let result = engine
            .upsert(
                &context,
                vec![point(Uuid::new_v4(), vec![1.0, 2.0, 3.0], "math")],
            )
            .await;
        assert!(result.is_err());
        assert_eq!(engine.health().await.details["points"], 0);
    }

    #[tokio::test]
    async fn rejects_zero_norm_vectors_and_invalid_score_floors() {
        let engine = InMemoryVectorEngine::new(2).unwrap();
        let context = EngineRequestContext::new(Uuid::new_v4());
        let error = engine
            .upsert(
                &context,
                vec![point(Uuid::new_v4(), vec![0.0, 0.0], "math")],
            )
            .await
            .unwrap_err();
        assert_eq!(error.code, "vector_zero_norm");

        let error = engine
            .search(
                &context,
                VectorQuery {
                    vector: vec![1.0, 0.0],
                    namespace: "knowledge".into(),
                    limit: 1,
                    minimum_score: Some(f32::NAN),
                    filter: MetadataFilter::default(),
                },
            )
            .await
            .unwrap_err();
        assert_eq!(error.code, "vector_minimum_score_invalid");
    }
}
