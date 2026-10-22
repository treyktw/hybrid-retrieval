//! Embedding engine contracts.
//!
//! Providers implement [`EmbeddingEngine`]. Consumers use [`embed_all`] so
//! provider batch limits, response cardinality, dimensions, and non-finite
//! values are validated consistently.

#![forbid(unsafe_code)]

use async_trait::async_trait;
use engine_core::{
    CostClass, Engine, EngineHealth, EngineId, EngineKind, EngineMetadata, EngineRequestContext,
    EngineResult,
};
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingPurpose {
    Document,
    Query,
    Classification,
    Clustering,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    pub texts: Vec<String>,
    pub purpose: EmbeddingPurpose,
}

impl EmbeddingRequest {
    pub fn one(text: impl Into<String>, purpose: EmbeddingPurpose) -> Self {
        Self {
            texts: vec![text.into()],
            purpose,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingVector {
    pub values: Vec<f32>,
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub embeddings: Vec<EmbeddingVector>,
    pub input_count: usize,
    pub model: String,
    pub dimensions: usize,
    pub took_ms: u64,
    /// Tokens the provider says it charged for, when it says.
    ///
    /// The provider's own number rather than a count of our own, because it is
    /// the one the bill is written from. `None` where a provider does not
    /// report it, which is the difference between not knowing and zero.
    pub billed_tokens: Option<u64>,
}

#[async_trait]
pub trait EmbeddingEngine: Engine {
    async fn embed(
        &self,
        context: &EngineRequestContext,
        request: EmbeddingRequest,
    ) -> EngineResult<EmbeddingResponse>;

    fn dimensions(&self) -> usize;
    fn model(&self) -> &str;

    fn max_batch(&self) -> usize {
        32
    }
}

/// Embed an arbitrarily sized input while enforcing the provider contract.
pub async fn embed_all(
    engine: &dyn EmbeddingEngine,
    context: &EngineRequestContext,
    texts: &[String],
    purpose: EmbeddingPurpose,
) -> EngineResult<Vec<EmbeddingVector>> {
    context.ensure_active()?;
    if texts.is_empty() {
        return Ok(Vec::new());
    }
    if engine.max_batch() == 0 {
        return Err(engine_core::EngineError::invalid_input(
            "embedding_batch_limit_zero",
            "embedding engine max_batch must be greater than zero",
        ));
    }

    let mut output = Vec::with_capacity(texts.len());
    for batch in texts.chunks(engine.max_batch()) {
        context.ensure_active()?;
        let response = engine
            .embed(
                context,
                EmbeddingRequest {
                    texts: batch.to_vec(),
                    purpose,
                },
            )
            .await?;
        validate_response(engine, batch.len(), &response)?;
        output.extend(response.embeddings);
    }
    Ok(output)
}

pub fn validate_response(
    engine: &dyn EmbeddingEngine,
    expected_count: usize,
    response: &EmbeddingResponse,
) -> EngineResult<()> {
    if response.embeddings.len() != expected_count {
        return Err(engine_core::EngineError::dependency(
            "embedding_count_mismatch",
            format!(
                "embedding engine returned {} vectors for {expected_count} inputs",
                response.embeddings.len()
            ),
        ));
    }

    for (index, embedding) in response.embeddings.iter().enumerate() {
        if embedding.values.len() != engine.dimensions() {
            return Err(engine_core::EngineError::dependency(
                "embedding_dimension_mismatch",
                format!(
                    "embedding {index} has {} dimensions; expected {}",
                    embedding.values.len(),
                    engine.dimensions()
                ),
            ));
        }
        if embedding.values.iter().any(|value| !value.is_finite()) {
            return Err(engine_core::EngineError::dependency(
                "embedding_non_finite",
                format!("embedding {index} contains NaN or infinity"),
            ));
        }
        if embedding.values.iter().all(|value| *value == 0.0) {
            return Err(engine_core::EngineError::dependency(
                "embedding_zero_norm",
                format!("embedding {index} has zero magnitude"),
            ));
        }
    }
    Ok(())
}

/// Dependency-free deterministic implementation for tests and local smoke
/// environments. It is intentionally not advertised as semantically useful.
pub struct DeterministicEmbeddingEngine {
    metadata: EngineMetadata,
    model: String,
    dimensions: usize,
    max_batch: usize,
}

impl DeterministicEmbeddingEngine {
    pub fn new(dimensions: usize) -> EngineResult<Self> {
        if dimensions == 0 {
            return Err(engine_core::EngineError::invalid_input(
                "embedding_dimensions_zero",
                "embedding dimensions must be greater than zero",
            ));
        }
        let id = EngineId::new("embedding.deterministic-v1")?;
        let mut metadata =
            EngineMetadata::new(id, EngineKind::Embedding, "Deterministic embedding engine")
                .with_capabilities([
                    "batch_embeddings",
                    "document_embeddings",
                    "query_embeddings",
                ]);
        metadata.cost_class = CostClass::Free;
        metadata
            .attributes
            .insert("dimensions".into(), dimensions.into());
        Ok(Self {
            metadata,
            model: "deterministic-hash-v1".into(),
            dimensions,
            max_batch: 256,
        })
    }

    fn vector(&self, text: &str, purpose: EmbeddingPurpose) -> Vec<f32> {
        let mut output = vec![0.0_f32; self.dimensions];
        let purpose_seed = purpose as u64 + 1;
        for (position, token) in text.split_whitespace().enumerate() {
            let mut hash = 0xcbf29ce484222325_u64 ^ purpose_seed;
            for byte in token.bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x100000001b3);
            }
            let index = (hash as usize) % self.dimensions;
            let sign = if hash & 1 == 0 { 1.0 } else { -1.0 };
            output[index] += sign * (1.0 + 1.0 / (position as f32 + 1.0));
        }
        normalize(&mut output);
        output
    }
}

#[async_trait]
impl Engine for DeterministicEmbeddingEngine {
    fn metadata(&self) -> EngineMetadata {
        self.metadata.clone()
    }

    async fn health(&self) -> EngineHealth {
        EngineHealth::healthy(0)
    }
}

#[async_trait]
impl EmbeddingEngine for DeterministicEmbeddingEngine {
    async fn embed(
        &self,
        context: &EngineRequestContext,
        request: EmbeddingRequest,
    ) -> EngineResult<EmbeddingResponse> {
        context.ensure_active()?;
        if request.texts.len() > self.max_batch {
            return Err(engine_core::EngineError::invalid_input(
                "embedding_batch_too_large",
                format!(
                    "batch contains {}; maximum is {}",
                    request.texts.len(),
                    self.max_batch
                ),
            ));
        }
        let started = Instant::now();
        let embeddings = request
            .texts
            .iter()
            .map(|text| EmbeddingVector {
                values: self.vector(text, request.purpose),
                model: self.model.clone(),
            })
            .collect::<Vec<_>>();
        Ok(EmbeddingResponse {
            input_count: embeddings.len(),
            embeddings,
            model: self.model.clone(),
            dimensions: self.dimensions,
            took_ms: started.elapsed().as_millis() as u64,
            billed_tokens: None,
        })
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn max_batch(&self) -> usize {
        self.max_batch
    }
}

fn normalize(values: &mut [f32]) {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        values.iter_mut().for_each(|value| *value /= norm);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[tokio::test]
    async fn deterministic_engine_is_stable_and_normalized() {
        let engine = DeterministicEmbeddingEngine::new(32).unwrap();
        let context = EngineRequestContext::new(Uuid::new_v4());
        let input = vec!["binary search halves the search space".to_owned()];
        let first = embed_all(&engine, &context, &input, EmbeddingPurpose::Document)
            .await
            .unwrap();
        let second = embed_all(&engine, &context, &input, EmbeddingPurpose::Document)
            .await
            .unwrap();
        assert_eq!(first, second);
        let norm = first[0]
            .values
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        assert!((norm - 1.0).abs() < 0.0001);
    }

    #[tokio::test]
    async fn empty_input_does_not_call_the_engine() {
        let engine = DeterministicEmbeddingEngine::new(8).unwrap();
        let context = EngineRequestContext::new(Uuid::new_v4());
        assert!(
            embed_all(&engine, &context, &[], EmbeddingPurpose::Query)
                .await
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn zero_norm_provider_responses_are_rejected() {
        let engine = DeterministicEmbeddingEngine::new(3).unwrap();
        let response = EmbeddingResponse {
            embeddings: vec![EmbeddingVector {
                values: vec![0.0; 3],
                model: "broken-provider".into(),
            }],
            input_count: 1,
            model: "broken-provider".into(),
            dimensions: 3,
            took_ms: 0,
            billed_tokens: None,
        };
        assert_eq!(
            validate_response(&engine, 1, &response).unwrap_err().code,
            "embedding_zero_norm"
        );
    }
}
