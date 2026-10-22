//! A feature-hashing embedder that is crude but actually comparable.
//!
//! `engine_embedding::DeterministicEmbeddingEngine` is a contract-test double:
//! it mixes the embedding *purpose* into every hash, so a query vector and a
//! document vector for the same words land in unrelated buckets. That is fine
//! for proving batching and validation, and useless for showing what hybrid
//! retrieval does.
//!
//! This one hashes lower-cased words and character trigrams into signed
//! buckets, the same way for every purpose. Shared vocabulary and shared word
//! fragments ("extract", "extraction", "extracts") therefore raise cosine
//! similarity, which is enough to make the semantic arm disagree with the
//! keyword arm in interesting ways without a model or an API key.

use async_trait::async_trait;
use engine_core::{
    CostClass, Engine, EngineError, EngineHealth, EngineId, EngineKind, EngineMetadata,
    EngineRequestContext, EngineResult,
};
use engine_embedding::{EmbeddingEngine, EmbeddingRequest, EmbeddingResponse, EmbeddingVector};
use std::time::Instant;

pub struct HashingEmbeddingEngine {
    metadata: EngineMetadata,
    model: String,
    dimensions: usize,
}

impl HashingEmbeddingEngine {
    pub fn new(dimensions: usize) -> EngineResult<Self> {
        if dimensions == 0 {
            return Err(EngineError::invalid_input(
                "embedding_dimensions_zero",
                "embedding dimensions must be greater than zero",
            ));
        }
        let id = EngineId::new("embedding.feature-hashing-v1")?;
        let mut metadata = EngineMetadata::new(
            id,
            EngineKind::Embedding,
            "Feature-hashing embedding engine",
        )
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
            model: "feature-hashing-v1".into(),
            dimensions,
        })
    }

    fn vector(&self, text: &str) -> Vec<f32> {
        let mut output = vec![0.0_f32; self.dimensions];
        let words = text
            .split(|character: char| !character.is_alphanumeric())
            .filter(|word| word.len() > 2)
            .map(str::to_lowercase)
            .collect::<Vec<_>>();
        for word in &words {
            self.add_feature(&mut output, word.as_bytes(), 1.0);
            let padded = format!("<{word}>");
            let characters = padded.chars().collect::<Vec<_>>();
            for window in characters.windows(3) {
                let trigram = window.iter().collect::<String>();
                self.add_feature(&mut output, trigram.as_bytes(), 0.35);
            }
        }
        let norm = output.iter().map(|value| value * value).sum::<f32>().sqrt();
        if norm > 0.0 {
            output.iter_mut().for_each(|value| *value /= norm);
        } else {
            // An input with no usable words still needs a valid, non-zero
            // vector or the embedding contract rejects the whole batch.
            output[0] = 1.0;
        }
        output
    }

    fn add_feature(&self, output: &mut [f32], feature: &[u8], weight: f32) {
        // FNV-1a: stable across runs and platforms, which the demo relies on.
        let mut hash = 0xcbf29ce484222325_u64;
        for byte in feature {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        let index = (hash % self.dimensions as u64) as usize;
        let sign = if (hash >> 63) == 0 { 1.0 } else { -1.0 };
        output[index] += sign * weight;
    }
}

#[async_trait]
impl Engine for HashingEmbeddingEngine {
    fn metadata(&self) -> EngineMetadata {
        self.metadata.clone()
    }

    async fn health(&self) -> EngineHealth {
        EngineHealth::healthy(0)
    }
}

#[async_trait]
impl EmbeddingEngine for HashingEmbeddingEngine {
    async fn embed(
        &self,
        context: &EngineRequestContext,
        request: EmbeddingRequest,
    ) -> EngineResult<EmbeddingResponse> {
        context.ensure_active()?;
        let started = Instant::now();
        let embeddings = request
            .texts
            .iter()
            .map(|text| EmbeddingVector {
                values: self.vector(text),
                model: self.model.clone(),
            })
            .collect::<Vec<_>>();
        Ok(EmbeddingResponse {
            input_count: embeddings.len(),
            embeddings,
            model: self.model.clone(),
            dimensions: self.dimensions,
            took_ms: started.elapsed().as_millis() as u64,
            billed_tokens: Some(0),
        })
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn max_batch(&self) -> usize {
        256
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_embedding::{EmbeddingPurpose, embed_all};
    use engine_vector::cosine_similarity;
    use uuid::Uuid;

    #[tokio::test]
    async fn related_text_scores_higher_than_unrelated_text() {
        let engine = HashingEmbeddingEngine::new(512).unwrap();
        let context = EngineRequestContext::new(Uuid::new_v4());
        let texts = [
            "bitter espresso from over extraction",
            "espresso turns bitter when it over-extracts",
            "store beans in an airtight container",
        ]
        .map(String::from);
        let query = embed_all(&engine, &context, &texts[..1], EmbeddingPurpose::Query)
            .await
            .unwrap();
        let documents = embed_all(&engine, &context, &texts[1..], EmbeddingPurpose::Document)
            .await
            .unwrap();
        let related = cosine_similarity(&query[0].values, &documents[0].values);
        let unrelated = cosine_similarity(&query[0].values, &documents[1].values);
        assert!(related > unrelated + 0.2, "{related} vs {unrelated}");
    }

    #[tokio::test]
    async fn empty_text_still_produces_a_valid_vector() {
        let engine = HashingEmbeddingEngine::new(16).unwrap();
        let context = EngineRequestContext::new(Uuid::new_v4());
        let vectors = embed_all(&engine, &context, &["?!".into()], EmbeddingPurpose::Query)
            .await
            .unwrap();
        assert_eq!(vectors[0].values.len(), 16);
    }
}
