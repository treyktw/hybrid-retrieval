//! A ceiling on what embedding is allowed to cost.
//!
//! Wraps any embedder rather than living inside one, because the limit is about
//! money rather than about a particular provider's API. The spend is kept in
//! Postgres so it survives a restart -- a cap that resets when the worker
//! reboots is not a cap, and a crash loop would be the most expensive thing the
//! system could do.
//!
//! This is a spend ceiling, not a rate limiter. They are often asked for in the
//! same breath and do different jobs: a rate limiter smooths requests to avoid
//! being throttled, and would happily spend an unlimited amount slowly. This
//! refuses to spend past a number.

use async_trait::async_trait;
use engine_core::{
    Engine, EngineError, EngineHealth, EngineMetadata, EngineRequestContext, EngineResult,
};
use engine_embedding::{EmbeddingEngine, EmbeddingRequest, EmbeddingResponse};
use sqlx::PgPool;
use std::sync::Arc;

/// Roughly how many characters make a token, for providers that do not say.
///
/// Only used to estimate what an unreported request cost. Deliberately
/// pessimistic: undercounting spends real money past the ceiling, while
/// overcounting only stops slightly early.
const CHARS_PER_TOKEN: f64 = 3.5;

pub struct BudgetedEmbedder {
    inner: Arc<dyn EmbeddingEngine>,
    pool: PgPool,
    provider: String,
    model: String,
    /// Dollars per million tokens for this model.
    price_per_million: f64,
    budget_usd: f64,
}

impl BudgetedEmbedder {
    pub fn new(
        inner: Arc<dyn EmbeddingEngine>,
        pool: PgPool,
        provider: impl Into<String>,
        model: impl Into<String>,
        price_per_million: f64,
        budget_usd: f64,
    ) -> Self {
        Self {
            inner,
            pool,
            provider: provider.into(),
            model: model.into(),
            price_per_million,
            budget_usd,
        }
    }

    /// Dollars for a number of tokens at this model's rate.
    pub fn cost_of(&self, tokens: u64) -> f64 {
        cost(tokens, self.price_per_million)
    }

    async fn spent_tokens(&self) -> u64 {
        sqlx::query_scalar::<_, i64>(
            "SELECT tokens FROM embedding_spend WHERE provider = $1 AND model = $2",
        )
        .bind(&self.provider)
        .bind(&self.model)
        .fetch_optional(&self.pool)
        .await
        .ok()
        .flatten()
        .unwrap_or(0)
        .max(0) as u64
    }

    async fn record(&self, tokens: u64) {
        // Advisory in the sense that failing to record must not fail the work
        // that already happened -- but logged loudly, because an unrecorded
        // spend is spend the ceiling cannot see.
        if let Err(error) = sqlx::query(
            "INSERT INTO embedding_spend (provider, model, tokens, requests) \
             VALUES ($1, $2, $3, 1) \
             ON CONFLICT (provider, model) DO UPDATE \
             SET tokens = embedding_spend.tokens + EXCLUDED.tokens, \
                 requests = embedding_spend.requests + 1, \
                 updated_at = now()",
        )
        .bind(&self.provider)
        .bind(&self.model)
        .bind(tokens as i64)
        .execute(&self.pool)
        .await
        {
            tracing::error!(%error, tokens, "could not record embedding spend");
        }
    }
}

/// Dollars for a number of tokens at a price per million.
pub fn cost(tokens: u64, price_per_million: f64) -> f64 {
    (tokens as f64 / 1_000_000.0) * price_per_million
}

/// What a request will cost if the provider does not report it.
pub fn estimate_tokens(texts: &[String]) -> u64 {
    let characters: usize = texts.iter().map(|text| text.len()).sum();
    (characters as f64 / CHARS_PER_TOKEN).ceil() as u64
}

#[async_trait]
impl Engine for BudgetedEmbedder {
    fn metadata(&self) -> EngineMetadata {
        self.inner.metadata()
    }

    async fn health(&self) -> EngineHealth {
        self.inner.health().await
    }
}

#[async_trait]
impl EmbeddingEngine for BudgetedEmbedder {
    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    fn model(&self) -> &str {
        self.inner.model()
    }

    fn max_batch(&self) -> usize {
        self.inner.max_batch()
    }

    async fn embed(
        &self,
        context: &EngineRequestContext,
        request: EmbeddingRequest,
    ) -> EngineResult<EmbeddingResponse> {
        if self.budget_usd <= 0.0 {
            return self.inner.embed(context, request).await;
        }

        let spent = self.spent_tokens().await;
        let already = self.cost_of(spent);

        // Checked before the request rather than after. Refusing to start is
        // the only point at which the money has not been spent yet.
        if already >= self.budget_usd {
            return Err(EngineError::dependency(
                "embedding_budget_exhausted",
                format!(
                    "the embedding budget of ${:.2} is spent (${:.4} across {spent} tokens on {}); \
                     raise EMBEDDING_BUDGET_USD to continue",
                    self.budget_usd, already, self.model
                ),
            ));
        }

        let texts = request.texts.clone();
        let response = self.inner.embed(context, request).await?;

        // The provider's own figure where there is one, an estimate where there
        // is not. `None` means it did not say, which is not the same as zero --
        // treating it as zero would let an unreporting provider spend forever.
        let tokens = response
            .billed_tokens
            .unwrap_or_else(|| estimate_tokens(&texts));
        self.record(tokens).await;

        let after = self.cost_of(spent + tokens);
        if after >= self.budget_usd {
            tracing::warn!(
                model = %self.model,
                spent_usd = after,
                budget_usd = self.budget_usd,
                "the embedding budget is now spent; further requests will be refused"
            );
        }

        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rate that matters here: $0.13 per million for
    /// text-embedding-3-large.
    const LARGE: f64 = 0.13;

    #[test]
    fn ten_dollars_is_a_lot_of_tokens() {
        // Roughly 77 million, which at this corpus's rate is thousands of
        // hours of video. Worth asserting so a units slip is visible.
        let tokens = 76_900_000;
        assert!((cost(tokens, LARGE) - 9.997).abs() < 0.01);
    }

    #[test]
    fn a_small_corpus_costs_almost_nothing() {
        // The whole index at the time of writing: 2374 tokens.
        assert!(cost(2_374, LARGE) < 0.001);
    }

    #[test]
    fn nothing_spent_is_nothing_owed() {
        assert_eq!(cost(0, LARGE), 0.0);
    }

    /// Undercounting spends past the ceiling; overcounting only stops early.
    #[test]
    fn the_estimate_does_not_undercount_ordinary_prose() {
        let text = "The column space of a matrix is the span of its columns.".to_string();
        // 56 characters. Real tokenisers put this near 12; the estimate must
        // not come out below that.
        let estimated = estimate_tokens(&[text]);
        assert!(estimated >= 12, "estimated {estimated}, which is too few");
    }

    #[test]
    fn the_estimate_adds_up_across_a_batch() {
        let one = estimate_tokens(&["hello world".to_string()]);
        let two = estimate_tokens(&["hello world".to_string(), "hello world".to_string()]);
        assert!(two > one);
    }
}
