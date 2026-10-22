//! Database contract tests for the durable adapters.
//!
//! These are ignored by default because they require a PostgreSQL database
//! with pgvector >= 0.7 (for `halfvec`). Run them with:
//! `TEST_DATABASE_URL=... cargo test -p engine-postgres -- --ignored --test-threads=1`
//!
//! `docker compose up -d` in the repository root starts a suitable database on
//! port 54329; see the README.

use anyhow::{Context, Result};
use engine_core::{Engine, EngineRequestContext, HealthStatus};
use engine_embedding::{
    DeterministicEmbeddingEngine, EmbeddingEngine, EmbeddingPurpose, EmbeddingRequest,
};
use engine_postgres::{BudgetedEmbedder, PostgresVectorEngine, migrate};
use engine_vector::{DeleteVectors, MetadataFilter, VectorEngine, VectorPoint, VectorQuery};
use serde_json::json;
use sqlx::{PgPool, postgres::PgPoolOptions};
use std::collections::BTreeMap;
use std::sync::Arc;
use uuid::Uuid;

const MODEL: &str = "text-embedding-3-large";

async fn test_pool() -> Result<PgPool> {
    let database_url = std::env::var("TEST_DATABASE_URL")
        .context("TEST_DATABASE_URL is required for ignored Postgres engine tests")?;
    let pool = PgPoolOptions::new()
        .max_connections(4)
        .connect(&database_url)
        .await?;
    migrate(&pool).await?;
    Ok(pool)
}

fn unit_vector(index: usize) -> Vec<f32> {
    let mut vector = vec![0.0; 3_072];
    vector[index] = 1.0;
    vector
}

#[tokio::test]
#[ignore = "requires TEST_DATABASE_URL"]
async fn postgres_vector_contract_covers_ranking_filters_isolation_and_delete() -> Result<()> {
    let pool = test_pool().await?;
    let context = EngineRequestContext::new(Uuid::new_v4());
    let second_workspace = Uuid::new_v4();
    let engine = PostgresVectorEngine::new(pool, 3_072, MODEL)?;
    let primary_id = Uuid::new_v4();
    let secondary_id = Uuid::new_v4();
    let metadata = |topic: &str| BTreeMap::from([("topic".into(), json!(topic))]);

    engine
        .upsert(
            &context,
            vec![
                VectorPoint {
                    id: primary_id,
                    vector: unit_vector(0),
                    namespace: "knowledge".into(),
                    metadata: metadata("brewing"),
                },
                VectorPoint {
                    id: secondary_id,
                    vector: unit_vector(1),
                    namespace: "knowledge".into(),
                    metadata: metadata("roasting"),
                },
            ],
        )
        .await?;

    let result = engine
        .search(
            &context,
            VectorQuery {
                vector: unit_vector(0),
                namespace: "knowledge".into(),
                limit: 10,
                minimum_score: Some(0.9),
                filter: MetadataFilter {
                    equals: metadata("brewing"),
                },
            },
        )
        .await?;
    assert_eq!(result.matches.len(), 1);
    assert_eq!(result.matches[0].id, primary_id);
    assert!(result.matches[0].score > 0.99);

    let mut isolated_context = context.clone();
    isolated_context.workspace_id = second_workspace;
    let isolated = engine
        .search(
            &isolated_context,
            VectorQuery {
                vector: unit_vector(0),
                namespace: "knowledge".into(),
                limit: 10,
                minimum_score: None,
                filter: MetadataFilter::default(),
            },
        )
        .await?;
    assert!(isolated.matches.is_empty());

    assert_eq!(
        engine
            .delete(
                &context,
                DeleteVectors {
                    namespace: "knowledge".into(),
                    ids: vec![primary_id],
                    delete_namespace: false,
                },
            )
            .await?,
        1
    );
    assert_eq!(engine.health().await.status, HealthStatus::Healthy);
    Ok(())
}

#[tokio::test]
#[ignore = "requires TEST_DATABASE_URL"]
async fn postgres_vector_search_only_sees_the_asking_models_vectors() -> Result<()> {
    let pool = test_pool().await?;
    let context = EngineRequestContext::new(Uuid::new_v4());
    let writer = PostgresVectorEngine::new(pool.clone(), 3_072, MODEL)?;
    let other_model = PostgresVectorEngine::new(pool, 3_072, "some-other-model")?;
    writer
        .upsert(
            &context,
            vec![VectorPoint {
                id: Uuid::new_v4(),
                vector: unit_vector(5),
                namespace: "knowledge".into(),
                metadata: BTreeMap::new(),
            }],
        )
        .await?;
    let query = VectorQuery {
        vector: unit_vector(5),
        namespace: "knowledge".into(),
        limit: 10,
        minimum_score: None,
        filter: MetadataFilter::default(),
    };
    assert_eq!(
        writer.search(&context, query.clone()).await?.matches.len(),
        1
    );
    // Same width, different model: comparing them would be confident noise.
    assert!(
        other_model
            .search(&context, query)
            .await?
            .matches
            .is_empty()
    );
    Ok(())
}

#[tokio::test]
#[ignore = "requires TEST_DATABASE_URL"]
async fn budget_refuses_once_the_persisted_spend_reaches_the_ceiling() -> Result<()> {
    let pool = test_pool().await?;
    // A unique provider per run so the persisted spend starts at zero.
    let provider = format!("test-{}", Uuid::new_v4());
    let inner: Arc<dyn EmbeddingEngine> = Arc::new(DeterministicEmbeddingEngine::new(8)?);
    // $1,000,000 per million tokens: one token costs a dollar, so a $5 budget
    // is spent by the first request of a few dozen characters.
    let budgeted = BudgetedEmbedder::new(
        inner,
        pool.clone(),
        &provider,
        "fake-model",
        1_000_000.0,
        5.0,
    );
    let context = EngineRequestContext::new(Uuid::new_v4());
    let request = EmbeddingRequest::one(
        "a request long enough to cost more than five tokens",
        EmbeddingPurpose::Document,
    );
    budgeted.embed(&context, request.clone()).await?;

    // A fresh wrapper over the same pool: the spend outlived the instance.
    let inner: Arc<dyn EmbeddingEngine> = Arc::new(DeterministicEmbeddingEngine::new(8)?);
    let restarted = BudgetedEmbedder::new(inner, pool, &provider, "fake-model", 1_000_000.0, 5.0);
    let error = restarted.embed(&context, request).await.unwrap_err();
    assert_eq!(error.code, "embedding_budget_exhausted");
    Ok(())
}
