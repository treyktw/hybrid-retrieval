//! Postgres adapters for the engine contracts.
//!
//! [`PostgresVectorEngine`] implements `engine_vector::VectorEngine` on top of
//! pgvector, and [`BudgetedEmbedder`] wraps any `EmbeddingEngine` with a spend
//! ceiling that survives restarts. Both need the schema in `migrations/`,
//! which [`migrate`] applies.

#![forbid(unsafe_code)]

mod budget;
mod vector;

pub use budget::{BudgetedEmbedder, cost, estimate_tokens};
pub use vector::PostgresVectorEngine;

/// The schema both adapters expect. Idempotent: every statement is
/// `IF NOT EXISTS`, so running it on every start is safe.
pub const SCHEMA: &str = include_str!("../migrations/0001_vectors.sql");

/// Apply [`SCHEMA`]. Requires a role allowed to `CREATE EXTENSION vector`, or
/// a database where the extension is already installed.
pub async fn migrate(pool: &sqlx::PgPool) -> Result<(), sqlx::Error> {
    sqlx::raw_sql(SCHEMA).execute(pool).await?;
    Ok(())
}
