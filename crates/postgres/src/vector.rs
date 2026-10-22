use async_trait::async_trait;
use engine_core::{
    CostClass, Engine, EngineError, EngineErrorKind, EngineHealth, EngineId, EngineKind,
    EngineMetadata, EngineRequestContext, EngineResult, EngineRuntime, HealthStatus,
};
use engine_vector::{
    DeleteVectors, VectorEngine, VectorMatch, VectorPoint, VectorQuery, VectorSearchResult,
};
use serde_json::Value;
use sqlx::{AssertSqlSafe, PgPool, Row};
use std::collections::BTreeMap;
use std::time::Instant;

#[derive(Clone)]
pub struct PostgresVectorEngine {
    pool: PgPool,
    dimensions: usize,
    /// The model whose vectors this instance reads and writes.
    embedding_model: String,
    metadata: EngineMetadata,
}

impl PostgresVectorEngine {
    pub fn new(
        pool: PgPool,
        dimensions: usize,
        embedding_model: impl Into<String>,
    ) -> EngineResult<Self> {
        if dimensions == 0 {
            return Err(EngineError::invalid_input(
                "vector_dimensions_zero",
                "vector dimensions must be greater than zero",
            ));
        }
        let mut metadata = EngineMetadata::new(
            EngineId::new("vector.postgres-pgvector")?,
            EngineKind::Vector,
            "Postgres pgvector engine",
        )
        .with_capabilities([
            "durable_upsert",
            "cosine_search",
            "metadata_filter",
            "namespace_delete",
            "workspace_isolation",
            "transactional_batch",
        ]);
        metadata.runtime = EngineRuntime::InProcess;
        metadata.cost_class = CostClass::Low;
        metadata
            .attributes
            .insert("dimensions".into(), dimensions.into());
        metadata
            .attributes
            .insert("backend".into(), "pgvector".into());
        Ok(Self {
            pool,
            embedding_model: embedding_model.into(),
            dimensions,
            metadata,
        })
    }

    fn validate_vector(&self, vector: &[f32]) -> EngineResult<()> {
        if vector.len() != self.dimensions {
            return Err(EngineError::invalid_input(
                "vector_dimension_mismatch",
                format!(
                    "vector has {} dimensions; expected {}",
                    vector.len(),
                    self.dimensions
                ),
            ));
        }
        if vector.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::invalid_input(
                "vector_non_finite",
                "vector contains NaN or infinity",
            ));
        }
        if vector.iter().all(|value| *value == 0.0) {
            return Err(EngineError::invalid_input(
                "vector_zero_norm",
                "cosine similarity is undefined for an all-zero vector",
            ));
        }
        Ok(())
    }

    fn database_error(code: &'static str, error: sqlx::Error) -> EngineError {
        EngineError::dependency(code, error.to_string())
    }
}

fn encode_vector(values: &[f32]) -> String {
    let mut output = String::with_capacity(values.len() * 9 + 2);
    output.push('[');
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        output.push_str(&value.to_string());
    }
    output.push(']');
    output
}

#[async_trait]
impl Engine for PostgresVectorEngine {
    fn metadata(&self) -> EngineMetadata {
        self.metadata.clone()
    }

    async fn health(&self) -> EngineHealth {
        let started = Instant::now();
        match sqlx::query_scalar::<_, i64>("SELECT count(*) FROM engine_vectors")
            .fetch_one(&self.pool)
            .await
        {
            Ok(count) => {
                let mut health = EngineHealth::healthy(started.elapsed().as_millis() as u64);
                health.details.insert("points".into(), count.into());
                health
            }
            Err(error) => EngineHealth {
                status: HealthStatus::Unhealthy,
                checked_at: chrono::Utc::now(),
                latency_ms: started.elapsed().as_millis() as u64,
                message: Some(error.to_string()),
                details: BTreeMap::new(),
            },
        }
    }
}

#[async_trait]
impl VectorEngine for PostgresVectorEngine {
    fn dimensions(&self) -> usize {
        self.dimensions
    }

    #[tracing::instrument(name = "engine.vector.upsert", skip_all, fields(request_id = %context.request_id, workspace_id = %context.workspace_id, point_count = points.len()))]
    async fn upsert(
        &self,
        context: &EngineRequestContext,
        points: Vec<VectorPoint>,
    ) -> EngineResult<usize> {
        context.ensure_active()?;
        if points.len() > 5_000 {
            return Err(EngineError::invalid_input(
                "vector_batch_too_large",
                "vector upsert accepts at most 5000 points per transaction",
            ));
        }
        for point in &points {
            self.validate_vector(&point.vector)?;
            if point.namespace.trim().is_empty() {
                return Err(EngineError::invalid_input(
                    "vector_namespace_empty",
                    "vector namespace must not be empty",
                ));
            }
        }

        let count = points.len();
        let mut transaction = self
            .pool
            .begin()
            .await
            .map_err(|error| Self::database_error("vector_transaction_failed", error))?;
        for point in points {
            context.ensure_active()?;
            let metadata = serde_json::to_value(&point.metadata).map_err(|error| {
                EngineError::invalid_input("vector_metadata_invalid", error.to_string())
            })?;
            sqlx::query(
                "INSERT INTO engine_vectors
                    (workspace_id, namespace, id, embedding, metadata)
                 VALUES ($1, $2, $3, $4::vector, $5)
                 ON CONFLICT (workspace_id, namespace, id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    updated_at = now()",
            )
            .bind(context.workspace_id)
            .bind(point.namespace.clone())
            .bind(point.id)
            .bind(encode_vector(&point.vector))
            .bind(metadata)
            .execute(&mut *transaction)
            .await
            .map_err(|error| Self::database_error("vector_upsert_failed", error))?;

            // Both places during the transition: the column is the rollback
            // path while it is still being written.
            sqlx::query(
                "INSERT INTO engine_vector_embeddings
                    (workspace_id, namespace, vector_id, model, embedding)
                 VALUES ($1, $2, $3, $4, $5::vector)
                 ON CONFLICT (workspace_id, namespace, vector_id, model)
                 DO UPDATE SET embedding = EXCLUDED.embedding",
            )
            .bind(context.workspace_id)
            .bind(point.namespace)
            .bind(point.id)
            .bind(&self.embedding_model)
            .bind(encode_vector(&point.vector))
            .execute(&mut *transaction)
            .await
            .map_err(|error| Self::database_error("vector_embedding_upsert_failed", error))?;
        }
        transaction
            .commit()
            .await
            .map_err(|error| Self::database_error("vector_commit_failed", error))?;
        Ok(count)
    }

    #[tracing::instrument(name = "engine.vector.search", skip_all, fields(request_id = %context.request_id, workspace_id = %context.workspace_id, limit = query.limit))]
    async fn search(
        &self,
        context: &EngineRequestContext,
        query: VectorQuery,
    ) -> EngineResult<VectorSearchResult> {
        context.ensure_active()?;
        self.validate_vector(&query.vector)?;
        if query.namespace.trim().is_empty() {
            return Err(EngineError::invalid_input(
                "vector_namespace_empty",
                "vector namespace must not be empty",
            ));
        }
        if query.limit == 0 || query.limit > 1_000 {
            return Err(EngineError::invalid_input(
                "vector_limit_invalid",
                "vector search limit must be between 1 and 1000",
            ));
        }
        if query
            .minimum_score
            .is_some_and(|score| !score.is_finite() || !(-1.0..=1.0).contains(&score))
        {
            return Err(EngineError::invalid_input(
                "vector_minimum_score_invalid",
                "minimum score must be finite and between -1 and 1",
            ));
        }

        let started = Instant::now();
        let vector = encode_vector(&query.vector);
        let filter = serde_json::to_value(&query.filter.equals).map_err(|error| {
            EngineError::invalid_input("vector_filter_invalid", error.to_string())
        })?;
        let sql = format!(
            "SELECT v.id, v.namespace, v.metadata,
                    (1 - (e.embedding <=> $3::vector))::real AS score
             FROM engine_vectors v
             -- Only vectors from the model asking. Mixing spaces returns
             -- confident noise and reports nothing wrong.
             JOIN engine_vector_embeddings e
               ON e.workspace_id = v.workspace_id
              AND e.namespace = v.namespace
              AND e.vector_id = v.id
              AND e.model = $6
             WHERE v.workspace_id = $1
               AND v.namespace = $2
               AND v.metadata @> $4::jsonb
             ORDER BY e.embedding::halfvec({dimensions}) <=> $3::vector::halfvec({dimensions})
             LIMIT $5",
            dimensions = self.dimensions
        );
        // Fetch extra rows when a score floor is applied so filtering does not
        // needlessly return a short page when lower-ranked candidates qualify.
        let fetch_limit = if query.minimum_score.is_some() {
            query.limit.saturating_mul(4).min(4_000)
        } else {
            query.limit
        };
        // `dimensions` is a validated `usize`, so the only interpolated SQL
        // fragment is an integer type modifier and cannot contain SQL syntax.
        let rows = sqlx::query(AssertSqlSafe(sql))
            .bind(context.workspace_id)
            .bind(&query.namespace)
            .bind(vector)
            .bind(filter)
            .bind(i64::try_from(fetch_limit).unwrap_or(i64::MAX))
            .bind(&self.embedding_model)
            .fetch_all(&self.pool)
            .await
            .map_err(|error| Self::database_error("vector_search_failed", error))?;
        let examined = rows.len();
        let mut matches = rows
            .into_iter()
            .map(|row| {
                let metadata: Value = row
                    .try_get("metadata")
                    .map_err(|error| Self::database_error("vector_row_invalid", error))?;
                let metadata = serde_json::from_value(metadata).map_err(|error| {
                    EngineError::dependency("vector_metadata_decode_failed", error.to_string())
                })?;
                Ok(VectorMatch {
                    id: row
                        .try_get("id")
                        .map_err(|error| Self::database_error("vector_row_invalid", error))?,
                    score: row
                        .try_get("score")
                        .map_err(|error| Self::database_error("vector_row_invalid", error))?,
                    namespace: row
                        .try_get("namespace")
                        .map_err(|error| Self::database_error("vector_row_invalid", error))?,
                    metadata,
                })
            })
            .collect::<EngineResult<Vec<_>>>()?;
        if let Some(minimum) = query.minimum_score {
            matches.retain(|candidate| candidate.score >= minimum);
        }
        matches.truncate(query.limit);
        Ok(VectorSearchResult {
            matches,
            examined,
            took_ms: started.elapsed().as_millis() as u64,
        })
    }

    #[tracing::instrument(name = "engine.vector.delete", skip_all, fields(request_id = %context.request_id, workspace_id = %context.workspace_id, id_count = request.ids.len(), delete_namespace = request.delete_namespace))]
    async fn delete(
        &self,
        context: &EngineRequestContext,
        request: DeleteVectors,
    ) -> EngineResult<usize> {
        context.ensure_active()?;
        if request.namespace.trim().is_empty() {
            return Err(EngineError::invalid_input(
                "vector_namespace_empty",
                "vector namespace must not be empty",
            ));
        }
        if !request.delete_namespace && request.ids.is_empty() {
            return Ok(0);
        }
        let result = sqlx::query(
            "DELETE FROM engine_vectors
             WHERE workspace_id = $1
               AND namespace = $2
               AND ($3 OR id = ANY($4))",
        )
        .bind(context.workspace_id)
        .bind(request.namespace)
        .bind(request.delete_namespace)
        .bind(request.ids)
        .execute(&self.pool)
        .await
        .map_err(|error| Self::database_error("vector_delete_failed", error))?;
        usize::try_from(result.rows_affected()).map_err(|_| {
            EngineError::new(
                EngineErrorKind::Internal,
                "vector_delete_count_overflow",
                "deleted vector count does not fit usize",
                false,
            )
        })
    }
}
