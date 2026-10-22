//! Memory engine.
//!
//! Memory is user/workspace state, not source knowledge. Every record has an
//! explicit scope, provenance, confidence, lifecycle, and optimistic version.

#![forbid(unsafe_code)]

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use engine_core::{
    Engine, EngineHealth, EngineId, EngineKind, EngineMetadata, EngineRequestContext, EngineResult,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MemoryId(pub Uuid);

impl MemoryId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for MemoryId {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "id", rename_all = "snake_case")]
pub enum MemoryScope {
    Workspace,
    User(String),
    Conversation(Uuid),
    Project(Uuid),
    /// Personal memory visible across this user's workspaces.
    Global(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryKind {
    Fact,
    Preference,
    Instruction,
    Goal,
    Relationship,
    Summary,
    Misconception,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryStatus {
    Active,
    Stale,
    Expired,
    Forgotten,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryProvenance {
    pub origin: String,
    pub source_id: Option<Uuid>,
    pub conversation_id: Option<Uuid>,
    pub message_id: Option<Uuid>,
    pub observed_at: DateTime<Utc>,
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
}

impl MemoryProvenance {
    pub fn user_statement() -> Self {
        Self {
            origin: "user_statement".into(),
            source_id: None,
            conversation_id: None,
            message_id: None,
            observed_at: Utc::now(),
            metadata: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: MemoryId,
    pub workspace_id: Uuid,
    pub scope: MemoryScope,
    pub kind: MemoryKind,
    pub content: String,
    pub summary: Option<String>,
    pub confidence: f32,
    pub priority: u8,
    pub status: MemoryStatus,
    pub provenance: MemoryProvenance,
    #[serde(default)]
    pub tags: BTreeSet<String>,
    #[serde(default)]
    pub contradicts: BTreeSet<MemoryId>,
    pub version: u64,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub last_accessed_at: Option<DateTime<Utc>>,
    pub expires_at: Option<DateTime<Utc>>,
}

impl MemoryRecord {
    pub fn effective_status(&self, now: DateTime<Utc>) -> MemoryStatus {
        if self.status == MemoryStatus::Active
            && self.expires_at.is_some_and(|expiry| expiry <= now)
        {
            MemoryStatus::Expired
        } else {
            self.status
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RememberMemory {
    pub scope: MemoryScope,
    pub kind: MemoryKind,
    pub content: String,
    pub summary: Option<String>,
    pub confidence: f32,
    pub priority: u8,
    pub provenance: MemoryProvenance,
    #[serde(default)]
    pub tags: BTreeSet<String>,
    #[serde(default)]
    pub contradicts: BTreeSet<MemoryId>,
    pub expires_at: Option<DateTime<Utc>>,
}

impl RememberMemory {
    fn validate(&self) -> EngineResult<()> {
        if self.content.trim().is_empty() {
            return Err(engine_core::EngineError::invalid_input(
                "memory_content_empty",
                "memory content must not be empty",
            ));
        }
        if self.content.len() > 32_000 {
            return Err(engine_core::EngineError::invalid_input(
                "memory_content_too_large",
                "memory content exceeds 32000 bytes",
            ));
        }
        if !(0.0..=1.0).contains(&self.confidence) || !self.confidence.is_finite() {
            return Err(engine_core::EngineError::invalid_input(
                "memory_confidence_invalid",
                "memory confidence must be finite and between 0 and 1",
            ));
        }
        if self.priority > 100 {
            return Err(engine_core::EngineError::invalid_input(
                "memory_priority_invalid",
                "memory priority must be between 0 and 100",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UpdateMemory {
    pub expected_version: Option<u64>,
    pub content: Option<String>,
    pub summary: Option<Option<String>>,
    pub confidence: Option<f32>,
    pub priority: Option<u8>,
    pub status: Option<MemoryStatus>,
    pub expires_at: Option<Option<DateTime<Utc>>>,
    pub tags: Option<BTreeSet<String>>,
    pub contradicts: Option<BTreeSet<MemoryId>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySearch {
    pub query: String,
    pub limit: usize,
    pub scopes: Vec<MemoryScope>,
    pub kinds: Vec<MemoryKind>,
    #[serde(default)]
    pub tags: BTreeSet<String>,
    pub minimum_confidence: Option<f32>,
    pub include_stale: bool,
}

impl MemorySearch {
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            limit: 10,
            scopes: Vec::new(),
            kinds: Vec::new(),
            tags: BTreeSet::new(),
            minimum_confidence: None,
            include_stale: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMatch {
    pub memory: MemoryRecord,
    pub score: f32,
    pub matched_terms: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConflict {
    pub left: MemoryRecord,
    pub right: MemoryRecord,
    pub explicit: bool,
}

#[async_trait]
pub trait MemoryEngine: Engine {
    async fn remember(
        &self,
        context: &EngineRequestContext,
        request: RememberMemory,
    ) -> EngineResult<MemoryRecord>;

    async fn get(&self, context: &EngineRequestContext, id: MemoryId)
    -> EngineResult<MemoryRecord>;

    async fn search(
        &self,
        context: &EngineRequestContext,
        query: MemorySearch,
    ) -> EngineResult<Vec<MemoryMatch>>;

    async fn update(
        &self,
        context: &EngineRequestContext,
        id: MemoryId,
        update: UpdateMemory,
    ) -> EngineResult<MemoryRecord>;

    async fn forget(&self, context: &EngineRequestContext, id: MemoryId) -> EngineResult<()>;

    async fn conflicts(
        &self,
        context: &EngineRequestContext,
        id: MemoryId,
    ) -> EngineResult<Vec<MemoryConflict>>;
}

#[derive(Clone)]
pub struct InMemoryMemoryEngine {
    metadata: EngineMetadata,
    records: Arc<RwLock<HashMap<(Uuid, MemoryId), MemoryRecord>>>,
}

impl InMemoryMemoryEngine {
    pub fn new() -> EngineResult<Self> {
        let id = EngineId::new("memory.in-memory-v1")?;
        let metadata = EngineMetadata::new(id, EngineKind::Memory, "In-memory memory engine")
            .with_capabilities([
                "remember",
                "search",
                "optimistic_updates",
                "expiration",
                "forget",
                "conflicts",
                "provenance",
            ]);
        Ok(Self {
            metadata,
            records: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    fn visible(record: &MemoryRecord, context: &EngineRequestContext) -> bool {
        match &record.scope {
            MemoryScope::User(user_id) => context.actor_id.as_ref() == Some(user_id),
            MemoryScope::Global(user_id) => context.actor_id.as_ref() == Some(user_id),
            _ => record.workspace_id == context.workspace_id,
        }
    }

    fn storage_workspace(scope: &MemoryScope, workspace_id: Uuid) -> Uuid {
        if matches!(scope, MemoryScope::Global(_)) {
            Uuid::nil()
        } else {
            workspace_id
        }
    }

    fn storage_key(context: &EngineRequestContext, id: MemoryId, global: bool) -> (Uuid, MemoryId) {
        (
            if global {
                Uuid::nil()
            } else {
                context.workspace_id
            },
            id,
        )
    }
}

#[async_trait]
impl Engine for InMemoryMemoryEngine {
    fn metadata(&self) -> EngineMetadata {
        self.metadata.clone()
    }

    async fn health(&self) -> EngineHealth {
        let started = Instant::now();
        let records = self.records.read().await;
        let mut health = EngineHealth::healthy(started.elapsed().as_millis() as u64);
        health
            .details
            .insert("records".into(), records.len().into());
        health
    }
}

#[async_trait]
impl MemoryEngine for InMemoryMemoryEngine {
    async fn remember(
        &self,
        context: &EngineRequestContext,
        request: RememberMemory,
    ) -> EngineResult<MemoryRecord> {
        context.ensure_active()?;
        request.validate()?;
        let owned_scope = match &request.scope {
            MemoryScope::User(user_id) | MemoryScope::Global(user_id) => Some(user_id),
            _ => None,
        };
        if owned_scope.is_some_and(|user_id| context.actor_id.as_ref() != Some(user_id)) {
            return Err(engine_core::EngineError::invalid_input(
                "memory_scope_actor_mismatch",
                "personal memory scope does not match the request actor",
            ));
        }
        let now = Utc::now();
        let record = MemoryRecord {
            id: MemoryId::new(),
            workspace_id: context.workspace_id,
            scope: request.scope,
            kind: request.kind,
            content: request.content.trim().to_owned(),
            summary: request.summary,
            confidence: request.confidence,
            priority: request.priority,
            status: MemoryStatus::Active,
            provenance: request.provenance,
            tags: request.tags,
            contradicts: request.contradicts,
            version: 1,
            created_at: now,
            updated_at: now,
            last_accessed_at: None,
            expires_at: request.expires_at,
        };
        let storage_workspace = Self::storage_workspace(&record.scope, context.workspace_id);
        self.records
            .write()
            .await
            .insert((storage_workspace, record.id), record.clone());
        Ok(record)
    }

    async fn get(
        &self,
        context: &EngineRequestContext,
        id: MemoryId,
    ) -> EngineResult<MemoryRecord> {
        context.ensure_active()?;
        let mut records = self.records.write().await;
        let local_key = Self::storage_key(context, id, false);
        let global_key = Self::storage_key(context, id, true);
        let key = if records
            .get(&local_key)
            .is_some_and(|record| Self::visible(record, context))
        {
            local_key
        } else {
            global_key
        };
        let record = records
            .get_mut(&key)
            .filter(|record| Self::visible(record, context))
            .ok_or_else(|| {
                engine_core::EngineError::not_found("memory_not_found", "memory not found")
            })?;
        record.last_accessed_at = Some(Utc::now());
        Ok(record.clone())
    }

    async fn search(
        &self,
        context: &EngineRequestContext,
        query: MemorySearch,
    ) -> EngineResult<Vec<MemoryMatch>> {
        context.ensure_active()?;
        if query.limit == 0 || query.limit > 250 {
            return Err(engine_core::EngineError::invalid_input(
                "memory_search_limit_invalid",
                "memory search limit must be between 1 and 250",
            ));
        }
        let query_terms = tokenize(&query.query);
        let now = Utc::now();
        let records = self.records.read().await;
        let mut matches = records
            .values()
            .filter(|record| Self::visible(record, context))
            .filter(|record| {
                let status = record.effective_status(now);
                status == MemoryStatus::Active
                    || (query.include_stale && status == MemoryStatus::Stale)
            })
            .filter(|record| query.scopes.is_empty() || query.scopes.contains(&record.scope))
            .filter(|record| query.kinds.is_empty() || query.kinds.contains(&record.kind))
            .filter(|record| query.tags.is_subset(&record.tags))
            .filter(|record| {
                query
                    .minimum_confidence
                    .is_none_or(|minimum| record.confidence >= minimum)
            })
            .filter_map(|record| {
                let memory_terms = tokenize(&record.content);
                let matched = query_terms
                    .intersection(&memory_terms)
                    .cloned()
                    .collect::<Vec<_>>();
                if !query_terms.is_empty() && matched.is_empty() {
                    return None;
                }
                let lexical = if query_terms.is_empty() {
                    1.0
                } else {
                    matched.len() as f32 / query_terms.len() as f32
                };
                let score = lexical * 0.6
                    + record.confidence * 0.25
                    + f32::from(record.priority) / 100.0 * 0.15;
                Some(MemoryMatch {
                    memory: record.clone(),
                    score,
                    matched_terms: matched,
                })
            })
            .collect::<Vec<_>>();
        matches.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| right.memory.updated_at.cmp(&left.memory.updated_at))
        });
        matches.truncate(query.limit);
        Ok(matches)
    }

    async fn update(
        &self,
        context: &EngineRequestContext,
        id: MemoryId,
        update: UpdateMemory,
    ) -> EngineResult<MemoryRecord> {
        context.ensure_active()?;
        let mut records = self.records.write().await;
        let local_key = Self::storage_key(context, id, false);
        let global_key = Self::storage_key(context, id, true);
        let key = if records
            .get(&local_key)
            .is_some_and(|record| Self::visible(record, context))
        {
            local_key
        } else {
            global_key
        };
        let record = records
            .get_mut(&key)
            .filter(|record| Self::visible(record, context))
            .ok_or_else(|| {
                engine_core::EngineError::not_found("memory_not_found", "memory not found")
            })?;
        if update
            .expected_version
            .is_some_and(|expected| expected != record.version)
        {
            return Err(engine_core::EngineError::conflict(
                "memory_version_conflict",
                format!(
                    "memory version is {}; update expected another version",
                    record.version
                ),
            ));
        }
        if let Some(content) = update.content {
            if content.trim().is_empty() {
                return Err(engine_core::EngineError::invalid_input(
                    "memory_content_empty",
                    "memory content must not be empty",
                ));
            }
            record.content = content.trim().to_owned();
        }
        if let Some(summary) = update.summary {
            record.summary = summary;
        }
        if let Some(confidence) = update.confidence {
            if !(0.0..=1.0).contains(&confidence) || !confidence.is_finite() {
                return Err(engine_core::EngineError::invalid_input(
                    "memory_confidence_invalid",
                    "memory confidence must be finite and between 0 and 1",
                ));
            }
            record.confidence = confidence;
        }
        if let Some(priority) = update.priority {
            if priority > 100 {
                return Err(engine_core::EngineError::invalid_input(
                    "memory_priority_invalid",
                    "memory priority must be between 0 and 100",
                ));
            }
            record.priority = priority;
        }
        if let Some(status) = update.status {
            record.status = status;
        }
        if let Some(expires_at) = update.expires_at {
            record.expires_at = expires_at;
        }
        if let Some(tags) = update.tags {
            record.tags = tags;
        }
        if let Some(contradicts) = update.contradicts {
            record.contradicts = contradicts;
        }
        record.version += 1;
        record.updated_at = Utc::now();
        Ok(record.clone())
    }

    async fn forget(&self, context: &EngineRequestContext, id: MemoryId) -> EngineResult<()> {
        self.update(
            context,
            id,
            UpdateMemory {
                status: Some(MemoryStatus::Forgotten),
                ..UpdateMemory::default()
            },
        )
        .await?;
        Ok(())
    }

    async fn conflicts(
        &self,
        context: &EngineRequestContext,
        id: MemoryId,
    ) -> EngineResult<Vec<MemoryConflict>> {
        context.ensure_active()?;
        let records = self.records.read().await;
        let record = records
            .values()
            .find(|record| record.id == id && Self::visible(record, context))
            .ok_or_else(|| {
                engine_core::EngineError::not_found("memory_not_found", "memory not found")
            })?;
        Ok(record
            .contradicts
            .iter()
            .filter_map(|other_id| records.values().find(|record| record.id == *other_id))
            .filter(|other| Self::visible(other, context))
            .map(|other| MemoryConflict {
                left: record.clone(),
                right: other.clone(),
                explicit: true,
            })
            .collect())
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

    fn request(content: &str) -> RememberMemory {
        RememberMemory {
            scope: MemoryScope::Workspace,
            kind: MemoryKind::Preference,
            content: content.into(),
            summary: None,
            confidence: 0.9,
            priority: 80,
            provenance: MemoryProvenance::user_statement(),
            tags: BTreeSet::from(["learning".into()]),
            contradicts: BTreeSet::new(),
            expires_at: None,
        }
    }

    #[tokio::test]
    async fn memory_search_respects_workspace_and_returns_provenance() {
        let engine = InMemoryMemoryEngine::new().unwrap();
        let context = EngineRequestContext::new(Uuid::new_v4());
        let other = EngineRequestContext::new(Uuid::new_v4());
        let visible = engine
            .remember(&context, request("prefers visual calculus explanations"))
            .await
            .unwrap();
        engine
            .remember(&other, request("prefers calculus textbooks"))
            .await
            .unwrap();
        let found = engine
            .search(&context, MemorySearch::new("calculus visual"))
            .await
            .unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].memory.id, visible.id);
        assert_eq!(found[0].memory.provenance.origin, "user_statement");
    }

    #[tokio::test]
    async fn optimistic_updates_prevent_lost_writes() {
        let engine = InMemoryMemoryEngine::new().unwrap();
        let context = EngineRequestContext::new(Uuid::new_v4());
        let memory = engine
            .remember(&context, request("uses metric units"))
            .await
            .unwrap();
        engine
            .update(
                &context,
                memory.id,
                UpdateMemory {
                    expected_version: Some(1),
                    priority: Some(90),
                    ..UpdateMemory::default()
                },
            )
            .await
            .unwrap();
        let stale = engine
            .update(
                &context,
                memory.id,
                UpdateMemory {
                    expected_version: Some(1),
                    priority: Some(20),
                    ..UpdateMemory::default()
                },
            )
            .await;
        assert!(stale.is_err());
    }

    #[tokio::test]
    async fn forgotten_and_expired_memories_are_not_retrieved() {
        let engine = InMemoryMemoryEngine::new().unwrap();
        let context = EngineRequestContext::new(Uuid::new_v4());
        let forgotten = engine
            .remember(&context, request("old preference"))
            .await
            .unwrap();
        engine.forget(&context, forgotten.id).await.unwrap();
        let mut expired_request = request("expired preference");
        expired_request.expires_at = Some(Utc::now() - chrono::Duration::seconds(1));
        engine.remember(&context, expired_request).await.unwrap();
        assert!(
            engine
                .search(&context, MemorySearch::new("preference"))
                .await
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn personal_global_memory_crosses_workspaces_but_not_users() {
        let engine = InMemoryMemoryEngine::new().unwrap();
        let mut first = EngineRequestContext::new(Uuid::new_v4());
        first.actor_id = Some("user_a".into());
        let mut second = EngineRequestContext::new(Uuid::new_v4());
        second.actor_id = Some("user_a".into());
        let mut stranger = EngineRequestContext::new(Uuid::new_v4());
        stranger.actor_id = Some("user_b".into());
        let mut global = request("prefers concise explanations");
        global.scope = MemoryScope::Global("user_a".into());
        engine.remember(&first, global).await.unwrap();
        assert_eq!(
            engine
                .search(&second, MemorySearch::new("concise"))
                .await
                .unwrap()
                .len(),
            1
        );
        assert!(
            engine
                .search(&stranger, MemorySearch::new("concise"))
                .await
                .unwrap()
                .is_empty()
        );
    }
}
