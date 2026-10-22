//! Context assembly and budgeting.
//!
//! This engine does not retrieve knowledge itself. It accepts candidates from
//! conversation history, retrieval, memory, tools, and attachments, then
//! produces a deterministic, auditable context package within a model budget.

#![forbid(unsafe_code)]

use async_trait::async_trait;
use engine_core::{
    Engine, EngineHealth, EngineId, EngineKind, EngineMetadata, EngineRequestContext, EngineResult,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::time::Instant;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextSource {
    System,
    Conversation,
    Retrieval,
    Memory,
    Tool,
    Attachment,
    UserPinned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextRole {
    System,
    User,
    Assistant,
    Tool,
    Reference,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextCandidate {
    pub id: Uuid,
    pub source: ContextSource,
    pub role: ContextRole,
    pub content: String,
    pub estimated_tokens: usize,
    pub priority: u8,
    pub relevance: f32,
    pub required: bool,
    pub truncatable: bool,
    pub order: i64,
    pub dedupe_key: Option<String>,
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
}

impl ContextCandidate {
    pub fn new(source: ContextSource, role: ContextRole, content: impl Into<String>) -> Self {
        let content = content.into();
        Self {
            id: Uuid::new_v4(),
            source,
            role,
            estimated_tokens: estimate_tokens(&content),
            content,
            priority: 50,
            relevance: 0.5,
            required: false,
            truncatable: true,
            order: 0,
            dedupe_key: None,
            metadata: BTreeMap::new(),
        }
    }

    fn validate(&self) -> EngineResult<()> {
        if self.content.trim().is_empty() {
            return Err(engine_core::EngineError::invalid_input(
                "context_content_empty",
                "context candidate content must not be empty",
            ));
        }
        if self.priority > 100 {
            return Err(engine_core::EngineError::invalid_input(
                "context_priority_invalid",
                "context priority must be between 0 and 100",
            ));
        }
        if !self.relevance.is_finite() || !(0.0..=1.0).contains(&self.relevance) {
            return Err(engine_core::EngineError::invalid_input(
                "context_relevance_invalid",
                "context relevance must be finite and between 0 and 1",
            ));
        }
        if self.estimated_tokens == 0 {
            return Err(engine_core::EngineError::invalid_input(
                "context_tokens_zero",
                "context candidate token estimate must be greater than zero",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextBudget {
    pub model_window: usize,
    pub reserve_for_output: usize,
    pub fixed_overhead: usize,
    pub max_items: usize,
    #[serde(default)]
    pub source_limits: BTreeMap<ContextSource, usize>,
}

impl ContextBudget {
    pub fn available_tokens(&self) -> EngineResult<usize> {
        self.model_window
            .checked_sub(self.reserve_for_output.saturating_add(self.fixed_overhead))
            .filter(|available| *available > 0)
            .ok_or_else(|| {
                engine_core::EngineError::invalid_input(
                    "context_budget_exhausted",
                    "output reserve and fixed overhead consume the entire model window",
                )
            })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionStrategy {
    PriorityThenRelevance,
    RelevancePerToken,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextAssemblyRequest {
    pub budget: ContextBudget,
    pub candidates: Vec<ContextCandidate>,
    pub strategy: SelectionStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectedContext {
    pub candidate: ContextCandidate,
    pub selected_tokens: usize,
    pub truncated: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExclusionReason {
    Duplicate,
    TokenBudget,
    SourceBudget,
    ItemLimit,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExcludedContext {
    pub candidate_id: Uuid,
    pub source: ContextSource,
    pub reason: ExclusionReason,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextAssembly {
    pub items: Vec<SelectedContext>,
    pub excluded: Vec<ExcludedContext>,
    pub model_window: usize,
    pub available_tokens: usize,
    pub used_tokens: usize,
    pub reserved_output_tokens: usize,
    pub took_ms: u64,
    pub engine_id: EngineId,
}

#[async_trait]
pub trait ContextEngine: Engine {
    async fn assemble(
        &self,
        context: &EngineRequestContext,
        request: ContextAssemblyRequest,
    ) -> EngineResult<ContextAssembly>;
}

pub struct BudgetingContextEngine {
    metadata: EngineMetadata,
}

impl BudgetingContextEngine {
    pub fn new() -> EngineResult<Self> {
        let id = EngineId::new("context.budgeting-v1")?;
        let metadata = EngineMetadata::new(
            id,
            EngineKind::Context,
            "Deterministic context budgeting engine",
        )
        .with_capabilities([
            "token_budgeting",
            "source_quotas",
            "deduplication",
            "required_context",
            "deterministic_ordering",
            "bounded_truncation",
            "exclusion_audit",
        ]);
        Ok(Self { metadata })
    }
}

#[async_trait]
impl Engine for BudgetingContextEngine {
    fn metadata(&self) -> EngineMetadata {
        self.metadata.clone()
    }

    async fn health(&self) -> EngineHealth {
        EngineHealth::healthy(0)
    }
}

#[async_trait]
impl ContextEngine for BudgetingContextEngine {
    async fn assemble(
        &self,
        context: &EngineRequestContext,
        request: ContextAssemblyRequest,
    ) -> EngineResult<ContextAssembly> {
        context.ensure_active()?;
        let started = Instant::now();
        let available_tokens = request.budget.available_tokens()?;
        if request.budget.max_items == 0 || request.budget.max_items > 10_000 {
            return Err(engine_core::EngineError::invalid_input(
                "context_item_limit_invalid",
                "context max_items must be between 1 and 10000",
            ));
        }
        for candidate in &request.candidates {
            candidate.validate()?;
        }

        let (mut candidates, mut excluded) = deduplicate(request.candidates);
        candidates.sort_by(|left, right| rank_candidates(left, right, request.strategy));
        // Required items are never silently displaced by a high-scoring optional item.
        candidates.sort_by_key(|candidate| !candidate.required);

        let required_tokens = candidates
            .iter()
            .filter(|candidate| candidate.required)
            .map(|candidate| candidate.estimated_tokens)
            .sum::<usize>();
        let required_items = candidates
            .iter()
            .filter(|candidate| candidate.required)
            .count();
        if required_items > request.budget.max_items {
            return Err(engine_core::EngineError::invalid_input(
                "context_required_item_overflow",
                format!(
                    "required context needs {required_items} items; max_items is {}",
                    request.budget.max_items
                ),
            ));
        }
        if required_tokens > available_tokens {
            return Err(engine_core::EngineError::invalid_input(
                "context_required_overflow",
                format!(
                    "required context needs {required_tokens} tokens; only {available_tokens} are available"
                ),
            ));
        }
        let mut required_by_source = BTreeMap::<ContextSource, usize>::new();
        for candidate in candidates.iter().filter(|candidate| candidate.required) {
            *required_by_source.entry(candidate.source).or_default() += candidate.estimated_tokens;
        }
        for (source, tokens) in required_by_source {
            if request
                .budget
                .source_limits
                .get(&source)
                .is_some_and(|limit| tokens > *limit)
            {
                return Err(engine_core::EngineError::invalid_input(
                    "context_required_source_overflow",
                    format!("required {source:?} context exceeds its source token limit"),
                ));
            }
        }

        let mut selected = Vec::new();
        let mut used_tokens = 0_usize;
        let mut source_usage = BTreeMap::<ContextSource, usize>::new();
        for mut candidate in candidates {
            if selected.len() >= request.budget.max_items {
                excluded.push(exclusion(&candidate, ExclusionReason::ItemLimit));
                continue;
            }
            let source_used = source_usage
                .get(&candidate.source)
                .copied()
                .unwrap_or_default();
            let source_limit = request
                .budget
                .source_limits
                .get(&candidate.source)
                .copied()
                .unwrap_or(available_tokens);
            let total_remaining = available_tokens.saturating_sub(used_tokens);
            let source_remaining = source_limit.saturating_sub(source_used);
            let remaining = total_remaining.min(source_remaining);

            if candidate.estimated_tokens <= remaining {
                used_tokens += candidate.estimated_tokens;
                *source_usage.entry(candidate.source).or_default() += candidate.estimated_tokens;
                selected.push(SelectedContext {
                    selected_tokens: candidate.estimated_tokens,
                    candidate,
                    truncated: false,
                });
            } else if candidate.truncatable && remaining >= 16 && !candidate.required {
                candidate.content = truncate_to_tokens(&candidate.content, remaining);
                candidate.estimated_tokens = estimate_tokens(&candidate.content).min(remaining);
                used_tokens += candidate.estimated_tokens;
                *source_usage.entry(candidate.source).or_default() += candidate.estimated_tokens;
                selected.push(SelectedContext {
                    selected_tokens: candidate.estimated_tokens,
                    candidate,
                    truncated: true,
                });
            } else {
                let reason = if source_remaining < total_remaining {
                    ExclusionReason::SourceBudget
                } else {
                    ExclusionReason::TokenBudget
                };
                excluded.push(exclusion(&candidate, reason));
            }
        }

        // The model sees stable semantic order, not greedy selection order.
        selected.sort_by(|left, right| {
            left.candidate
                .order
                .cmp(&right.candidate.order)
                .then_with(|| left.candidate.id.cmp(&right.candidate.id))
        });
        Ok(ContextAssembly {
            items: selected,
            excluded,
            model_window: request.budget.model_window,
            available_tokens,
            used_tokens,
            reserved_output_tokens: request.budget.reserve_for_output,
            took_ms: started.elapsed().as_millis() as u64,
            engine_id: self.metadata.id.clone(),
        })
    }
}

fn deduplicate(candidates: Vec<ContextCandidate>) -> (Vec<ContextCandidate>, Vec<ExcludedContext>) {
    let mut by_key = HashMap::<String, ContextCandidate>::new();
    let mut without_key = Vec::new();
    let mut excluded = Vec::new();
    for candidate in candidates {
        let Some(key) = candidate.dedupe_key.clone() else {
            without_key.push(candidate);
            continue;
        };
        match by_key.entry(key) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(candidate);
            }
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                let existing = entry.get();
                let replace = candidate.required && !existing.required
                    || candidate.required == existing.required
                        && (candidate.priority, candidate.relevance)
                            > (existing.priority, existing.relevance);
                if replace {
                    excluded.push(exclusion(existing, ExclusionReason::Duplicate));
                    entry.insert(candidate);
                } else {
                    excluded.push(exclusion(&candidate, ExclusionReason::Duplicate));
                }
            }
        }
    }
    without_key.extend(by_key.into_values());
    (without_key, excluded)
}

fn rank_candidates(
    left: &ContextCandidate,
    right: &ContextCandidate,
    strategy: SelectionStrategy,
) -> std::cmp::Ordering {
    match strategy {
        SelectionStrategy::PriorityThenRelevance => right
            .priority
            .cmp(&left.priority)
            .then_with(|| right.relevance.total_cmp(&left.relevance))
            .then_with(|| left.estimated_tokens.cmp(&right.estimated_tokens))
            .then_with(|| left.id.cmp(&right.id)),
        SelectionStrategy::RelevancePerToken => {
            let left_utility = (f32::from(left.priority) / 100.0 + left.relevance)
                / left.estimated_tokens.max(1) as f32;
            let right_utility = (f32::from(right.priority) / 100.0 + right.relevance)
                / right.estimated_tokens.max(1) as f32;
            right_utility
                .total_cmp(&left_utility)
                .then_with(|| left.id.cmp(&right.id))
        }
    }
}

fn exclusion(candidate: &ContextCandidate, reason: ExclusionReason) -> ExcludedContext {
    ExcludedContext {
        candidate_id: candidate.id,
        source: candidate.source,
        reason,
    }
}

/// Conservative provider-independent estimate. Callers may replace it with a
/// tokenizer-specific estimate before assembly.
pub fn estimate_tokens(content: &str) -> usize {
    content.chars().count().div_ceil(4).max(1)
}

fn truncate_to_tokens(content: &str, tokens: usize) -> String {
    let character_limit = tokens.saturating_mul(4);
    if content.chars().count() <= character_limit {
        return content.to_owned();
    }
    let mut output = content
        .chars()
        .take(character_limit.saturating_sub(1))
        .collect::<String>();
    if let Some(boundary) = output.rfind(char::is_whitespace) {
        output.truncate(boundary);
    }
    output.push('…');
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn budget(tokens: usize) -> ContextBudget {
        ContextBudget {
            model_window: tokens + 100,
            reserve_for_output: 80,
            fixed_overhead: 20,
            max_items: 20,
            source_limits: BTreeMap::new(),
        }
    }

    #[tokio::test]
    async fn never_exceeds_the_available_budget() {
        let engine = BudgetingContextEngine::new().unwrap();
        let context = EngineRequestContext::new(Uuid::new_v4());
        let mut candidates = Vec::new();
        for index in 0..10 {
            let mut candidate = ContextCandidate::new(
                ContextSource::Retrieval,
                ContextRole::Reference,
                "x".repeat(160),
            );
            candidate.order = index;
            candidates.push(candidate);
        }
        let result = engine
            .assemble(
                &context,
                ContextAssemblyRequest {
                    budget: budget(100),
                    candidates,
                    strategy: SelectionStrategy::PriorityThenRelevance,
                },
            )
            .await
            .unwrap();
        assert!(result.used_tokens <= result.available_tokens);
        assert!(!result.excluded.is_empty());
    }

    #[tokio::test]
    async fn required_context_overflow_is_explicit() {
        let engine = BudgetingContextEngine::new().unwrap();
        let context = EngineRequestContext::new(Uuid::new_v4());
        let mut candidate =
            ContextCandidate::new(ContextSource::System, ContextRole::System, "x".repeat(800));
        candidate.required = true;
        let result = engine
            .assemble(
                &context,
                ContextAssemblyRequest {
                    budget: budget(100),
                    candidates: vec![candidate],
                    strategy: SelectionStrategy::PriorityThenRelevance,
                },
            )
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn deduplication_keeps_the_stronger_candidate() {
        let engine = BudgetingContextEngine::new().unwrap();
        let context = EngineRequestContext::new(Uuid::new_v4());
        let mut weaker =
            ContextCandidate::new(ContextSource::Retrieval, ContextRole::Reference, "old");
        weaker.dedupe_key = Some("source:1".into());
        weaker.priority = 20;
        let weaker_id = weaker.id;
        let mut stronger =
            ContextCandidate::new(ContextSource::Retrieval, ContextRole::Reference, "new");
        stronger.dedupe_key = Some("source:1".into());
        stronger.priority = 90;
        let stronger_id = stronger.id;
        let result = engine
            .assemble(
                &context,
                ContextAssemblyRequest {
                    budget: budget(100),
                    candidates: vec![weaker, stronger],
                    strategy: SelectionStrategy::PriorityThenRelevance,
                },
            )
            .await
            .unwrap();
        assert_eq!(result.items[0].candidate.id, stronger_id);
        assert!(result.excluded.iter().any(
            |item| item.candidate_id == weaker_id && item.reason == ExclusionReason::Duplicate
        ));
    }
}
