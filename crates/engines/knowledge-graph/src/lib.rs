//! Workspace-isolated knowledge graph engine with explicit provenance,
//! confidence, bounded traversal, and path finding.

#![forbid(unsafe_code)]

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use engine_core::{
    Engine, EngineHealth, EngineId, EngineKind, EngineMetadata, EngineRequestContext, EngineResult,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct NodeId(pub Uuid);

impl NodeId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for NodeId {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EdgeId(pub Uuid);

impl EdgeId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for EdgeId {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    Concept,
    Entity,
    Topic,
    Person,
    Organization,
    Place,
    Formula,
    CodeSymbol,
    Source,
    LearningObjective,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphProvenance {
    pub source_id: Option<Uuid>,
    pub chunk_id: Option<Uuid>,
    pub extractor: String,
    pub observed_at: DateTime<Utc>,
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: NodeId,
    pub workspace_id: Uuid,
    pub kind: NodeKind,
    pub label: String,
    #[serde(default)]
    pub aliases: BTreeSet<String>,
    #[serde(default)]
    pub properties: BTreeMap<String, Value>,
    pub confidence: f32,
    #[serde(default)]
    pub provenance: Vec<GraphProvenance>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub id: EdgeId,
    pub workspace_id: Uuid,
    pub from: NodeId,
    pub to: NodeId,
    pub relation: String,
    pub directed: bool,
    pub weight: f32,
    pub confidence: f32,
    #[serde(default)]
    pub properties: BTreeMap<String, Value>,
    #[serde(default)]
    pub provenance: Vec<GraphProvenance>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphMutation {
    #[serde(default)]
    pub nodes: Vec<GraphNode>,
    #[serde(default)]
    pub edges: Vec<GraphEdge>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraversalDirection {
    Outgoing,
    Incoming,
    Both,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversalQuery {
    pub start: Vec<NodeId>,
    pub direction: TraversalDirection,
    pub max_depth: usize,
    pub max_nodes: usize,
    #[serde(default)]
    pub relations: BTreeSet<String>,
    #[serde(default)]
    pub node_kinds: Vec<NodeKind>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversedNode {
    pub node: GraphNode,
    pub depth: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversalResult {
    pub nodes: Vec<TraversedNode>,
    pub edges: Vec<GraphEdge>,
    pub truncated: bool,
    pub took_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphPath {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

#[async_trait]
pub trait KnowledgeGraphEngine: Engine {
    async fn upsert(
        &self,
        context: &EngineRequestContext,
        mutation: GraphMutation,
    ) -> EngineResult<(usize, usize)>;

    async fn get_node(&self, context: &EngineRequestContext, id: NodeId)
    -> EngineResult<GraphNode>;

    async fn traverse(
        &self,
        context: &EngineRequestContext,
        query: TraversalQuery,
    ) -> EngineResult<TraversalResult>;

    async fn shortest_path(
        &self,
        context: &EngineRequestContext,
        from: NodeId,
        to: NodeId,
        max_depth: usize,
    ) -> EngineResult<Option<GraphPath>>;

    async fn remove_node(&self, context: &EngineRequestContext, id: NodeId) -> EngineResult<bool>;
}

#[derive(Default)]
struct Store {
    nodes: HashMap<(Uuid, NodeId), GraphNode>,
    edges: HashMap<(Uuid, EdgeId), GraphEdge>,
}

#[derive(Clone)]
pub struct InMemoryKnowledgeGraphEngine {
    metadata: EngineMetadata,
    store: Arc<RwLock<Store>>,
}

impl InMemoryKnowledgeGraphEngine {
    pub fn new() -> EngineResult<Self> {
        let id = EngineId::new("knowledge-graph.in-memory-v1")?;
        let metadata = EngineMetadata::new(
            id,
            EngineKind::KnowledgeGraph,
            "In-memory knowledge graph engine",
        )
        .with_capabilities([
            "node_upsert",
            "edge_upsert",
            "bounded_traversal",
            "shortest_path",
            "provenance",
            "workspace_isolation",
        ]);
        Ok(Self {
            metadata,
            store: Arc::new(RwLock::new(Store::default())),
        })
    }

    fn validate_node(context: &EngineRequestContext, node: &GraphNode) -> EngineResult<()> {
        if node.workspace_id != context.workspace_id {
            return Err(engine_core::EngineError::invalid_input(
                "graph_workspace_mismatch",
                "graph node workspace does not match request workspace",
            ));
        }
        if node.label.trim().is_empty() {
            return Err(engine_core::EngineError::invalid_input(
                "graph_node_label_empty",
                "graph node label must not be empty",
            ));
        }
        validate_confidence(node.confidence, "graph_node_confidence_invalid")
    }

    fn validate_edge(context: &EngineRequestContext, edge: &GraphEdge) -> EngineResult<()> {
        if edge.workspace_id != context.workspace_id {
            return Err(engine_core::EngineError::invalid_input(
                "graph_workspace_mismatch",
                "graph edge workspace does not match request workspace",
            ));
        }
        if edge.from == edge.to {
            return Err(engine_core::EngineError::invalid_input(
                "graph_self_edge",
                "self-referential graph edges are not accepted",
            ));
        }
        if edge.relation.trim().is_empty() {
            return Err(engine_core::EngineError::invalid_input(
                "graph_relation_empty",
                "graph edge relation must not be empty",
            ));
        }
        if !edge.weight.is_finite() || edge.weight < 0.0 {
            return Err(engine_core::EngineError::invalid_input(
                "graph_edge_weight_invalid",
                "graph edge weight must be finite and non-negative",
            ));
        }
        validate_confidence(edge.confidence, "graph_edge_confidence_invalid")
    }
}

fn validate_confidence(confidence: f32, code: &str) -> EngineResult<()> {
    if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
        return Err(engine_core::EngineError::invalid_input(
            code,
            "confidence must be finite and between 0 and 1",
        ));
    }
    Ok(())
}

#[async_trait]
impl Engine for InMemoryKnowledgeGraphEngine {
    fn metadata(&self) -> EngineMetadata {
        self.metadata.clone()
    }

    async fn health(&self) -> EngineHealth {
        let started = Instant::now();
        let store = self.store.read().await;
        let mut health = EngineHealth::healthy(started.elapsed().as_millis() as u64);
        health
            .details
            .insert("nodes".into(), store.nodes.len().into());
        health
            .details
            .insert("edges".into(), store.edges.len().into());
        health
    }
}

#[async_trait]
impl KnowledgeGraphEngine for InMemoryKnowledgeGraphEngine {
    async fn upsert(
        &self,
        context: &EngineRequestContext,
        mutation: GraphMutation,
    ) -> EngineResult<(usize, usize)> {
        context.ensure_active()?;
        if mutation.nodes.len() > 5_000 || mutation.edges.len() > 10_000 {
            return Err(engine_core::EngineError::invalid_input(
                "graph_mutation_too_large",
                "a graph mutation accepts at most 5000 nodes and 10000 edges",
            ));
        }
        for node in &mutation.nodes {
            Self::validate_node(context, node)?;
        }
        for edge in &mutation.edges {
            Self::validate_edge(context, edge)?;
        }
        let store = self.store.read().await;
        let new_node_ids = mutation
            .nodes
            .iter()
            .map(|node| node.id)
            .collect::<HashSet<_>>();
        for edge in &mutation.edges {
            for endpoint in [edge.from, edge.to] {
                if !new_node_ids.contains(&endpoint)
                    && !store.nodes.contains_key(&(context.workspace_id, endpoint))
                {
                    return Err(engine_core::EngineError::invalid_input(
                        "graph_edge_endpoint_missing",
                        format!("graph edge endpoint {} does not exist", endpoint.0),
                    ));
                }
            }
        }
        drop(store);

        let node_count = mutation.nodes.len();
        let edge_count = mutation.edges.len();
        let mut store = self.store.write().await;
        for node in mutation.nodes {
            store.nodes.insert((context.workspace_id, node.id), node);
        }
        for edge in mutation.edges {
            store.edges.insert((context.workspace_id, edge.id), edge);
        }
        Ok((node_count, edge_count))
    }

    async fn get_node(
        &self,
        context: &EngineRequestContext,
        id: NodeId,
    ) -> EngineResult<GraphNode> {
        context.ensure_active()?;
        self.store
            .read()
            .await
            .nodes
            .get(&(context.workspace_id, id))
            .cloned()
            .ok_or_else(|| {
                engine_core::EngineError::not_found("graph_node_not_found", "graph node not found")
            })
    }

    async fn traverse(
        &self,
        context: &EngineRequestContext,
        query: TraversalQuery,
    ) -> EngineResult<TraversalResult> {
        context.ensure_active()?;
        validate_bounds(query.max_depth, query.max_nodes)?;
        if query.start.is_empty() {
            return Err(engine_core::EngineError::invalid_input(
                "graph_start_empty",
                "graph traversal requires at least one start node",
            ));
        }
        let started = Instant::now();
        let store = self.store.read().await;
        for node in &query.start {
            if !store.nodes.contains_key(&(context.workspace_id, *node)) {
                return Err(engine_core::EngineError::not_found(
                    "graph_node_not_found",
                    format!("start node {} not found", node.0),
                ));
            }
        }

        let mut queue = query
            .start
            .iter()
            .map(|node| (*node, 0))
            .collect::<VecDeque<_>>();
        let mut visited = HashMap::<NodeId, usize>::new();
        let mut edge_ids = BTreeSet::new();
        let mut truncated = false;
        while let Some((node_id, depth)) = queue.pop_front() {
            if visited.contains_key(&node_id) {
                continue;
            }
            if visited.len() >= query.max_nodes {
                truncated = true;
                break;
            }
            visited.insert(node_id, depth);
            if depth >= query.max_depth {
                continue;
            }
            for edge in store
                .edges
                .values()
                .filter(|edge| edge.workspace_id == context.workspace_id)
            {
                if !query.relations.is_empty() && !query.relations.contains(&edge.relation) {
                    continue;
                }
                let next = adjacent(edge, node_id, query.direction);
                if let Some(next) = next
                    && !visited.contains_key(&next)
                {
                    edge_ids.insert(edge.id);
                    queue.push_back((next, depth + 1));
                }
            }
        }
        let mut nodes = visited
            .into_iter()
            .filter_map(|(id, depth)| {
                store
                    .nodes
                    .get(&(context.workspace_id, id))
                    .cloned()
                    .map(|node| (node, depth))
            })
            .filter(|(node, _)| {
                query.node_kinds.is_empty() || query.node_kinds.contains(&node.kind)
            })
            .map(|(node, depth)| TraversedNode { node, depth })
            .collect::<Vec<_>>();
        nodes.sort_by(|left, right| {
            left.depth
                .cmp(&right.depth)
                .then_with(|| left.node.id.cmp(&right.node.id))
        });
        let edges = edge_ids
            .into_iter()
            .filter_map(|id| store.edges.get(&(context.workspace_id, id)).cloned())
            .collect();
        Ok(TraversalResult {
            nodes,
            edges,
            truncated,
            took_ms: started.elapsed().as_millis() as u64,
        })
    }

    async fn shortest_path(
        &self,
        context: &EngineRequestContext,
        from: NodeId,
        to: NodeId,
        max_depth: usize,
    ) -> EngineResult<Option<GraphPath>> {
        context.ensure_active()?;
        validate_bounds(max_depth, 10_000)?;
        let store = self.store.read().await;
        for node in [from, to] {
            if !store.nodes.contains_key(&(context.workspace_id, node)) {
                return Err(engine_core::EngineError::not_found(
                    "graph_node_not_found",
                    "graph path endpoint not found",
                ));
            }
        }
        let mut queue = VecDeque::from([(from, 0)]);
        let mut visited = HashSet::from([from]);
        let mut previous = HashMap::<NodeId, (NodeId, EdgeId)>::new();
        while let Some((node, depth)) = queue.pop_front() {
            if node == to {
                break;
            }
            if depth >= max_depth {
                continue;
            }
            for edge in store
                .edges
                .values()
                .filter(|edge| edge.workspace_id == context.workspace_id)
            {
                if let Some(next) = adjacent(edge, node, TraversalDirection::Both)
                    && visited.insert(next)
                {
                    previous.insert(next, (node, edge.id));
                    queue.push_back((next, depth + 1));
                }
            }
        }
        if !visited.contains(&to) {
            return Ok(None);
        }
        let mut node_ids = vec![to];
        let mut edge_ids = Vec::new();
        let mut cursor = to;
        while cursor != from {
            let (prior, edge) = previous[&cursor];
            edge_ids.push(edge);
            node_ids.push(prior);
            cursor = prior;
        }
        node_ids.reverse();
        edge_ids.reverse();
        Ok(Some(GraphPath {
            nodes: node_ids
                .into_iter()
                .filter_map(|id| store.nodes.get(&(context.workspace_id, id)).cloned())
                .collect(),
            edges: edge_ids
                .into_iter()
                .filter_map(|id| store.edges.get(&(context.workspace_id, id)).cloned())
                .collect(),
        }))
    }

    async fn remove_node(&self, context: &EngineRequestContext, id: NodeId) -> EngineResult<bool> {
        context.ensure_active()?;
        let mut store = self.store.write().await;
        let removed = store.nodes.remove(&(context.workspace_id, id)).is_some();
        if removed {
            store.edges.retain(|(workspace_id, _), edge| {
                *workspace_id != context.workspace_id || (edge.from != id && edge.to != id)
            });
        }
        Ok(removed)
    }
}

fn validate_bounds(max_depth: usize, max_nodes: usize) -> EngineResult<()> {
    if max_depth == 0 || max_depth > 12 {
        return Err(engine_core::EngineError::invalid_input(
            "graph_depth_invalid",
            "graph traversal depth must be between 1 and 12",
        ));
    }
    if max_nodes == 0 || max_nodes > 10_000 {
        return Err(engine_core::EngineError::invalid_input(
            "graph_node_limit_invalid",
            "graph traversal node limit must be between 1 and 10000",
        ));
    }
    Ok(())
}

fn adjacent(edge: &GraphEdge, node: NodeId, direction: TraversalDirection) -> Option<NodeId> {
    match direction {
        TraversalDirection::Outgoing if edge.from == node => Some(edge.to),
        TraversalDirection::Outgoing if !edge.directed && edge.to == node => Some(edge.from),
        TraversalDirection::Incoming if edge.to == node => Some(edge.from),
        TraversalDirection::Incoming if !edge.directed && edge.from == node => Some(edge.to),
        TraversalDirection::Both if edge.from == node => Some(edge.to),
        TraversalDirection::Both if edge.to == node => Some(edge.from),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(workspace_id: Uuid, label: &str) -> GraphNode {
        let now = Utc::now();
        GraphNode {
            id: NodeId::new(),
            workspace_id,
            kind: NodeKind::Concept,
            label: label.into(),
            aliases: BTreeSet::new(),
            properties: BTreeMap::new(),
            confidence: 0.9,
            provenance: Vec::new(),
            created_at: now,
            updated_at: now,
        }
    }

    fn edge(workspace_id: Uuid, from: NodeId, to: NodeId, relation: &str) -> GraphEdge {
        let now = Utc::now();
        GraphEdge {
            id: EdgeId::new(),
            workspace_id,
            from,
            to,
            relation: relation.into(),
            directed: true,
            weight: 1.0,
            confidence: 0.9,
            properties: BTreeMap::new(),
            provenance: Vec::new(),
            created_at: now,
            updated_at: now,
        }
    }

    #[tokio::test]
    async fn traversal_and_path_are_bounded_and_provenance_safe() {
        let engine = InMemoryKnowledgeGraphEngine::new().unwrap();
        let context = EngineRequestContext::new(Uuid::new_v4());
        let a = node(context.workspace_id, "Derivative");
        let b = node(context.workspace_id, "Rate of change");
        let c = node(context.workspace_id, "Tangent slope");
        engine
            .upsert(
                &context,
                GraphMutation {
                    nodes: vec![a.clone(), b.clone(), c.clone()],
                    edges: vec![
                        edge(context.workspace_id, a.id, b.id, "means"),
                        edge(context.workspace_id, b.id, c.id, "relates_to"),
                    ],
                },
            )
            .await
            .unwrap();
        let traversal = engine
            .traverse(
                &context,
                TraversalQuery {
                    start: vec![a.id],
                    direction: TraversalDirection::Outgoing,
                    max_depth: 2,
                    max_nodes: 10,
                    relations: BTreeSet::new(),
                    node_kinds: vec![],
                },
            )
            .await
            .unwrap();
        assert_eq!(traversal.nodes.len(), 3);
        assert_eq!(
            engine
                .shortest_path(&context, a.id, c.id, 2)
                .await
                .unwrap()
                .unwrap()
                .edges
                .len(),
            2
        );
    }

    #[tokio::test]
    async fn invalid_edges_are_rejected_before_any_write() {
        let engine = InMemoryKnowledgeGraphEngine::new().unwrap();
        let context = EngineRequestContext::new(Uuid::new_v4());
        let a = node(context.workspace_id, "A");
        let result = engine
            .upsert(
                &context,
                GraphMutation {
                    nodes: vec![a.clone()],
                    edges: vec![edge(context.workspace_id, a.id, NodeId::new(), "missing")],
                },
            )
            .await;
        assert!(result.is_err());
        assert!(engine.get_node(&context, a.id).await.is_err());
    }

    #[tokio::test]
    async fn workspaces_cannot_read_each_others_graphs() {
        let engine = InMemoryKnowledgeGraphEngine::new().unwrap();
        let first = EngineRequestContext::new(Uuid::new_v4());
        let second = EngineRequestContext::new(Uuid::new_v4());
        let hidden = node(first.workspace_id, "Private concept");
        engine
            .upsert(
                &first,
                GraphMutation {
                    nodes: vec![hidden.clone()],
                    edges: vec![],
                },
            )
            .await
            .unwrap();
        assert!(engine.get_node(&second, hidden.id).await.is_err());
    }

    #[tokio::test]
    async fn undirected_edges_are_traversable_from_either_endpoint() {
        let engine = InMemoryKnowledgeGraphEngine::new().unwrap();
        let context = EngineRequestContext::new(Uuid::new_v4());
        let left = node(context.workspace_id, "Left");
        let right = node(context.workspace_id, "Right");
        let mut relationship = edge(context.workspace_id, left.id, right.id, "related_to");
        relationship.directed = false;
        engine
            .upsert(
                &context,
                GraphMutation {
                    nodes: vec![left.clone(), right.clone()],
                    edges: vec![relationship],
                },
            )
            .await
            .unwrap();

        let result = engine
            .traverse(
                &context,
                TraversalQuery {
                    start: vec![right.id],
                    direction: TraversalDirection::Outgoing,
                    max_depth: 1,
                    max_nodes: 10,
                    relations: BTreeSet::new(),
                    node_kinds: Vec::new(),
                },
            )
            .await
            .unwrap();
        assert!(result.nodes.iter().any(|node| node.node.id == left.id));
    }
}
