"""Hybrid Research Gap Discovery Engine.

This agent discovers research gaps using a combination of:
1. Graph algorithms (community detection, centrality analysis)
2. Semantic clustering (GapSpotter algorithm)
3. LLM reasoning for validation

It identifies underexplored areas, missing connections, and potential
breakthrough opportunities in the research landscape.
"""

import logging
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from backend.agents.knowledge_graph import KnowledgeGraphAgent, NodeType, RelationType
from backend.utils.embeddings import EmbeddingService


logger = logging.getLogger(__name__)


@dataclass
class ResearchGap:
    """Represents a discovered research gap."""
    gap_id: str
    gap_type: str  # "missing_link", "underexplored_area", "isolated_cluster", "emerging_trend"
    title: str
    description: str
    entities: List[str]
    confidence: float
    potential_impact: str  # "low", "medium", "high", "breakthrough"
    supporting_evidence: List[str]
    related_papers: List[str]

    def to_dict(self) -> Dict:
        return {
            "gap_id": self.gap_id,
            "gap_type": self.gap_type,
            "title": self.title,
            "description": self.description,
            "entities": self.entities,
            "confidence": self.confidence,
            "potential_impact": self.potential_impact,
            "supporting_evidence": self.supporting_evidence,
            "related_papers": self.related_papers
        }


class GapDiscoveryAgent:
    """Discovers research gaps using hybrid graph + semantic + LLM approach."""

    def __init__(
        self,
        knowledge_graph: KnowledgeGraphAgent,
        embedding_service: Optional[EmbeddingService] = None
    ):
        """Initialize gap discovery agent.

        Args:
            knowledge_graph: Knowledge graph to analyze
            embedding_service: Service for semantic embeddings
        """
        self.kg = knowledge_graph
        self.embedding_service = embedding_service or EmbeddingService()
        self.gaps: List[ResearchGap] = []

        logger.info("GapDiscoveryAgent initialized")

    def discover_all_gaps(self) -> List[ResearchGap]:
        """Discover all types of research gaps.

        Returns:
            List of discovered gaps
        """
        logger.info("Starting comprehensive gap discovery...")

        self.gaps = []

        # 1. Graph-based gap discovery
        self.gaps.extend(self._find_missing_links())
        self.gaps.extend(self._find_isolated_clusters())
        self.gaps.extend(self._find_bridge_opportunities())

        # 2. Semantic gap discovery
        self.gaps.extend(self._find_semantic_gaps())

        # 3. Temporal gap discovery
        self.gaps.extend(self._find_emerging_trends())

        # 4. Structural gap discovery
        self.gaps.extend(self._find_underexplored_combinations())

        logger.info(f"Discovered {len(self.gaps)} total research gaps")

        # Rank by potential impact
        self.gaps.sort(key=lambda g: self._calculate_impact_score(g), reverse=True)

        return self.gaps

    def _find_missing_links(self) -> List[ResearchGap]:
        """Find missing links between entities that should be connected.

        Uses graph topology to identify pairs of nodes that:
        - Share many common neighbors
        - Are not directly connected
        - Would create valuable shortcuts if connected
        """
        gaps = []

        # Get all paper nodes
        paper_nodes = [n for n, data in self.kg.graph.nodes(data=True)
                      if data.get('type') == NodeType.PAPER]

        # Calculate resource allocation index for missing links
        # Higher score = more likely to be a valuable missing link
        missing_links = []

        for i, paper1 in enumerate(paper_nodes):
            for paper2 in paper_nodes[i+1:]:
                # Skip if already connected
                if self.kg.graph.has_edge(paper1, paper2) or self.kg.graph.has_edge(paper2, paper1):
                    continue

                # Calculate common neighbors
                neighbors1 = set(self.kg.graph.neighbors(paper1))
                neighbors2 = set(self.kg.graph.neighbors(paper2))
                common = neighbors1 & neighbors2

                if len(common) >= 3:  # At least 3 common entities
                    # Calculate resource allocation index
                    score = sum(1.0 / max(self.kg.graph.degree(n), 1) for n in common)

                    missing_links.append((paper1, paper2, common, score))

        # Convert top missing links to gaps
        missing_links.sort(key=lambda x: x[3], reverse=True)

        for paper1, paper2, common_entities, score in missing_links[:10]:  # Top 10
            paper1_name = self.kg.graph.nodes[paper1].get('name', paper1)
            paper2_name = self.kg.graph.nodes[paper2].get('name', paper2)

            entity_names = [self.kg.graph.nodes[e].get('name', e) for e in list(common_entities)[:5]]

            gap = ResearchGap(
                gap_id=f"missing_link_{len(gaps)}",
                gap_type="missing_link",
                title=f"Unexplored connection between research areas",
                description=(
                    f"Papers '{paper1_name[:50]}...' and '{paper2_name[:50]}...' "
                    f"share {len(common_entities)} common entities ({', '.join(entity_names[:3])}) "
                    f"but have not been directly connected. Exploring this relationship could yield insights."
                ),
                entities=list(common_entities),
                confidence=min(score / 5.0, 1.0),
                potential_impact="high" if score > 5 else "medium",
                supporting_evidence=[f"Shared entities: {', '.join(entity_names)}"],
                related_papers=[paper1, paper2]
            )
            gaps.append(gap)

        logger.info(f"Found {len(gaps)} missing link gaps")
        return gaps

    def _find_isolated_clusters(self) -> List[ResearchGap]:
        """Find isolated clusters of research that lack connections.

        Uses community detection to identify disconnected research areas.
        """
        gaps = []

        # Detect communities using Louvain algorithm
        try:
            # Convert to undirected for community detection
            undirected = self.kg.graph.to_undirected()
            communities = nx.community.louvain_communities(undirected)

            logger.info(f"Detected {len(communities)} research communities")

            # Find isolated communities (small size, few external connections)
            for i, community in enumerate(communities):
                if len(community) < 3 or len(community) > 20:
                    continue

                # Count external connections
                external_edges = 0
                for node in community:
                    neighbors = set(self.kg.graph.neighbors(node))
                    external = neighbors - community
                    external_edges += len(external)

                isolation_score = 1.0 - (external_edges / (len(community) * 10))

                if isolation_score > 0.6:  # Fairly isolated
                    # Get representative entities
                    entity_names = []
                    for node in list(community)[:5]:
                        name = self.kg.graph.nodes[node].get('name', node)
                        entity_names.append(name)

                    gap = ResearchGap(
                        gap_id=f"isolated_cluster_{i}",
                        gap_type="isolated_cluster",
                        title=f"Isolated research cluster needs integration",
                        description=(
                            f"A cluster of {len(community)} related research entities "
                            f"including {', '.join(entity_names[:3])} is relatively isolated "
                            f"from the broader research landscape. Connecting this cluster "
                            f"could lead to cross-pollination of ideas."
                        ),
                        entities=list(community),
                        confidence=isolation_score,
                        potential_impact="medium",
                        supporting_evidence=[
                            f"Cluster size: {len(community)} entities",
                            f"Isolation score: {isolation_score:.2f}"
                        ],
                        related_papers=[n for n in community if self.kg.nodes[n].node_type == NodeType.PAPER]
                    )
                    gaps.append(gap)

        except Exception as e:
            logger.warning(f"Community detection failed: {e}")

        logger.info(f"Found {len(gaps)} isolated cluster gaps")
        return gaps

    def _find_bridge_opportunities(self) -> List[ResearchGap]:
        """Find opportunities to bridge disconnected research areas.

        Identifies nodes with high betweenness centrality that could
        serve as bridges between communities.
        """
        gaps = []

        # Calculate betweenness centrality
        try:
            betweenness = nx.betweenness_centrality(self.kg.graph)

            # Find high-betweenness non-paper nodes
            bridge_nodes = sorted(
                [(node, score) for node, score in betweenness.items()
                 if score > 0.01 and self.kg.nodes[node].node_type != NodeType.PAPER],
                key=lambda x: x[1],
                reverse=True
            )[:5]

            for node, score in bridge_nodes:
                node_name = self.kg.nodes[node].name
                node_type = self.kg.nodes[node].node_type

                # Get papers using this bridge entity
                related_papers = []
                for pred in self.kg.graph.predecessors(node):
                    if self.kg.nodes[pred].node_type == NodeType.PAPER:
                        related_papers.append(pred)

                if len(related_papers) >= 2:
                    gap = ResearchGap(
                        gap_id=f"bridge_{node}",
                        gap_type="bridge_opportunity",
                        title=f"Key {node_type} '{node_name}' bridges multiple research areas",
                        description=(
                            f"The {node_type} '{node_name}' connects {len(related_papers)} "
                            f"papers and has high betweenness centrality ({score:.3f}). "
                            f"Further research on this entity could unlock new connections."
                        ),
                        entities=[node],
                        confidence=min(score * 10, 1.0),
                        potential_impact="high",
                        supporting_evidence=[
                            f"Betweenness centrality: {score:.3f}",
                            f"Connected papers: {len(related_papers)}"
                        ],
                        related_papers=related_papers[:5]
                    )
                    gaps.append(gap)

        except Exception as e:
            logger.warning(f"Betweenness centrality calculation failed: {e}")

        logger.info(f"Found {len(gaps)} bridge opportunity gaps")
        return gaps

    def _find_semantic_gaps(self) -> List[ResearchGap]:
        """Find semantic gaps using GapSpotter algorithm.

        Clusters entities by semantic similarity and identifies gaps
        where semantically similar entities lack connections.
        """
        gaps = []

        # Get all non-paper entities
        entities = [
            (node_id, node.name)
            for node_id, node in self.kg.nodes.items()
            if node.node_type in [NodeType.MODEL, NodeType.DATASET, NodeType.TASK, NodeType.METHOD]
        ]

        if len(entities) < 5:
            return gaps

        # Generate embeddings
        entity_ids = [e[0] for e in entities]
        entity_names = [e[1] for e in entities]

        try:
            embeddings = self.embedding_service.embed_texts(entity_names)
            embeddings_array = np.array(embeddings)

            # Cluster using DBSCAN
            clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
            labels = clustering.fit_predict(embeddings_array)

            # Analyze each cluster
            unique_labels = set(labels)
            unique_labels.discard(-1)  # Remove noise

            for label in unique_labels:
                cluster_indices = np.where(labels == label)[0]
                cluster_entities = [entity_ids[i] for i in cluster_indices]

                # Check if entities in cluster are connected in graph
                connected_pairs = 0
                total_pairs = 0

                for i, e1 in enumerate(cluster_entities):
                    for e2 in cluster_entities[i+1:]:
                        total_pairs += 1
                        # Check if connected through any path of length <= 2
                        try:
                            path_length = nx.shortest_path_length(self.kg.graph, e1, e2)
                            if path_length <= 2:
                                connected_pairs += 1
                        except nx.NetworkXNoPath:
                            pass

                # If semantically similar but not well connected = gap
                connectivity_ratio = connected_pairs / max(total_pairs, 1)

                if connectivity_ratio < 0.3 and len(cluster_entities) >= 3:
                    cluster_names = [entity_names[i] for i in cluster_indices]

                    gap = ResearchGap(
                        gap_id=f"semantic_gap_{label}",
                        gap_type="semantic_gap",
                        title=f"Semantically related entities lack research connections",
                        description=(
                            f"Entities {', '.join(cluster_names[:4])} are semantically similar "
                            f"but only {connectivity_ratio*100:.1f}% are connected in the research graph. "
                            f"This suggests an underexplored research area."
                        ),
                        entities=cluster_entities,
                        confidence=1.0 - connectivity_ratio,
                        potential_impact="medium",
                        supporting_evidence=[
                            f"Semantic cluster size: {len(cluster_entities)}",
                            f"Connectivity: {connectivity_ratio*100:.1f}%"
                        ],
                        related_papers=[]
                    )
                    gaps.append(gap)

        except Exception as e:
            logger.warning(f"Semantic gap discovery failed: {e}")

        logger.info(f"Found {len(gaps)} semantic gaps")
        return gaps

    def _find_emerging_trends(self) -> List[ResearchGap]:
        """Find emerging trends by analyzing temporal patterns."""
        gaps = []

        # Group papers by year
        papers_by_year = defaultdict(list)

        for node_id, node in self.kg.nodes.items():
            if node.node_type == NodeType.PAPER:
                year = node.properties.get('year')
                if year:
                    papers_by_year[year].append(node_id)

        if len(papers_by_year) < 2:
            return gaps

        # Find entities appearing frequently in recent papers
        recent_years = sorted(papers_by_year.keys(), reverse=True)[:2]
        recent_entities = defaultdict(int)

        for year in recent_years:
            for paper_id in papers_by_year[year]:
                # Get entities connected to this paper
                for entity in self.kg.graph.neighbors(paper_id):
                    if self.kg.nodes[entity].node_type in [NodeType.MODEL, NodeType.METHOD, NodeType.TASK]:
                        recent_entities[entity] += 1

        # Find entities with increasing popularity
        trending = sorted(
            [(e, count) for e, count in recent_entities.items() if count >= 2],
            key=lambda x: x[1],
            reverse=True
        )[:5]

        for entity, count in trending:
            entity_name = self.kg.nodes[entity].name
            entity_type = self.kg.nodes[entity].node_type

            gap = ResearchGap(
                gap_id=f"emerging_{entity}",
                gap_type="emerging_trend",
                title=f"Emerging trend: {entity_name}",
                description=(
                    f"The {entity_type} '{entity_name}' appears in {count} recent papers, "
                    f"indicating an emerging research trend worth exploring further."
                ),
                entities=[entity],
                confidence=min(count / 5.0, 1.0),
                potential_impact="high" if count >= 4 else "medium",
                supporting_evidence=[f"Appears in {count} recent papers"],
                related_papers=[]
            )
            gaps.append(gap)

        logger.info(f"Found {len(gaps)} emerging trend gaps")
        return gaps

    def _find_underexplored_combinations(self) -> List[ResearchGap]:
        """Find underexplored combinations of datasets, models, and tasks."""
        gaps = []

        # Get all datasets, models, and tasks
        datasets = [n for n in self.kg.nodes.values() if n.node_type == NodeType.DATASET]
        models = [n for n in self.kg.nodes.values() if n.node_type == NodeType.MODEL]
        tasks = [n for n in self.kg.nodes.values() if n.node_type == NodeType.TASK]

        # Find popular entities (used by multiple papers)
        popular_datasets = [d for d in datasets if len(d.properties.get('papers', [])) >= 2]
        popular_models = [m for m in models if len(m.properties.get('papers', [])) >= 2]
        popular_tasks = [t for t in tasks if len(t.properties.get('papers', [])) >= 2]

        # Find combinations that don't exist
        for dataset in popular_datasets[:3]:
            for model in popular_models[:3]:
                # Check if any paper uses both
                dataset_papers = set(dataset.properties.get('papers', []))
                model_papers = set(model.properties.get('papers', []))
                common_papers = dataset_papers & model_papers

                if len(common_papers) == 0:
                    # Underexplored combination
                    gap = ResearchGap(
                        gap_id=f"combination_{dataset.node_id}_{model.node_id}",
                        gap_type="underexplored_combination",
                        title=f"Unexplored: {model.name} on {dataset.name}",
                        description=(
                            f"The model '{model.name}' has not been evaluated on "
                            f"the dataset '{dataset.name}', despite both being "
                            f"popular in the research community."
                        ),
                        entities=[dataset.node_id, model.node_id],
                        confidence=0.7,
                        potential_impact="medium",
                        supporting_evidence=[
                            f"{dataset.name} used in {len(dataset_papers)} papers",
                            f"{model.name} used in {len(model_papers)} papers",
                            "No papers combine both"
                        ],
                        related_papers=[]
                    )
                    gaps.append(gap)

        logger.info(f"Found {len(gaps)} underexplored combination gaps")
        return gaps[:10]  # Limit to avoid explosion

    def _calculate_impact_score(self, gap: ResearchGap) -> float:
        """Calculate potential impact score for a gap.

        Args:
            gap: Research gap to score

        Returns:
            Impact score (0-1)
        """
        # Base score from confidence
        score = gap.confidence * 0.4

        # Add points for impact level
        impact_scores = {
            "low": 0.1,
            "medium": 0.3,
            "high": 0.5,
            "breakthrough": 0.8
        }
        score += impact_scores.get(gap.potential_impact, 0.2)

        # Add points for number of entities involved
        score += min(len(gap.entities) / 10.0, 0.2)

        return min(score, 1.0)

    def get_gaps_by_type(self, gap_type: str) -> List[ResearchGap]:
        """Get gaps of a specific type.

        Args:
            gap_type: Type of gap to filter

        Returns:
            List of gaps of that type
        """
        return [g for g in self.gaps if g.gap_type == gap_type]

    def get_top_gaps(self, n: int = 10) -> List[ResearchGap]:
        """Get top N gaps by impact score.

        Args:
            n: Number of gaps to return

        Returns:
            List of top gaps
        """
        return self.gaps[:n]

    def export_gaps(self) -> List[Dict]:
        """Export gaps to dictionary format.

        Returns:
            List of gap dictionaries
        """
        return [gap.to_dict() for gap in self.gaps]
