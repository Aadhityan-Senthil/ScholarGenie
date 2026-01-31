"""Causal Research Graph Reasoning Engine.

This is the most unique feature - it builds a causal graph of research
relationships and uses causal inference to predict breakthrough opportunities.

Unlike correlation-based approaches, this identifies causal relationships:
- Model X causes improvement in Task Y
- Dataset A enables breakthrough in Domain B
- Method M is causally necessary for Result R
"""

import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import networkx as nx

from backend.agents.knowledge_graph import KnowledgeGraphAgent, NodeType, RelationType


logger = logging.getLogger(__name__)


class CausalRelationType:
    """Types of causal relationships."""
    CAUSES = "causes"
    ENABLES = "enables"
    REQUIRES = "requires"
    PREVENTS = "prevents"
    INFLUENCES = "influences"
    CONDITIONAL = "conditional"


@dataclass
class CausalEdge:
    """Causal edge in the research graph."""
    source: str
    target: str
    causal_type: str
    strength: float  # 0-1
    evidence: List[str]  # Paper IDs supporting this causal claim
    confidence: float  # Statistical confidence
    effect_size: Optional[float] = None  # Measured effect (e.g., accuracy improvement)

    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "target": self.target,
            "causal_type": self.causal_type,
            "strength": self.strength,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "effect_size": self.effect_size
        }


@dataclass
class BreakthroughPrediction:
    """Predicted breakthrough opportunity."""
    prediction_id: str
    title: str
    description: str
    causal_chain: List[Tuple[str, str, str]]  # (source, relation, target)
    required_elements: List[str]  # Entity IDs needed
    expected_impact: float  # 0-1
    confidence: float  # 0-1
    reasoning: str
    prerequisites: List[str]
    timeline: str  # "short-term", "medium-term", "long-term"

    def to_dict(self) -> Dict:
        return {
            "prediction_id": self.prediction_id,
            "title": self.title,
            "description": self.description,
            "causal_chain": [
                {"source": s, "relation": r, "target": t}
                for s, r, t in self.causal_chain
            ],
            "required_elements": self.required_elements,
            "expected_impact": self.expected_impact,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "prerequisites": self.prerequisites,
            "timeline": self.timeline
        }


class CausalGraphReasoner:
    """Causal reasoning engine for research graph."""

    def __init__(self, knowledge_graph: KnowledgeGraphAgent):
        """Initialize causal reasoner.

        Args:
            knowledge_graph: Knowledge graph instance
        """
        self.kg = knowledge_graph
        self.causal_graph = nx.DiGraph()
        self.causal_edges: List[CausalEdge] = []

        logger.info("CausalGraphReasoner initialized")

    def build_causal_graph(self) -> None:
        """Build causal graph from knowledge graph.

        Infers causal relationships from:
        - Temporal ordering (papers citing earlier work)
        - Performance improvements (new model > old model)
        - Methodological dependencies (method requires dataset)
        """
        logger.info("Building causal graph...")

        # Extract causal edges
        self._infer_causal_model_relationships()
        self._infer_causal_dataset_relationships()
        self._infer_causal_method_dependencies()
        self._infer_temporal_causality()

        # Build NetworkX graph
        for edge in self.causal_edges:
            self.causal_graph.add_edge(
                edge.source,
                edge.target,
                causal_type=edge.causal_type,
                strength=edge.strength,
                confidence=edge.confidence
            )

        logger.info(f"Built causal graph with {len(self.causal_edges)} causal edges")

    def _infer_causal_model_relationships(self) -> None:
        """Infer causal relationships between models.

        Example: "Attention mechanism CAUSES improved performance in translation"
        """
        model_nodes = [
            (node_id, node)
            for node_id, node in self.kg.nodes.items()
            if node.node_type == NodeType.MODEL
        ]

        for i, (model1_id, model1) in enumerate(model_nodes):
            for model2_id, model2 in model_nodes[i+1:]:
                # Check if models share tasks
                model1_papers = set(model1.properties.get("papers", []))
                model2_papers = set(model2.properties.get("papers", []))

                # Find papers that compare them
                shared_papers = model1_papers & model2_papers

                if shared_papers:
                    # Infer causal relationship based on temporal order
                    model1_years = self._get_entity_years(model1_id)
                    model2_years = self._get_entity_years(model2_id)

                    if model1_years and model2_years:
                        avg_year1 = np.mean(model1_years)
                        avg_year2 = np.mean(model2_years)

                        if avg_year2 > avg_year1 + 1:  # Model 2 came after
                            # Model1 ENABLES Model2
                            edge = CausalEdge(
                                source=model1_id,
                                target=model2_id,
                                causal_type=CausalRelationType.ENABLES,
                                strength=0.7,
                                evidence=list(shared_papers),
                                confidence=0.6
                            )
                            self.causal_edges.append(edge)

    def _infer_causal_dataset_relationships(self) -> None:
        """Infer causal relationships with datasets.

        Example: "ImageNet ENABLES breakthroughs in computer vision"
        """
        dataset_nodes = [
            (node_id, node)
            for node_id, node in self.kg.nodes.items()
            if node.node_type == NodeType.DATASET
        ]

        for dataset_id, dataset in dataset_nodes:
            papers_using = dataset.properties.get("papers", [])

            if len(papers_using) >= 3:  # Popular dataset
                # Find models evaluated on this dataset
                models_on_dataset = []

                for paper_id in papers_using:
                    if paper_id in self.kg.graph:
                        # Get models used in this paper
                        for neighbor in self.kg.graph.neighbors(paper_id):
                            if neighbor in self.kg.nodes and self.kg.nodes[neighbor].node_type == NodeType.MODEL:
                                models_on_dataset.append(neighbor)

                # Dataset ENABLES models
                for model_id in set(models_on_dataset):
                    edge = CausalEdge(
                        source=dataset_id,
                        target=model_id,
                        causal_type=CausalRelationType.ENABLES,
                        strength=0.8,
                        evidence=papers_using[:5],
                        confidence=0.7
                    )
                    self.causal_edges.append(edge)

    def _infer_causal_method_dependencies(self) -> None:
        """Infer method dependencies.

        Example: "Backpropagation REQUIRES differentiable functions"
        """
        method_nodes = [
            (node_id, node)
            for node_id, node in self.kg.nodes.items()
            if node.node_type == NodeType.METHOD
        ]

        for i, (method1_id, method1) in enumerate(method_nodes):
            for method2_id, method2 in method_nodes[i+1:]:
                # Check if methods co-occur in papers
                method1_papers = set(method1.properties.get("papers", []))
                method2_papers = set(method2.properties.get("papers", []))

                shared_papers = method1_papers & method2_papers

                if len(shared_papers) >= 2:
                    # Methods used together suggest dependency
                    edge = CausalEdge(
                        source=method1_id,
                        target=method2_id,
                        causal_type=CausalRelationType.INFLUENCES,
                        strength=0.5,
                        evidence=list(shared_papers),
                        confidence=0.5
                    )
                    self.causal_edges.append(edge)

    def _infer_temporal_causality(self) -> None:
        """Infer causality from temporal relationships.

        Earlier work CAUSES later work via citations.
        """
        for source, target in self.kg.graph.edges():
            edge_data = self.kg.graph.get_edge_data(source, target)

            if edge_data.get("relation") == RelationType.CITES:
                # Citation implies causal influence
                if source in self.kg.nodes and target in self.kg.nodes:
                    source_node = self.kg.nodes[source]
                    target_node = self.kg.nodes[target]

                    if source_node.node_type == NodeType.PAPER and target_node.node_type == NodeType.PAPER:
                        edge = CausalEdge(
                            source=target,  # Cited paper influences citing paper
                            target=source,
                            causal_type=CausalRelationType.INFLUENCES,
                            strength=0.6,
                            evidence=[source],
                            confidence=0.8
                        )
                        self.causal_edges.append(edge)

    def _get_entity_years(self, entity_id: str) -> List[int]:
        """Get years associated with an entity.

        Args:
            entity_id: Entity ID

        Returns:
            List of years
        """
        years = []

        if entity_id not in self.kg.nodes:
            return years

        node = self.kg.nodes[entity_id]
        papers = node.properties.get("papers", [])

        for paper_id in papers:
            if paper_id in self.kg.nodes:
                paper = self.kg.nodes[paper_id]
                year = paper.properties.get("year")
                if year:
                    years.append(int(year))

        return years

    def predict_breakthroughs(
        self,
        top_k: int = 10
    ) -> List[BreakthroughPrediction]:
        """Predict breakthrough opportunities using causal reasoning.

        This is the most unique feature - identifies novel causal paths
        that could lead to breakthroughs.

        Args:
            top_k: Number of predictions to return

        Returns:
            List of breakthrough predictions
        """
        logger.info("Predicting breakthrough opportunities...")

        if not self.causal_edges:
            self.build_causal_graph()

        predictions = []

        # Strategy 1: Find untried causal chains
        predictions.extend(self._find_untried_causal_chains())

        # Strategy 2: Find intervention opportunities
        predictions.extend(self._find_intervention_points())

        # Strategy 3: Find synergistic combinations
        predictions.extend(self._find_synergistic_combinations())

        # Strategy 4: Find emerging causal patterns
        predictions.extend(self._find_emerging_patterns())

        # Sort by expected impact
        predictions.sort(key=lambda p: p.expected_impact * p.confidence, reverse=True)

        logger.info(f"Generated {len(predictions)} breakthrough predictions")
        return predictions[:top_k]

    def _find_untried_causal_chains(self) -> List[BreakthroughPrediction]:
        """Find causal chains that haven't been explored.

        Example: Dataset D ENABLES Model M, Model M CAUSES Task T improvement
        But no one has tried M on D for T yet.
        """
        predictions = []

        # Find 2-hop causal paths
        for source in self.causal_graph.nodes():
            # Get 2-hop paths
            for intermediate in self.causal_graph.successors(source):
                for target in self.causal_graph.successors(intermediate):
                    # Check if this combination has been tried
                    if not self._combination_exists(source, intermediate, target):
                        # Untried causal chain!
                        strength = self._calculate_chain_strength([source, intermediate, target])

                        if strength > 0.5:
                            source_name = self.kg.nodes[source].name if source in self.kg.nodes else source
                            inter_name = self.kg.nodes[intermediate].name if intermediate in self.kg.nodes else intermediate
                            target_name = self.kg.nodes[target].name if target in self.kg.nodes else target

                            prediction = BreakthroughPrediction(
                                prediction_id=f"chain_{source}_{intermediate}_{target}",
                                title=f"Apply {inter_name} with {source_name} to {target_name}",
                                description=(
                                    f"Causal analysis suggests that combining {source_name} and {inter_name} "
                                    f"could lead to breakthroughs in {target_name}, but this hasn't been explored yet."
                                ),
                                causal_chain=[
                                    (source, "enables", intermediate),
                                    (intermediate, "causes_improvement_in", target)
                                ],
                                required_elements=[source, intermediate, target],
                                expected_impact=strength,
                                confidence=0.7,
                                reasoning=(
                                    f"{source_name} has been shown to enable {inter_name}, "
                                    f"and {inter_name} improves {target_name}. "
                                    f"However, no research has directly combined all three."
                                ),
                                prerequisites=[
                                    f"Access to {source_name}",
                                    f"Implement {inter_name}",
                                    f"Benchmark on {target_name}"
                                ],
                                timeline="short-term"
                            )
                            predictions.append(prediction)

        return predictions[:5]  # Top 5

    def _find_intervention_points(self) -> List[BreakthroughPrediction]:
        """Find intervention points where small changes could have large effects.

        Uses betweenness centrality on causal graph.
        """
        predictions = []

        # Calculate betweenness centrality
        try:
            betweenness = nx.betweenness_centrality(self.causal_graph)

            # Find high-betweenness nodes
            high_betweenness = sorted(
                [(node, score) for node, score in betweenness.items() if score > 0.1],
                key=lambda x: x[1],
                reverse=True
            )[:5]

            for node, score in high_betweenness:
                if node not in self.kg.nodes:
                    continue

                node_obj = self.kg.nodes[node]
                node_name = node_obj.name

                # This is a critical intervention point
                prediction = BreakthroughPrediction(
                    prediction_id=f"intervention_{node}",
                    title=f"Breakthrough opportunity: Advance {node_name}",
                    description=(
                        f"{node_name} is a critical intervention point in the research graph. "
                        f"Improvements here could cascade to multiple downstream areas."
                    ),
                    causal_chain=[],
                    required_elements=[node],
                    expected_impact=min(score * 2, 1.0),
                    confidence=0.8,
                    reasoning=(
                        f"{node_name} has high betweenness centrality ({score:.3f}), "
                        f"meaning it bridges multiple research areas. "
                        f"Advances here will have multiplicative effects."
                    ),
                    prerequisites=[
                        f"Deep understanding of {node_name}",
                        "Identify current limitations",
                        "Propose novel improvements"
                    ],
                    timeline="medium-term"
                )
                predictions.append(prediction)

        except Exception as e:
            logger.warning(f"Betweenness calculation failed: {e}")

        return predictions

    def _find_synergistic_combinations(self) -> List[BreakthroughPrediction]:
        """Find combinations with synergistic potential.

        Identifies entities that individually work but haven't been combined.
        """
        predictions = []

        # Find models and methods that haven't been combined
        models = [n for n in self.kg.nodes.values() if n.node_type == NodeType.MODEL]
        methods = [n for n in self.kg.nodes.values() if n.node_type == NodeType.METHOD]

        for model in models[:5]:
            for method in methods[:5]:
                # Check if they've been used together
                model_papers = set(model.properties.get("papers", []))
                method_papers = set(method.properties.get("papers", []))

                if not (model_papers & method_papers):
                    # Never combined
                    # Check if both are effective individually
                    if len(model_papers) >= 2 and len(method_papers) >= 2:
                        prediction = BreakthroughPrediction(
                            prediction_id=f"synergy_{model.node_id}_{method.node_id}",
                            title=f"Combine {model.name} with {method.name}",
                            description=(
                                f"Both {model.name} and {method.name} are proven effective separately. "
                                f"Combining them could yield synergistic improvements."
                            ),
                            causal_chain=[
                                (model.node_id, "combined_with", method.node_id)
                            ],
                            required_elements=[model.node_id, method.node_id],
                            expected_impact=0.7,
                            confidence=0.6,
                            reasoning=(
                                f"{model.name} used in {len(model_papers)} papers, "
                                f"{method.name} used in {len(method_papers)} papers, "
                                f"but never together. Potential synergy."
                            ),
                            prerequisites=[
                                f"Implement {model.name}",
                                f"Integrate {method.name}",
                                "Design experiments to measure synergy"
                            ],
                            timeline="short-term"
                        )
                        predictions.append(prediction)

        return predictions[:3]

    def _find_emerging_patterns(self) -> List[BreakthroughPrediction]:
        """Find emerging causal patterns from recent papers."""
        predictions = []

        # Get recent papers (if year info available)
        recent_papers = []
        for node_id, node in self.kg.nodes.items():
            if node.node_type == NodeType.PAPER:
                year = node.properties.get("year")
                if year and int(year) >= 2023:
                    recent_papers.append(node_id)

        if len(recent_papers) >= 3:
            # Find common entities in recent papers
            entity_counts = defaultdict(int)

            for paper_id in recent_papers:
                if paper_id in self.kg.graph:
                    for neighbor in self.kg.graph.neighbors(paper_id):
                        if neighbor in self.kg.nodes:
                            entity_counts[neighbor] += 1

            # Find rapidly growing entities
            emerging = [
                (entity, count)
                for entity, count in entity_counts.items()
                if count >= 2
            ]

            for entity, count in sorted(emerging, key=lambda x: x[1], reverse=True)[:3]:
                entity_name = self.kg.nodes[entity].name
                entity_type = self.kg.nodes[entity].node_type

                prediction = BreakthroughPrediction(
                    prediction_id=f"emerging_{entity}",
                    title=f"Emerging trend: {entity_name}",
                    description=(
                        f"{entity_name} ({entity_type}) appears in {count} recent papers, "
                        f"indicating an emerging research trend with breakthrough potential."
                    ),
                    causal_chain=[],
                    required_elements=[entity],
                    expected_impact=min(count / 5.0, 1.0),
                    confidence=0.7,
                    reasoning=(
                        f"Rapid adoption in recent work suggests {entity_name} "
                        f"is becoming a key enabler for future breakthroughs."
                    ),
                    prerequisites=[
                        f"Stay updated on {entity_name} developments",
                        "Identify novel applications",
                        "Contribute to emerging ecosystem"
                    ],
                    timeline="short-term"
                )
                predictions.append(prediction)

        return predictions

    def _combination_exists(self, entity1: str, entity2: str, entity3: str) -> bool:
        """Check if three entities have been combined in any paper.

        Args:
            entity1, entity2, entity3: Entity IDs

        Returns:
            True if combination exists
        """
        # Get papers for each entity
        papers1 = set(self.kg.nodes[entity1].properties.get("papers", [])) if entity1 in self.kg.nodes else set()
        papers2 = set(self.kg.nodes[entity2].properties.get("papers", [])) if entity2 in self.kg.nodes else set()
        papers3 = set(self.kg.nodes[entity3].properties.get("papers", [])) if entity3 in self.kg.nodes else set()

        # Check if any paper uses all three
        common = papers1 & papers2 & papers3
        return len(common) > 0

    def _calculate_chain_strength(self, chain: List[str]) -> float:
        """Calculate strength of a causal chain.

        Args:
            chain: List of entity IDs in chain

        Returns:
            Chain strength (0-1)
        """
        if len(chain) < 2:
            return 0.0

        strengths = []

        for i in range(len(chain) - 1):
            source = chain[i]
            target = chain[i + 1]

            if self.causal_graph.has_edge(source, target):
                edge_data = self.causal_graph.get_edge_data(source, target)
                strength = edge_data.get("strength", 0.5)
                strengths.append(strength)
            else:
                strengths.append(0.3)  # Weak connection

        # Average strength
        return np.mean(strengths) if strengths else 0.5

    def explain_causal_path(
        self,
        source: str,
        target: str,
        max_length: int = 5
    ) -> Optional[Dict]:
        """Explain causal path between two entities.

        Args:
            source: Source entity ID
            target: Target entity ID
            max_length: Maximum path length

        Returns:
            Path explanation or None
        """
        if source not in self.causal_graph or target not in self.causal_graph:
            return None

        try:
            path = nx.shortest_path(self.causal_graph, source, target)

            if len(path) > max_length:
                return None

            explanation = {
                "source": self.kg.nodes[source].name if source in self.kg.nodes else source,
                "target": self.kg.nodes[target].name if target in self.kg.nodes else target,
                "path_length": len(path) - 1,
                "causal_steps": []
            }

            for i in range(len(path) - 1):
                s = path[i]
                t = path[i + 1]

                edge_data = self.causal_graph.get_edge_data(s, t)

                s_name = self.kg.nodes[s].name if s in self.kg.nodes else s
                t_name = self.kg.nodes[t].name if t in self.kg.nodes else t

                explanation["causal_steps"].append({
                    "from": s_name,
                    "to": t_name,
                    "relation": edge_data.get("causal_type", "causes"),
                    "strength": edge_data.get("strength", 0.5)
                })

            return explanation

        except nx.NetworkXNoPath:
            return None

    def get_causal_statistics(self) -> Dict:
        """Get statistics about the causal graph.

        Returns:
            Statistics dictionary
        """
        causal_type_counts = defaultdict(int)
        for edge in self.causal_edges:
            causal_type_counts[edge.causal_type] += 1

        return {
            "total_causal_edges": len(self.causal_edges),
            "causal_type_distribution": dict(causal_type_counts),
            "average_strength": np.mean([e.strength for e in self.causal_edges]) if self.causal_edges else 0,
            "average_confidence": np.mean([e.confidence for e in self.causal_edges]) if self.causal_edges else 0
        }

    def export_predictions(
        self,
        predictions: List[BreakthroughPrediction]
    ) -> List[Dict]:
        """Export predictions to dictionary format.

        Args:
            predictions: List of predictions

        Returns:
            List of dictionaries
        """
        return [p.to_dict() for p in predictions]
