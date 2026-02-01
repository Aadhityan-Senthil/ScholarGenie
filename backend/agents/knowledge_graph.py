"""Knowledge Graph Construction Agent.

This agent builds a scientific knowledge graph from ingested papers,
extracting entities (models, datasets, metrics, tasks, concepts) and
their relationships to create a queryable graph structure.
"""

import re
import logging
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import networkx as nx
from dataclasses import dataclass
from enum import Enum

from backend.utils.metadata import PaperMetadata, ExtractedData


logger = logging.getLogger(__name__)


class NodeType(str, Enum):
    """Types of nodes in the knowledge graph."""
    PAPER = "paper"
    MODEL = "model"
    DATASET = "dataset"
    METRIC = "metric"
    TASK = "task"
    METHOD = "method"
    CONCEPT = "concept"
    AUTHOR = "author"
    VENUE = "venue"


class RelationType(str, Enum):
    """Types of relationships in the knowledge graph."""
    USES = "uses"
    EVALUATED_ON = "evaluated_on"
    MEASURES = "measures"
    ADDRESSES = "addresses"
    IMPROVES = "improves"
    BASED_ON = "based_on"
    CITES = "cites"
    AUTHORED_BY = "authored_by"
    PUBLISHED_IN = "published_in"
    SIMILAR_TO = "similar_to"
    DEPENDS_ON = "depends_on"


@dataclass
class GraphNode:
    """Node in the knowledge graph."""
    node_id: str
    node_type: NodeType
    name: str
    properties: Dict[str, any]

    def __hash__(self):
        return hash(self.node_id)


@dataclass
class GraphEdge:
    """Edge in the knowledge graph."""
    source: str
    target: str
    relation: RelationType
    properties: Dict[str, any]
    confidence: float = 1.0


class KnowledgeGraphAgent:
    """Constructs and manages a scientific knowledge graph."""

    def __init__(self):
        """Initialize the knowledge graph agent."""
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []

        # Entity extraction patterns
        self.model_patterns = [
            r'\b(BERT|GPT-?\d*|T5|BART|RoBERTa|XLNet|ELECTRA|DeBERTa|ALBERT)\b',
            r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)*(?:Net|Transformer|Former|Model|LM))\b',
            r'\b(Transformer|LSTM|GRU|CNN|ResNet|VGG|Inception|MobileNet)\b',
        ]

        self.dataset_patterns = [
            r'\b([A-Z][A-Z0-9]+(?:-\d+)?)\b(?=\s+(?:dataset|corpus|benchmark))',
            r'\b(ImageNet|COCO|GLUE|SQuAD|WMT|CoNLL|MNIST|CIFAR)\b',
            r'(?:trained|evaluated|tested)\s+on\s+(?:the\s+)?([A-Z][A-Za-z0-9\-]+)',
        ]

        self.metric_patterns = [
            r'\b(accuracy|precision|recall|F1(?:-score)?|BLEU|ROUGE|perplexity|AUC)\b',
            r'\b(top-[1-9]|MAP|mAP|IoU|METEOR|CIDEr|SPICE)\b',
        ]

        self.task_patterns = [
            r'\b(classification|segmentation|detection|generation|translation|summarization)\b',
            r'\b(question\s+answering|named\s+entity\s+recognition|NER|POS\s+tagging)\b',
            r'\b(sentiment\s+analysis|machine\s+translation|text\s+generation)\b',
        ]

        logger.info("KnowledgeGraphAgent initialized")

    def build_graph_from_paper(
        self,
        paper: PaperMetadata,
        extracted_data: ExtractedData
    ) -> None:
        """Build knowledge graph from a single paper.

        Args:
            paper: Parsed paper metadata
            extracted_data: Extracted structured data
        """
        logger.info(f"Building graph for paper: {paper.paper_id}")

        # Create paper node
        paper_node = GraphNode(
            node_id=paper.paper_id,
            node_type=NodeType.PAPER,
            name=paper.title,
            properties={
                "abstract": paper.abstract,
                "year": paper.year,
                "citation_count": paper.citation_count,
            }
        )
        self.add_node(paper_node)

        # Extract and link entities
        self._extract_models(paper, extracted_data)
        self._extract_datasets(paper, extracted_data)
        self._extract_metrics(paper, extracted_data)
        self._extract_tasks(paper, extracted_data)
        self._extract_methods(paper, extracted_data)
        self._extract_concepts(paper)
        self._extract_authors(paper)
        self._extract_citations(paper)

        logger.info(f"Graph built: {len(self.nodes)} nodes, {len(self.edges)} edges")

    def _extract_models(
        self,
        paper: PaperMetadata,
        extracted_data: ExtractedData
    ) -> None:
        """Extract model entities and create nodes."""
        models = set(extracted_data.models) if extracted_data.models else set()

        # Also extract from full text using patterns
        full_text = paper.get_full_text()
        for pattern in self.model_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            models.update(matches)

        for model_name in models:
            if not model_name or len(model_name) < 2:
                continue

            node_id = f"model_{model_name.lower().replace(' ', '_')}"
            node = GraphNode(
                node_id=node_id,
                node_type=NodeType.MODEL,
                name=model_name,
                properties={"papers": [paper.paper_id]}
            )
            self.add_node(node)

            # Create edge: paper USES model
            edge = GraphEdge(
                source=paper.paper_id,
                target=node_id,
                relation=RelationType.USES,
                properties={},
                confidence=0.8
            )
            self.add_edge(edge)

    def _extract_datasets(
        self,
        paper: PaperMetadata,
        extracted_data: ExtractedData
    ) -> None:
        """Extract dataset entities and create nodes."""
        datasets = set(extracted_data.datasets) if extracted_data.datasets else set()

        full_text = paper.get_full_text()
        for pattern in self.dataset_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            datasets.update(matches)

        for dataset_name in datasets:
            if not dataset_name or len(dataset_name) < 2:
                continue

            node_id = f"dataset_{dataset_name.lower().replace(' ', '_')}"
            node = GraphNode(
                node_id=node_id,
                node_type=NodeType.DATASET,
                name=dataset_name,
                properties={"papers": [paper.paper_id]}
            )
            self.add_node(node)

            # Create edge: paper EVALUATED_ON dataset
            edge = GraphEdge(
                source=paper.paper_id,
                target=node_id,
                relation=RelationType.EVALUATED_ON,
                properties={},
                confidence=0.8
            )
            self.add_edge(edge)

    def _extract_metrics(
        self,
        paper: PaperMetadata,
        extracted_data: ExtractedData
    ) -> None:
        """Extract metric entities and create nodes."""
        metrics = set(extracted_data.metrics) if extracted_data.metrics else set()

        full_text = paper.get_full_text()
        for pattern in self.metric_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            metrics.update(matches)

        for metric_name in metrics:
            if not metric_name or len(metric_name) < 2:
                continue

            node_id = f"metric_{metric_name.lower().replace(' ', '_').replace('-', '_')}"
            node = GraphNode(
                node_id=node_id,
                node_type=NodeType.METRIC,
                name=metric_name,
                properties={"papers": [paper.paper_id]}
            )
            self.add_node(node)

            # Create edge: paper MEASURES metric
            edge = GraphEdge(
                source=paper.paper_id,
                target=node_id,
                relation=RelationType.MEASURES,
                properties={},
                confidence=0.7
            )
            self.add_edge(edge)

    def _extract_tasks(
        self,
        paper: PaperMetadata,
        extracted_data: ExtractedData
    ) -> None:
        """Extract task entities and create nodes."""
        tasks = set()

        full_text = paper.get_full_text()
        for pattern in self.task_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            tasks.update(matches)

        for task_name in tasks:
            if not task_name or len(task_name) < 3:
                continue

            node_id = f"task_{task_name.lower().replace(' ', '_')}"
            node = GraphNode(
                node_id=node_id,
                node_type=NodeType.TASK,
                name=task_name,
                properties={"papers": [paper.paper_id]}
            )
            self.add_node(node)

            # Create edge: paper ADDRESSES task
            edge = GraphEdge(
                source=paper.paper_id,
                target=node_id,
                relation=RelationType.ADDRESSES,
                properties={},
                confidence=0.7
            )
            self.add_edge(edge)

    def _extract_methods(
        self,
        paper: PaperMetadata,
        extracted_data: ExtractedData
    ) -> None:
        """Extract method entities and create nodes."""
        methods = set(extracted_data.methods) if extracted_data.methods else set()

        for method_name in methods:
            if not method_name or len(method_name) < 3:
                continue

            node_id = f"method_{method_name.lower().replace(' ', '_')}"
            node = GraphNode(
                node_id=node_id,
                node_type=NodeType.METHOD,
                name=method_name,
                properties={"papers": [paper.paper_id]}
            )
            self.add_node(node)

            # Create edge: paper USES method
            edge = GraphEdge(
                source=paper.paper_id,
                target=node_id,
                relation=RelationType.USES,
                properties={},
                confidence=0.6
            )
            self.add_edge(edge)

    def _extract_concepts(self, paper: PaperMetadata) -> None:
        """Extract key concepts from paper."""
        # Extract from title and abstract
        text = f"{paper.title} {paper.abstract}"

        # Simple concept extraction (can be enhanced with NLP)
        concept_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b(?=\s+(?:is|are|has|have))',
            r'\b(neural\s+\w+|deep\s+\w+|machine\s+\w+)\b',
        ]

        concepts = set()
        for pattern in concept_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            concepts.update(matches)

        for concept_name in list(concepts)[:5]:  # Limit to top 5
            if not concept_name or len(concept_name) < 5:
                continue

            node_id = f"concept_{concept_name.lower().replace(' ', '_')}"
            node = GraphNode(
                node_id=node_id,
                node_type=NodeType.CONCEPT,
                name=concept_name,
                properties={"papers": [paper.paper_id]}
            )
            self.add_node(node)

    def _extract_authors(self, paper: PaperMetadata) -> None:
        """Extract author entities and create nodes."""
        for author in paper.authors:
            node_id = f"author_{author.name.lower().replace(' ', '_')}"
            node = GraphNode(
                node_id=node_id,
                node_type=NodeType.AUTHOR,
                name=author.name,
                properties={
                    "email": author.email,
                    "affiliation": author.affiliation,
                    "papers": [paper.paper_id]
                }
            )
            self.add_node(node)

            # Create edge: paper AUTHORED_BY author
            edge = GraphEdge(
                source=paper.paper_id,
                target=node_id,
                relation=RelationType.AUTHORED_BY,
                properties={},
                confidence=1.0
            )
            self.add_edge(edge)

    def _extract_citations(self, paper: PaperMetadata) -> None:
        """Extract citation relationships."""
        for reference in paper.references:
            cited_paper_id = reference.paper_id or f"ref_{reference.title.lower().replace(' ', '_')[:50]}"

            # Create cited paper node if not exists
            node = GraphNode(
                node_id=cited_paper_id,
                node_type=NodeType.PAPER,
                name=reference.title,
                properties={
                    "authors": reference.authors,
                    "year": reference.year,
                }
            )
            self.add_node(node)

            # Create edge: paper CITES cited_paper
            edge = GraphEdge(
                source=paper.paper_id,
                target=cited_paper_id,
                relation=RelationType.CITES,
                properties={},
                confidence=1.0
            )
            self.add_edge(edge)

    def add_node(self, node: GraphNode) -> None:
        """Add or update a node in the graph."""
        if node.node_id in self.nodes:
            # Merge properties
            existing = self.nodes[node.node_id]
            if "papers" in node.properties and "papers" in existing.properties:
                existing.properties["papers"].extend(node.properties["papers"])
                existing.properties["papers"] = list(set(existing.properties["papers"]))
        else:
            self.nodes[node.node_id] = node
            self.graph.add_node(
                node.node_id,
                type=node.node_type,
                name=node.name,
                **node.properties
            )

    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)
        self.graph.add_edge(
            edge.source,
            edge.target,
            relation=edge.relation,
            confidence=edge.confidence,
            **edge.properties
        )

    def get_subgraph(
        self,
        center_node: str,
        depth: int = 2,
        relation_types: Optional[List[RelationType]] = None
    ) -> nx.DiGraph:
        """Get a subgraph around a center node.

        Args:
            center_node: Node ID to center the subgraph
            depth: How many hops from the center
            relation_types: Filter by specific relation types

        Returns:
            Subgraph as NetworkX DiGraph
        """
        if center_node not in self.graph:
            return nx.DiGraph()

        # Get nodes within depth
        ego_graph = nx.ego_graph(self.graph, center_node, radius=depth)

        # Filter by relation types if specified
        if relation_types:
            edges_to_remove = []
            for u, v, data in ego_graph.edges(data=True):
                if data.get('relation') not in relation_types:
                    edges_to_remove.append((u, v))
            ego_graph.remove_edges_from(edges_to_remove)

        return ego_graph

    def find_similar_papers(
        self,
        paper_id: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Find similar papers based on shared entities.

        Args:
            paper_id: Source paper ID
            top_k: Number of similar papers to return

        Returns:
            List of (paper_id, similarity_score) tuples
        """
        if paper_id not in self.graph:
            return []

        # Get all entities connected to this paper
        neighbors = set(self.graph.neighbors(paper_id))

        # Find other papers sharing these entities
        paper_scores = defaultdict(float)

        for entity in neighbors:
            # Get papers connected to this entity
            entity_neighbors = self.graph.predecessors(entity)
            for other_paper in entity_neighbors:
                if other_paper != paper_id and self.nodes[other_paper].node_type == NodeType.PAPER:
                    # Weight by entity type
                    entity_type = self.nodes[entity].node_type
                    weight = {
                        NodeType.MODEL: 3.0,
                        NodeType.DATASET: 2.5,
                        NodeType.TASK: 2.0,
                        NodeType.METRIC: 1.5,
                        NodeType.METHOD: 1.5,
                        NodeType.CONCEPT: 1.0,
                    }.get(entity_type, 1.0)

                    paper_scores[other_paper] += weight

        # Normalize by number of shared entities
        for paper in paper_scores:
            shared_count = len(set(self.graph.neighbors(paper)) & neighbors)
            paper_scores[paper] = paper_scores[paper] / max(shared_count, 1)

        # Sort and return top k
        sorted_papers = sorted(paper_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_papers[:top_k]

    def get_graph_statistics(self) -> Dict[str, any]:
        """Get statistics about the knowledge graph."""
        node_type_counts = defaultdict(int)
        for node in self.nodes.values():
            node_type_counts[node.node_type] += 1

        relation_type_counts = defaultdict(int)
        for edge in self.edges:
            relation_type_counts[edge.relation] += 1

        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_types": dict(node_type_counts),
            "relation_types": dict(relation_type_counts),
            "average_degree": sum(dict(self.graph.degree()).values()) / max(len(self.nodes), 1),
            "density": nx.density(self.graph),
            "is_connected": nx.is_weakly_connected(self.graph),
        }

    def export_to_dict(self) -> Dict[str, any]:
        """Export graph to dictionary format."""
        nodes_list = [
            {
                "id": node.node_id,
                "type": node.node_type,
                "name": node.name,
                "properties": node.properties
            }
            for node in self.nodes.values()
        ]

        edges_list = [
            {
                "source": edge.source,
                "target": edge.target,
                "relation": edge.relation,
                "confidence": edge.confidence,
                "properties": edge.properties
            }
            for edge in self.edges
        ]

        return {
            "nodes": nodes_list,
            "edges": edges_list,
            "statistics": self.get_graph_statistics()
        }

    def save_to_file(self, filepath: str) -> None:
        """Save graph to GraphML file."""
        nx.write_graphml(self.graph, filepath)
        logger.info(f"Graph saved to {filepath}")

    def load_from_file(self, filepath: str) -> None:
        """Load graph from GraphML file."""
        self.graph = nx.read_graphml(filepath)
        # Rebuild nodes and edges from graph
        self.nodes = {}
        self.edges = []

        for node_id, data in self.graph.nodes(data=True):
            node = GraphNode(
                node_id=node_id,
                node_type=NodeType(data.get('type', 'concept')),
                name=data.get('name', node_id),
                properties={k: v for k, v in data.items() if k not in ['type', 'name']}
            )
            self.nodes[node_id] = node

        for source, target, data in self.graph.edges(data=True):
            edge = GraphEdge(
                source=source,
                target=target,
                relation=RelationType(data.get('relation', 'related_to')),
                confidence=float(data.get('confidence', 1.0)),
                properties={k: v for k, v in data.items() if k not in ['relation', 'confidence']}
            )
            self.edges.append(edge)

        logger.info(f"Graph loaded from {filepath}")
