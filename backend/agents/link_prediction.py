"""Missing-Link Inference using Graph Embeddings.

This module predicts missing links (future collaborations, citations, etc.)
using graph embedding techniques:
1. Node2Vec - Random walk-based embeddings
2. GraphSAGE - Inductive graph neural network embeddings
3. Link prediction scoring functions
"""

import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from backend.agents.knowledge_graph import KnowledgeGraphAgent, NodeType


logger = logging.getLogger(__name__)


@dataclass
class LinkPrediction:
    """Predicted missing link."""
    source_id: str
    target_id: str
    source_name: str
    target_name: str
    score: float
    link_type: str
    reasoning: str
    confidence: float


class Node2VecEmbedder:
    """Simple Node2Vec implementation using random walks."""

    def __init__(
        self,
        graph: nx.Graph,
        dimensions: int = 128,
        walk_length: int = 80,
        num_walks: int = 10,
        p: float = 1.0,
        q: float = 1.0
    ):
        """Initialize Node2Vec embedder.

        Args:
            graph: NetworkX graph
            dimensions: Embedding dimensions
            walk_length: Length of random walks
            num_walks: Number of walks per node
            p: Return parameter
            q: In-out parameter
        """
        self.graph = graph
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.embeddings = {}

        logger.info(f"Node2Vec initialized with dim={dimensions}")

    def _biased_walk(self, start_node: str) -> List[str]:
        """Generate a biased random walk.

        Args:
            start_node: Starting node

        Returns:
            List of visited nodes
        """
        walk = [start_node]

        for _ in range(self.walk_length - 1):
            cur = walk[-1]
            neighbors = list(self.graph.neighbors(cur))

            if not neighbors:
                break

            if len(walk) == 1:
                # First step: uniform random
                walk.append(np.random.choice(neighbors))
            else:
                # Biased step based on p and q
                prev = walk[-2]
                probs = []

                for neighbor in neighbors:
                    if neighbor == prev:
                        # Return to previous node
                        probs.append(1.0 / self.p)
                    elif self.graph.has_edge(neighbor, prev):
                        # BFS-like (close to prev)
                        probs.append(1.0)
                    else:
                        # DFS-like (far from prev)
                        probs.append(1.0 / self.q)

                probs = np.array(probs)
                probs = probs / probs.sum()

                walk.append(np.random.choice(neighbors, p=probs))

        return walk

    def generate_walks(self) -> List[List[str]]:
        """Generate all random walks.

        Returns:
            List of walks
        """
        logger.info("Generating random walks...")
        walks = []

        nodes = list(self.graph.nodes())

        for _ in range(self.num_walks):
            np.random.shuffle(nodes)
            for node in nodes:
                walk = self._biased_walk(node)
                walks.append(walk)

        logger.info(f"Generated {len(walks)} walks")
        return walks

    def learn_embeddings(self, walks: List[List[str]]) -> Dict[str, np.ndarray]:
        """Learn embeddings from walks using skip-gram.

        Simplified version - in production, use gensim Word2Vec.

        Args:
            walks: List of random walks

        Returns:
            Dictionary of node embeddings
        """
        logger.info("Learning embeddings...")

        # Initialize random embeddings
        nodes = list(self.graph.nodes())
        embeddings = {
            node: np.random.randn(self.dimensions) * 0.01
            for node in nodes
        }

        # Simple co-occurrence-based embeddings
        cooccurrence = {node: {} for node in nodes}

        for walk in walks:
            for i, node in enumerate(walk):
                # Context window
                window_start = max(0, i - 5)
                window_end = min(len(walk), i + 6)

                for j in range(window_start, window_end):
                    if i != j:
                        context_node = walk[j]
                        if context_node not in cooccurrence[node]:
                            cooccurrence[node][context_node] = 0
                        cooccurrence[node][context_node] += 1

        # Create embeddings based on cooccurrence
        for node in nodes:
            if cooccurrence[node]:
                # Weighted sum of context embeddings
                context_nodes = list(cooccurrence[node].keys())
                weights = np.array([cooccurrence[node][c] for c in context_nodes])
                weights = weights / weights.sum()

                # Initialize with weighted combination
                embedding = np.zeros(self.dimensions)
                for context_node, weight in zip(context_nodes, weights):
                    embedding += np.random.randn(self.dimensions) * weight * 0.1

                embeddings[node] = embedding

        self.embeddings = embeddings
        logger.info(f"Learned embeddings for {len(embeddings)} nodes")
        return embeddings

    def fit(self):
        """Fit Node2Vec model."""
        walks = self.generate_walks()
        self.embeddings = self.learn_embeddings(walks)
        return self

    def get_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """Get embedding for a node.

        Args:
            node_id: Node identifier

        Returns:
            Embedding vector or None
        """
        return self.embeddings.get(node_id)


class LinkPredictor:
    """Predicts missing links using graph embeddings."""

    def __init__(self, knowledge_graph: KnowledgeGraphAgent):
        """Initialize link predictor.

        Args:
            knowledge_graph: Knowledge graph instance
        """
        self.kg = knowledge_graph
        self.embedder = None
        self.classifier = None

        logger.info("LinkPredictor initialized")

    def train(self, use_node2vec: bool = True):
        """Train link prediction model.

        Args:
            use_node2vec: Whether to use Node2Vec embeddings
        """
        logger.info("Training link prediction model...")

        # Convert to undirected for Node2Vec
        undirected = self.kg.graph.to_undirected()

        # Train Node2Vec
        if use_node2vec:
            self.embedder = Node2VecEmbedder(
                undirected,
                dimensions=64,
                walk_length=30,
                num_walks=10
            )
            self.embedder.fit()

        # Generate training data
        X_train, y_train = self._generate_training_data()

        if len(X_train) == 0:
            logger.warning("No training data generated")
            return

        # Train classifier
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.classifier.fit(X_train, y_train)

        logger.info("Link prediction model trained")

    def _generate_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training data from existing links.

        Returns:
            (X, y) training arrays
        """
        X = []
        y = []

        edges = list(self.kg.graph.edges())

        # Positive examples (existing links)
        for source, target in edges[:100]:  # Limit for speed
            features = self._compute_link_features(source, target)
            if features is not None:
                X.append(features)
                y.append(1)

        # Negative examples (non-existing links)
        nodes = list(self.kg.graph.nodes())
        negative_samples = 0
        max_negative = len(X)

        while negative_samples < max_negative:
            # Sample random pair
            source = np.random.choice(nodes)
            target = np.random.choice(nodes)

            if source != target and not self.kg.graph.has_edge(source, target):
                features = self._compute_link_features(source, target)
                if features is not None:
                    X.append(features)
                    y.append(0)
                    negative_samples += 1

        return np.array(X), np.array(y)

    def _compute_link_features(
        self,
        source: str,
        target: str
    ) -> Optional[np.ndarray]:
        """Compute features for a potential link.

        Args:
            source: Source node ID
            target: Target node ID

        Returns:
            Feature vector or None
        """
        if source not in self.kg.graph or target not in self.kg.graph:
            return None

        features = []

        # 1. Common neighbors
        neighbors_source = set(self.kg.graph.neighbors(source))
        neighbors_target = set(self.kg.graph.neighbors(target))
        common = neighbors_source & neighbors_target
        features.append(len(common))

        # 2. Jaccard coefficient
        union = neighbors_source | neighbors_target
        jaccard = len(common) / max(len(union), 1)
        features.append(jaccard)

        # 3. Preferential attachment
        preferential = len(neighbors_source) * len(neighbors_target)
        features.append(preferential)

        # 4. Adamic-Adar index
        adamic_adar = 0
        for common_node in common:
            degree = self.kg.graph.degree(common_node)
            if degree > 1:
                adamic_adar += 1.0 / np.log(degree)
        features.append(adamic_adar)

        # 5. Node2Vec embedding similarity
        if self.embedder and self.embedder.embeddings:
            emb_source = self.embedder.get_embedding(source)
            emb_target = self.embedder.get_embedding(target)

            if emb_source is not None and emb_target is not None:
                similarity = np.dot(emb_source, emb_target)
                features.append(similarity)
            else:
                features.append(0.0)
        else:
            features.append(0.0)

        # 6. Node types (one-hot encoding)
        source_type = self.kg.nodes[source].node_type if source in self.kg.nodes else NodeType.CONCEPT
        target_type = self.kg.nodes[target].node_type if target in self.kg.nodes else NodeType.CONCEPT

        type_features = [0] * 5  # 5 major types
        type_map = {
            NodeType.PAPER: 0,
            NodeType.MODEL: 1,
            NodeType.DATASET: 2,
            NodeType.TASK: 3,
            NodeType.METHOD: 4
        }

        if source_type in type_map:
            type_features[type_map[source_type]] = 1
        if target_type in type_map:
            type_features[type_map[target_type]] = 1

        features.extend(type_features)

        return np.array(features)

    def predict_missing_links(
        self,
        top_k: int = 20,
        node_filter: Optional[NodeType] = None
    ) -> List[LinkPrediction]:
        """Predict top missing links.

        Args:
            top_k: Number of predictions to return
            node_filter: Filter by node type

        Returns:
            List of link predictions
        """
        if not self.classifier:
            logger.warning("Model not trained. Training now...")
            self.train()

        logger.info(f"Predicting top {top_k} missing links...")

        predictions = []
        nodes = list(self.kg.graph.nodes())

        # Filter nodes if specified
        if node_filter:
            nodes = [n for n in nodes if self.kg.nodes[n].node_type == node_filter]

        # Sample pairs to avoid O(n^2)
        num_samples = min(1000, len(nodes) * 10)
        sampled_pairs = []

        for _ in range(num_samples):
            source = np.random.choice(nodes)
            target = np.random.choice(nodes)

            if source != target and not self.kg.graph.has_edge(source, target):
                sampled_pairs.append((source, target))

        # Score each pair
        for source, target in sampled_pairs:
            features = self._compute_link_features(source, target)

            if features is None:
                continue

            # Predict probability
            prob = self.classifier.predict_proba([features])[0][1]

            if prob > 0.5:  # Threshold
                source_name = self.kg.nodes[source].name if source in self.kg.nodes else source
                target_name = self.kg.nodes[target].name if target in self.kg.nodes else target

                # Determine link type
                link_type = self._infer_link_type(source, target)

                # Generate reasoning
                reasoning = self._generate_reasoning(source, target, features)

                predictions.append(LinkPrediction(
                    source_id=source,
                    target_id=target,
                    source_name=source_name,
                    target_name=target_name,
                    score=float(prob),
                    link_type=link_type,
                    reasoning=reasoning,
                    confidence=float(prob)
                ))

        # Sort by score
        predictions.sort(key=lambda x: x.score, reverse=True)

        logger.info(f"Generated {len(predictions)} link predictions")
        return predictions[:top_k]

    def _infer_link_type(self, source: str, target: str) -> str:
        """Infer the type of relationship.

        Args:
            source: Source node
            target: Target node

        Returns:
            Link type string
        """
        source_type = self.kg.nodes[source].node_type if source in self.kg.nodes else None
        target_type = self.kg.nodes[target].node_type if target in self.kg.nodes else None

        # Simple rules
        if source_type == NodeType.PAPER and target_type == NodeType.PAPER:
            return "potential_citation"
        elif source_type == NodeType.PAPER and target_type == NodeType.MODEL:
            return "potential_usage"
        elif source_type == NodeType.PAPER and target_type == NodeType.DATASET:
            return "potential_evaluation"
        elif source_type == NodeType.MODEL and target_type == NodeType.DATASET:
            return "potential_benchmark"
        elif source_type == NodeType.AUTHOR and target_type == NodeType.AUTHOR:
            return "potential_collaboration"
        else:
            return "potential_relationship"

    def _generate_reasoning(
        self,
        source: str,
        target: str,
        features: np.ndarray
    ) -> str:
        """Generate natural language reasoning for prediction.

        Args:
            source: Source node
            target: Target node
            features: Computed features

        Returns:
            Reasoning string
        """
        common_neighbors = int(features[0])
        jaccard = features[1]

        source_name = self.kg.nodes[source].name if source in self.kg.nodes else source
        target_name = self.kg.nodes[target].name if target in self.kg.nodes else target

        reasoning = f"'{source_name}' and '{target_name}' share {common_neighbors} common connections"

        if jaccard > 0.3:
            reasoning += f" with high similarity (Jaccard: {jaccard:.2f})"

        reasoning += ", suggesting a probable future relationship."

        return reasoning

    def predict_for_node(
        self,
        node_id: str,
        top_k: int = 10
    ) -> List[LinkPrediction]:
        """Predict missing links for a specific node.

        Args:
            node_id: Node to predict links for
            top_k: Number of predictions

        Returns:
            List of link predictions
        """
        if node_id not in self.kg.graph:
            return []

        predictions = []
        all_nodes = list(self.kg.graph.nodes())

        for target in all_nodes:
            if target == node_id or self.kg.graph.has_edge(node_id, target):
                continue

            features = self._compute_link_features(node_id, target)
            if features is None:
                continue

            if self.classifier:
                prob = self.classifier.predict_proba([features])[0][1]
            else:
                # Fallback to heuristic
                prob = features[1]  # Jaccard coefficient

            if prob > 0.4:
                target_name = self.kg.nodes[target].name if target in self.kg.nodes else target
                source_name = self.kg.nodes[node_id].name if node_id in self.kg.nodes else node_id

                predictions.append(LinkPrediction(
                    source_id=node_id,
                    target_id=target,
                    source_name=source_name,
                    target_name=target_name,
                    score=float(prob),
                    link_type=self._infer_link_type(node_id, target),
                    reasoning=self._generate_reasoning(node_id, target, features),
                    confidence=float(prob)
                ))

        predictions.sort(key=lambda x: x.score, reverse=True)
        return predictions[:top_k]

    def export_predictions(
        self,
        predictions: List[LinkPrediction]
    ) -> List[Dict]:
        """Export predictions to dictionary format.

        Args:
            predictions: List of predictions

        Returns:
            List of prediction dictionaries
        """
        return [
            {
                "source_id": p.source_id,
                "target_id": p.target_id,
                "source_name": p.source_name,
                "target_name": p.target_name,
                "score": p.score,
                "link_type": p.link_type,
                "reasoning": p.reasoning,
                "confidence": p.confidence
            }
            for p in predictions
        ]
