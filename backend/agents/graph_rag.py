"""GraphRAG - Graph Retrieval-Augmented Generation.

This module combines vector-based semantic search with graph traversal
to provide more comprehensive and contextual retrieval for RAG systems.

Hybrid retrieval combines:
1. Vector similarity search (dense retrieval)
2. Graph neighborhood expansion (structural context)
3. Path-based reasoning (relational context)
"""

import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import numpy as np
import networkx as nx

from backend.agents.knowledge_graph import KnowledgeGraphAgent, NodeType
from backend.utils.embeddings import EmbeddingService
from backend.utils.storage import VectorStore


logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from hybrid retrieval."""
    content: str
    source_id: str
    source_type: str
    score: float
    retrieval_method: str  # "vector", "graph", "hybrid"
    context: Optional[Dict] = None


class GraphRAG:
    """Graph Retrieval-Augmented Generation engine."""

    def __init__(
        self,
        knowledge_graph: KnowledgeGraphAgent,
        vector_store: VectorStore,
        embedding_service: EmbeddingService
    ):
        """Initialize GraphRAG.

        Args:
            knowledge_graph: Knowledge graph for structural retrieval
            vector_store: Vector store for semantic retrieval
            embedding_service: Embedding service for query encoding
        """
        self.kg = knowledge_graph
        self.vector_store = vector_store
        self.embedding_service = embedding_service

        logger.info("GraphRAG initialized")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        method: str = "hybrid",
        expansion_depth: int = 1,
        rerank: bool = True
    ) -> List[RetrievalResult]:
        """Retrieve relevant documents using hybrid approach.

        Args:
            query: Search query
            top_k: Number of results to return
            method: Retrieval method ("vector", "graph", "hybrid")
            expansion_depth: Graph expansion depth
            rerank: Whether to rerank results

        Returns:
            List of retrieval results
        """
        logger.info(f"Retrieving with method={method}, top_k={top_k}")

        if method == "vector":
            results = self._vector_retrieval(query, top_k)
        elif method == "graph":
            results = self._graph_retrieval(query, top_k, expansion_depth)
        elif method == "hybrid":
            results = self._hybrid_retrieval(query, top_k, expansion_depth)
        else:
            raise ValueError(f"Unknown method: {method}")

        if rerank:
            results = self._rerank_results(query, results)

        return results[:top_k]

    def _vector_retrieval(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Pure vector similarity search.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of retrieval results
        """
        # Search vector store
        search_results = self.vector_store.search(query, top_k=top_k * 2)

        results = []
        for result in search_results:
            results.append(RetrievalResult(
                content=result["content"],
                source_id=result.get("paper_id", "unknown"),
                source_type="paper_chunk",
                score=result.get("score", 0.0),
                retrieval_method="vector",
                context={"chunk_id": result.get("chunk_id")}
            ))

        return results

    def _graph_retrieval(
        self,
        query: str,
        top_k: int,
        expansion_depth: int
    ) -> List[RetrievalResult]:
        """Graph-based retrieval using structural relationships.

        Args:
            query: Search query
            top_k: Number of results
            expansion_depth: How many hops to expand

        Returns:
            List of retrieval results
        """
        # First, find relevant entities using vector search
        query_embedding = self.embedding_service.embed_texts([query])[0]

        # Find nodes in graph that match query
        relevant_nodes = self._find_relevant_graph_nodes(query, top_k=5)

        results = []
        visited = set()

        for node_id, initial_score in relevant_nodes:
            if node_id in visited:
                continue
            visited.add(node_id)

            # Get node information
            node = self.kg.nodes.get(node_id)
            if not node:
                continue

            # Add the node itself
            results.append(RetrievalResult(
                content=self._format_node_content(node),
                source_id=node_id,
                source_type=str(node.node_type),
                score=initial_score,
                retrieval_method="graph",
                context={"node_type": str(node.node_type)}
            ))

            # Expand to neighbors
            if expansion_depth > 0:
                neighbors = self._get_neighbors_with_context(node_id, expansion_depth)

                for neighbor_id, relation, depth in neighbors:
                    if neighbor_id in visited:
                        continue
                    visited.add(neighbor_id)

                    neighbor_node = self.kg.nodes.get(neighbor_id)
                    if not neighbor_node:
                        continue

                    # Score decreases with depth
                    score = initial_score * (0.7 ** depth)

                    results.append(RetrievalResult(
                        content=self._format_node_content(neighbor_node),
                        source_id=neighbor_id,
                        source_type=str(neighbor_node.node_type),
                        score=score,
                        retrieval_method="graph",
                        context={
                            "node_type": str(neighbor_node.node_type),
                            "relation": str(relation),
                            "depth": depth,
                            "from_node": node_id
                        }
                    ))

        return results

    def _hybrid_retrieval(
        self,
        query: str,
        top_k: int,
        expansion_depth: int
    ) -> List[RetrievalResult]:
        """Hybrid retrieval combining vector and graph methods.

        Args:
            query: Search query
            top_k: Number of results
            expansion_depth: Graph expansion depth

        Returns:
            List of retrieval results
        """
        # Get vector results
        vector_results = self._vector_retrieval(query, top_k=top_k)

        # Get graph results
        graph_results = self._graph_retrieval(query, top_k=top_k, expansion_depth=expansion_depth)

        # Merge and deduplicate
        combined = {}

        for result in vector_results:
            combined[result.source_id] = result
            result.retrieval_method = "hybrid"
            result.score *= 0.6  # Weight vector results

        for result in graph_results:
            if result.source_id in combined:
                # Already exists from vector search - boost score
                combined[result.source_id].score += result.score * 0.4
                combined[result.source_id].context = {
                    **combined[result.source_id].context,
                    "graph_context": result.context
                }
            else:
                # New from graph search
                result.score *= 0.4  # Weight graph results
                result.retrieval_method = "hybrid"
                combined[result.source_id] = result

        # Convert back to list and sort
        results = list(combined.values())
        results.sort(key=lambda x: x.score, reverse=True)

        return results

    def _find_relevant_graph_nodes(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Find graph nodes relevant to query using embeddings.

        Args:
            query: Search query
            top_k: Number of nodes to return

        Returns:
            List of (node_id, score) tuples
        """
        # Embed query
        query_embedding = self.embedding_service.embed_texts([query])[0]

        # Embed all node names
        node_ids = []
        node_texts = []

        for node_id, node in self.kg.nodes.items():
            # Include name and properties in text
            text = node.name
            if node.properties.get('abstract'):
                text += " " + node.properties['abstract'][:200]

            node_ids.append(node_id)
            node_texts.append(text)

        # Compute similarities
        node_embeddings = self.embedding_service.embed_texts(node_texts)

        similarities = []
        for i, node_emb in enumerate(node_embeddings):
            similarity = np.dot(query_embedding, node_emb)
            similarities.append((node_ids[i], float(similarity)))

        # Sort and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def _get_neighbors_with_context(
        self,
        node_id: str,
        max_depth: int
    ) -> List[Tuple[str, str, int]]:
        """Get neighbors with relationship context.

        Args:
            node_id: Source node
            max_depth: Maximum traversal depth

        Returns:
            List of (neighbor_id, relation_type, depth) tuples
        """
        neighbors = []

        # BFS traversal
        visited = {node_id}
        queue = [(node_id, 0)]

        while queue:
            current, depth = queue.pop(0)

            if depth >= max_depth:
                continue

            # Get outgoing edges
            for neighbor in self.kg.graph.neighbors(current):
                if neighbor in visited:
                    continue

                visited.add(neighbor)

                # Get edge relation
                edge_data = self.kg.graph.get_edge_data(current, neighbor)
                relation = edge_data.get('relation', 'related_to')

                neighbors.append((neighbor, relation, depth + 1))

                if depth + 1 < max_depth:
                    queue.append((neighbor, depth + 1))

        return neighbors

    def _format_node_content(self, node) -> str:
        """Format node as readable content.

        Args:
            node: Graph node

        Returns:
            Formatted content string
        """
        content = f"{node.node_type}: {node.name}"

        # Add relevant properties
        if node.properties.get('abstract'):
            content += f"\nAbstract: {node.properties['abstract'][:300]}..."

        if node.properties.get('year'):
            content += f"\nYear: {node.properties['year']}"

        if node.properties.get('authors'):
            authors = node.properties['authors']
            if isinstance(authors, list):
                content += f"\nAuthors: {', '.join(authors[:3])}"

        return content

    def _rerank_results(
        self,
        query: str,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Rerank results using cross-encoder or LLM.

        For now, uses a simple diversity-aware reranking.

        Args:
            query: Original query
            results: Initial results

        Returns:
            Reranked results
        """
        # Simple diversity reranking - ensure different source types
        reranked = []
        source_type_counts = {}

        # First pass: high-scoring results
        for result in sorted(results, key=lambda x: x.score, reverse=True):
            source_type = result.source_type
            count = source_type_counts.get(source_type, 0)

            # Allow more papers, fewer of other types
            max_per_type = 3 if source_type == "paper" else 2

            if count < max_per_type:
                reranked.append(result)
                source_type_counts[source_type] = count + 1

        # Second pass: fill remaining slots
        for result in results:
            if result not in reranked:
                reranked.append(result)

        return reranked

    def retrieve_with_reasoning(
        self,
        query: str,
        top_k: int = 5,
        reasoning_paths: bool = True
    ) -> Dict:
        """Retrieve with reasoning paths explaining the results.

        Args:
            query: Search query
            top_k: Number of results
            reasoning_paths: Whether to include reasoning paths

        Returns:
            Dictionary with results and reasoning
        """
        results = self.retrieve(query, top_k=top_k, method="hybrid")

        output = {
            "query": query,
            "results": [self._result_to_dict(r) for r in results],
            "reasoning": []
        }

        if reasoning_paths:
            # For each result, find path from query to result
            for result in results[:3]:  # Top 3 only
                if result.source_id in self.kg.nodes:
                    path_explanation = self._explain_retrieval_path(query, result)
                    output["reasoning"].append(path_explanation)

        return output

    def _result_to_dict(self, result: RetrievalResult) -> Dict:
        """Convert result to dictionary.

        Args:
            result: Retrieval result

        Returns:
            Dictionary representation
        """
        return {
            "content": result.content,
            "source_id": result.source_id,
            "source_type": result.source_type,
            "score": result.score,
            "retrieval_method": result.retrieval_method,
            "context": result.context
        }

    def _explain_retrieval_path(
        self,
        query: str,
        result: RetrievalResult
    ) -> Dict:
        """Explain why a result was retrieved.

        Args:
            query: Original query
            result: Retrieved result

        Returns:
            Explanation dictionary
        """
        explanation = {
            "result_id": result.source_id,
            "result_name": self.kg.nodes[result.source_id].name if result.source_id in self.kg.nodes else result.source_id,
            "score": result.score,
            "reasons": []
        }

        # Vector similarity reason
        if result.retrieval_method in ["vector", "hybrid"]:
            explanation["reasons"].append({
                "type": "semantic_similarity",
                "description": f"Content is semantically similar to query '{query}'"
            })

        # Graph context reason
        if result.context and result.context.get("graph_context"):
            graph_ctx = result.context["graph_context"]
            if graph_ctx.get("from_node"):
                from_node_name = self.kg.nodes[graph_ctx["from_node"]].name
                relation = graph_ctx.get("relation", "related to")
                explanation["reasons"].append({
                    "type": "graph_relationship",
                    "description": f"{relation} '{from_node_name}' which matched the query"
                })

        # Node type reason
        if result.source_type in ["model", "dataset", "method"]:
            explanation["reasons"].append({
                "type": "entity_type",
                "description": f"Relevant {result.source_type} entity"
            })

        return explanation

    def multi_hop_query(
        self,
        start_entity: str,
        end_entity: str,
        max_hops: int = 3
    ) -> List[List[str]]:
        """Find reasoning paths between two entities.

        Args:
            start_entity: Starting entity ID
            end_entity: Target entity ID
            max_hops: Maximum path length

        Returns:
            List of paths (each path is list of node IDs)
        """
        if start_entity not in self.kg.graph or end_entity not in self.kg.graph:
            return []

        try:
            # Find all simple paths up to max_hops
            paths = list(nx.all_simple_paths(
                self.kg.graph,
                start_entity,
                end_entity,
                cutoff=max_hops
            ))

            # Limit to top 5 shortest paths
            paths.sort(key=len)
            return paths[:5]

        except nx.NetworkXNoPath:
            return []

    def explain_path(self, path: List[str]) -> str:
        """Explain a reasoning path in natural language.

        Args:
            path: List of node IDs forming a path

        Returns:
            Natural language explanation
        """
        if not path:
            return "No path found."

        explanation_parts = []

        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]

            source_name = self.kg.nodes[source].name if source in self.kg.nodes else source
            target_name = self.kg.nodes[target].name if target in self.kg.nodes else target

            # Get edge relation
            edge_data = self.kg.graph.get_edge_data(source, target)
            relation = edge_data.get('relation', 'related to') if edge_data else 'related to'

            explanation_parts.append(f"'{source_name}' {relation} '{target_name}'")

        return " → ".join(explanation_parts)
