"""Embedding generation utilities."""

import os
import logging
from typing import List, Optional
import yaml
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize embedding service.

        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model_name = self.config["embedding"]["model_name"]
        self.device = self.config["embedding"].get("device", "cpu")
        self.chunk_size = self.config["embedding"].get("chunk_size", 512)
        self.chunk_overlap = self.config["embedding"].get("chunk_overlap", 50)

        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Embedding model loaded successfully")
        return self._model

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector as numpy array
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input texts
            batch_size: Batch size for encoding

        Returns:
            Array of embeddings
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100
        )
        return embeddings

    def chunk_and_embed(self, text: str) -> tuple[List[str], np.ndarray]:
        """Split text into chunks and generate embeddings.

        Args:
            text: Input text

        Returns:
            Tuple of (chunks, embeddings)
        """
        chunks = self._chunk_text(text)
        embeddings = self.embed_texts(chunks)
        return chunks, embeddings

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks.

        Args:
            text: Input text

        Returns:
            List of text chunks
        """
        # Simple word-based chunking
        words = text.split()
        chunks = []

        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    def get_embedding_dim(self) -> int:
        """Get the dimension of embeddings.

        Returns:
            Embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        emb1 = self.embed_text(text1)
        emb2 = self.embed_text(text2)

        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        return float(similarity)

    def find_most_similar(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5
    ) -> List[tuple[int, float]]:
        """Find most similar texts to a query.

        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of results to return

        Returns:
            List of (index, similarity_score) tuples
        """
        query_emb = self.embed_text(query)
        candidate_embs = self.embed_texts(candidates)

        # Compute similarities
        similarities = np.dot(candidate_embs, query_emb) / (
            np.linalg.norm(candidate_embs, axis=1) * np.linalg.norm(query_emb)
        )

        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]

        return results
