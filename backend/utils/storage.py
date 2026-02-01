"""Vector storage and retrieval utilities."""

import os
import logging
from typing import List, Dict, Any, Optional
import yaml
import chromadb
from chromadb.config import Settings
import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """Wrapper for vector database operations."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize vector store.

        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.store_type = self.config["vector_store"]["type"]
        self.persist_dir = self.config["vector_store"]["persist_directory"]
        self.collection_name = self.config["vector_store"]["collection_name"]

        # Ensure persist directory exists
        os.makedirs(self.persist_dir, exist_ok=True)

        self._client = None
        self._collection = None

        logger.info(f"Initialized {self.store_type} vector store at {self.persist_dir}")

    @property
    def client(self):
        """Lazy load the vector store client."""
        if self._client is None:
            if self.store_type == "chroma":
                self._client = chromadb.Client(
                    Settings(
                        persist_directory=self.persist_dir,
                        anonymized_telemetry=False
                    )
                )
            else:
                raise ValueError(f"Unsupported vector store type: {self.store_type}")

        return self._client

    @property
    def collection(self):
        """Get or create the collection."""
        if self._collection is None:
            if self.store_type == "chroma":
                self._collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"description": "ScholarGenie paper embeddings"}
                )

        return self._collection

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        embeddings: Optional[np.ndarray] = None
    ) -> List[str]:
        """Add texts to the vector store.

        Args:
            texts: List of text chunks
            metadatas: Optional metadata for each text
            ids: Optional IDs for each text (auto-generated if None)
            embeddings: Optional pre-computed embeddings

        Returns:
            List of IDs for added texts
        """
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in texts]

        if metadatas is None:
            metadatas = [{}] * len(texts)

        # Convert embeddings to list if provided
        if embeddings is not None:
            embeddings = embeddings.tolist()

        if self.store_type == "chroma":
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )

        logger.info(f"Added {len(texts)} texts to vector store")
        return ids

    def add_paper(
        self,
        paper_id: str,
        chunks: List[str],
        embeddings: np.ndarray,
        metadata: Dict[str, Any]
    ) -> List[str]:
        """Add a paper's chunks to the vector store.

        Args:
            paper_id: Paper identifier
            chunks: Text chunks
            embeddings: Chunk embeddings
            metadata: Paper metadata

        Returns:
            List of chunk IDs
        """
        # Create IDs for chunks
        chunk_ids = [f"{paper_id}_chunk_{i}" for i in range(len(chunks))]

        # Create metadata for each chunk
        chunk_metadatas = [
            {
                **metadata,
                "paper_id": paper_id,
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            for i in range(len(chunks))
        ]

        return self.add_texts(
            texts=chunks,
            metadatas=chunk_metadatas,
            ids=chunk_ids,
            embeddings=embeddings
        )

    def search(
        self,
        query_text: Optional[str] = None,
        query_embedding: Optional[np.ndarray] = None,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Search for similar texts.

        Args:
            query_text: Query text (used if query_embedding is None)
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_dict: Metadata filter

        Returns:
            Search results with IDs, documents, distances, and metadata
        """
        if query_embedding is not None:
            query_embedding = query_embedding.tolist()

        if self.store_type == "chroma":
            results = self.collection.query(
                query_texts=[query_text] if query_text else None,
                query_embeddings=[query_embedding] if query_embedding is not None else None,
                n_results=top_k,
                where=filter_dict
            )

            return {
                "ids": results["ids"][0] if results["ids"] else [],
                "documents": results["documents"][0] if results["documents"] else [],
                "distances": results["distances"][0] if results["distances"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else []
            }

        return {"ids": [], "documents": [], "distances": [], "metadatas": []}

    def get_paper_chunks(self, paper_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific paper.

        Args:
            paper_id: Paper identifier

        Returns:
            List of chunks with metadata
        """
        if self.store_type == "chroma":
            results = self.collection.get(
                where={"paper_id": paper_id}
            )

            chunks = []
            for i in range(len(results["ids"])):
                chunks.append({
                    "id": results["ids"][i],
                    "text": results["documents"][i],
                    "metadata": results["metadatas"][i]
                })

            return chunks

        return []

    def delete_paper(self, paper_id: str) -> int:
        """Delete all chunks for a paper.

        Args:
            paper_id: Paper identifier

        Returns:
            Number of chunks deleted
        """
        if self.store_type == "chroma":
            # Get all chunk IDs for this paper
            results = self.collection.get(
                where={"paper_id": paper_id}
            )

            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} chunks for paper {paper_id}")
                return len(results["ids"])

        return 0

    def count(self) -> int:
        """Get total number of documents in the store.

        Returns:
            Document count
        """
        if self.store_type == "chroma":
            return self.collection.count()
        return 0

    def clear(self):
        """Clear all data from the vector store."""
        if self.store_type == "chroma":
            self.client.delete_collection(name=self.collection_name)
            self._collection = None
            logger.warning(f"Cleared vector store collection: {self.collection_name}")

    def get_all_paper_ids(self) -> List[str]:
        """Get list of all unique paper IDs in the store.

        Returns:
            List of paper IDs
        """
        if self.store_type == "chroma":
            all_results = self.collection.get()
            metadatas = all_results["metadatas"]

            paper_ids = set()
            for metadata in metadatas:
                if "paper_id" in metadata:
                    paper_ids.add(metadata["paper_id"])

            return sorted(list(paper_ids))

        return []
