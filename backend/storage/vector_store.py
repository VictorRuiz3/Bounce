import numpy as np
from typing import List, Tuple, Dict, Union, Optional
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, dimension: Optional[int] = None):
        """Initialize vector store with optional dimension."""
        self.dimension = dimension
        self.embeddings: Optional[np.ndarray] = None
        self.chunks: List[str] = []
        self.sources: List[str] = []

    def _validate_and_convert_embeddings(self, embeddings: Union[List, np.ndarray]) -> np.ndarray:
        """Validate and convert embeddings to proper numpy array format."""
        try:
            # Convert to numpy array if not already
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)

            # Ensure 2D array
            if len(embeddings.shape) == 1:
                embeddings = embeddings.reshape(1, -1)
            elif len(embeddings.shape) > 2:
                raise ValueError(f"Embeddings must be 1D or 2D array, got shape {embeddings.shape}")

            return embeddings
        except Exception as e:
            logger.error(f"Error converting embeddings: {str(e)}")
            raise ValueError(f"Failed to process embeddings: {str(e)}")

    def add_documents(self, chunks: List[str], embeddings: Union[List, np.ndarray], source: str):
        """Add document chunks and their embeddings to the store."""
        try:
            if not chunks:
                raise ValueError("No chunks provided")

            # Convert and validate embeddings
            embeddings_array = self._validate_and_convert_embeddings(embeddings)

            if len(chunks) != embeddings_array.shape[0]:
                raise ValueError(
                    f"Number of chunks ({len(chunks)}) does not match "
                    f"number of embeddings ({embeddings_array.shape[0]})"
                )

            # Set or validate dimension
            if self.dimension is None:
                self.dimension = embeddings_array.shape[1]
                logger.info(f"Set vector store dimension to {self.dimension}")
            elif embeddings_array.shape[1] != self.dimension:
                logger.warning(
                    f"Embedding dimension mismatch. Expected {self.dimension}, "
                    f"got {embeddings_array.shape[1]}. Adjusting..."
                )
                if embeddings_array.shape[1] < self.dimension:
                    # Pad with zeros
                    padding = np.zeros((embeddings_array.shape[0], self.dimension - embeddings_array.shape[1]))
                    embeddings_array = np.hstack([embeddings_array, padding])
                else:
                    # Truncate to match dimension
                    embeddings_array = embeddings_array[:, :self.dimension]

            # Add embeddings
            if self.embeddings is None:
                self.embeddings = embeddings_array
            else:
                self.embeddings = np.vstack([self.embeddings, embeddings_array])

            # Add chunks and sources
            self.chunks.extend(chunks)
            self.sources.extend([source] * len(chunks))

            logger.info(
                f"Successfully added {len(chunks)} chunks from {source}. "
                f"Total chunks: {len(self.chunks)}"
            )

        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise Exception(f"Failed to add documents to vector store: {str(e)}")

    def search(self, query_embedding: Union[List, np.ndarray], k: int = 3) -> List[Dict]:
        """Search for most similar chunks to the query."""
        try:
            if self.embeddings is None or len(self.chunks) == 0:
                return []

            # Convert and validate query embedding
            query_array = self._validate_and_convert_embeddings(query_embedding)

            # Handle dimension mismatch
            if query_array.shape[1] != self.dimension:
                if query_array.shape[1] < self.dimension:
                    # Pad with zeros
                    padding = np.zeros((1, self.dimension - query_array.shape[1]))
                    query_array = np.hstack([query_array, padding])
                else:
                    # Truncate to match dimension
                    query_array = query_array[:, :self.dimension]

            # Calculate cosine similarity
            similarities = cosine_similarity(query_array, self.embeddings)[0]

            # Get top k most similar chunks
            top_indices = np.argsort(similarities)[-k:][::-1]

            # Combine results with sources
            results = []
            for idx in top_indices:
                if idx < len(self.chunks):
                    results.append({
                        "text": self.chunks[idx],
                        "score": float(similarities[idx]),
                        "source": self.sources[idx]
                    })

            return results

        except Exception as e:
            logger.error(f"Error performing vector search: {str(e)}")
            raise Exception(f"Failed to perform vector search: {str(e)}")

    def clear(self):
        """Clear the vector store."""
        self.embeddings = None
        self.chunks = []
        self.sources = []
        logger.info("Vector store cleared")