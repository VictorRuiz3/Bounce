from typing import Dict, List
import numpy as np
from mistralai.client import MistralClient
from backend.config import MISTRAL_API_KEY, LLM_MODEL, SYSTEM_PROMPT
from backend.storage.cache_manager import CacheManager
import logging

logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self, document_processor, vector_store):
        self.document_processor = document_processor
        self.vector_store = vector_store
        self.cache_manager = CacheManager()

        # Validate API key
        if not MISTRAL_API_KEY:
            raise ValueError("Mistral API key is not set. Please set the MISTRAL_API_KEY environment variable.")

        try:
            self.client = MistralClient(api_key=MISTRAL_API_KEY)
        except Exception as e:
            raise ValueError(f"Failed to initialize Mistral client: {str(e)}")

    def process_query(self, query: str, k: int = 3) -> Dict:
        """Process a query and generate a response using RAG with caching."""
        try:
            # Check cache first
            cached_result = self.cache_manager.get_query_cache(query)
            if cached_result:
                logger.info("Using cached query result")
                return cached_result

            # Generate query embedding using Mistral
            cached_embedding = self.cache_manager.get_embedding_cache(query)
            if cached_embedding:
                logger.info("Using cached query embedding")
                query_embedding = cached_embedding
            else:
                query_embedding = self.document_processor.create_embeddings([query])[0]
                self.cache_manager.cache_embedding(query, query_embedding.tolist())

            # Retrieve relevant chunks
            relevant_chunks = self.vector_store.search(query_embedding, k)

            # Prepare context with source information and token limit
            context_parts = []
            total_tokens = 0
            max_tokens = 60000  # Keep total context well under model's limit

            for chunk in relevant_chunks:
                chunk_text = f"From document '{chunk['source']}':\n{chunk['text']}"
                # Approximate token count (roughly 4 chars per token)
                estimated_tokens = len(chunk_text) // 4

                if total_tokens + estimated_tokens > max_tokens:
                    break

                context_parts.append(chunk_text)
                total_tokens += estimated_tokens

            context = "\n\n".join(context_parts)

            # Generate response
            response = self.generate_response(query, context)

            result = {
                "response": response,
                "context": relevant_chunks
            }

            # Cache the result
            self.cache_manager.cache_query(query, result)

            return result
        except Exception as e:
            logger.error(f"Failed to process query: {str(e)}")
            raise Exception(f"Failed to process query: {e}")

    def generate_response(self, query: str, context: str) -> str:
        """Generate a response using Mistral's model."""
        try:
            response = self.client.chat(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                ]
            )

            return response.choices[0].message.content
        except Exception as e:
            if 'invalid_api_key' in str(e).lower():
                raise Exception("Invalid Mistral API key. Please check your API key and try again.")
            raise Exception(f"Failed to generate response: {e}")

    def clear_cache(self):
        """Clear all cached data."""
        self.cache_manager.clear_cache()

    def clean_expired_cache(self):
        """Remove expired cache entries."""
        self.cache_manager.clean_expired_entries()
