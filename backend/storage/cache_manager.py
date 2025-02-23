import os
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, cache_dir: str = ".cache"):
        """Initialize cache manager with directory path."""
        self.cache_dir = cache_dir
        self.embedding_cache_file = os.path.join(cache_dir, "embeddings_cache.json")
        self.query_cache_file = os.path.join(cache_dir, "query_cache.json")
        self.cache_ttl = timedelta(hours=24)  # Cache TTL of 24 hours
        self._init_cache()

    def _init_cache(self):
        """Initialize cache directory and files."""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            # Initialize embedding cache if not exists
            if not os.path.exists(self.embedding_cache_file):
                self._save_cache({}, self.embedding_cache_file)
            # Initialize query cache if not exists
            if not os.path.exists(self.query_cache_file):
                self._save_cache({}, self.query_cache_file)
        except Exception as e:
            logger.error(f"Failed to initialize cache: {e}")
            raise

    def _load_cache(self, cache_file: str) -> Dict:
        """Load cache from file."""
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Failed to load cache from {cache_file}: {e}")
            return {}

    def _save_cache(self, cache_data: Dict, cache_file: str):
        """Save cache to file."""
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            logger.error(f"Failed to save cache to {cache_file}: {e}")

    def _is_cache_valid(self, timestamp: str) -> bool:
        """Check if cache entry is still valid based on TTL."""
        try:
            cache_time = datetime.fromisoformat(timestamp)
            return datetime.now() - cache_time < self.cache_ttl
        except Exception:
            return False

    def get_embedding_cache(self, text: str) -> Optional[List[float]]:
        """Get cached embedding for text if available."""
        try:
            cache = self._load_cache(self.embedding_cache_file)
            cache_key = hash(text)
            if str(cache_key) in cache:
                entry = cache[str(cache_key)]
                if self._is_cache_valid(entry["timestamp"]):
                    return entry["embedding"]
            return None
        except Exception as e:
            logger.error(f"Error retrieving from embedding cache: {e}")
            return None

    def cache_embedding(self, text: str, embedding: List[float]):
        """Cache embedding for text."""
        try:
            cache = self._load_cache(self.embedding_cache_file)
            cache_key = hash(text)
            cache[str(cache_key)] = {
                "embedding": embedding,
                "timestamp": datetime.now().isoformat()
            }
            self._save_cache(cache, self.embedding_cache_file)
        except Exception as e:
            logger.error(f"Error saving to embedding cache: {e}")

    def get_query_cache(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached query result if available."""
        try:
            cache = self._load_cache(self.query_cache_file)
            cache_key = hash(query)
            if str(cache_key) in cache:
                entry = cache[str(cache_key)]
                if self._is_cache_valid(entry["timestamp"]):
                    return entry["result"]
            return None
        except Exception as e:
            logger.error(f"Error retrieving from query cache: {e}")
            return None

    def cache_query(self, query: str, result: Dict[str, Any]):
        """Cache query result."""
        try:
            cache = self._load_cache(self.query_cache_file)
            cache_key = hash(query)
            cache[str(cache_key)] = {
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            self._save_cache(cache, self.query_cache_file)
        except Exception as e:
            logger.error(f"Error saving to query cache: {e}")

    def clear_cache(self):
        """Clear all cache files."""
        try:
            self._save_cache({}, self.embedding_cache_file)
            self._save_cache({}, self.query_cache_file)
            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

    def clean_expired_entries(self):
        """Remove expired entries from cache files."""
        try:
            # Clean embedding cache
            embedding_cache = self._load_cache(self.embedding_cache_file)
            embedding_cache = {
                k: v for k, v in embedding_cache.items()
                if self._is_cache_valid(v["timestamp"])
            }
            self._save_cache(embedding_cache, self.embedding_cache_file)

            # Clean query cache
            query_cache = self._load_cache(self.query_cache_file)
            query_cache = {
                k: v for k, v in query_cache.items()
                if self._is_cache_valid(v["timestamp"])
            }
            self._save_cache(query_cache, self.query_cache_file)

            logger.info("Cleaned expired cache entries")
        except Exception as e:
            logger.error(f"Error cleaning expired cache entries: {e}")
