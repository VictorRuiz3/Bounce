from typing import List, Dict, Optional, Callable
import numpy as np
from mistralai.client import MistralClient
from backend.config import (
    MISTRAL_API_KEY, 
    CHUNK_SIZE, 
    CHUNK_OVERLAP, 
    EMBEDDING_MODEL,
    MAX_BATCH_SIZE,
    MAX_RECURSIVE_CHUNKS
)
from backend.storage.cache_manager import CacheManager
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        if not MISTRAL_API_KEY:
            raise ValueError("Mistral API key is not set")
        self.client = MistralClient(api_key=MISTRAL_API_KEY)
        self.processing_cancelled = False
        self.cache_manager = CacheManager()

    def cancel_processing(self):
        """Cancel ongoing document processing."""
        self.processing_cancelled = True

    def reset_cancel_flag(self):
        """Reset the cancellation flag."""
        self.processing_cancelled = False

    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            logger.warning("Empty text provided for preprocessing")
            return ""

        logger.info(f"Preprocessing text of length {len(text)}")
        text = ' '.join(text.split())
        text = ''.join(char for char in text if char.isprintable())
        return text

    def recursive_chunk_text(self, text: str, depth: int = 0) -> List[str]:
        """Recursively split text into smaller chunks if too large."""
        if self.processing_cancelled:
            raise Exception("Processing cancelled by user")

        if not text:
            logger.warning("Empty text provided for chunking")
            return []

        logger.debug(f"Recursive chunking at depth {depth}")
        words = text.split()

        # Return if text is small enough or max depth reached
        if len(words) <= CHUNK_SIZE or depth >= MAX_RECURSIVE_CHUNKS:
            return [text] if text.strip() else []

        sections = text.split('\n\n')
        if len(sections) == 1:
            sections = text.split('. ')

        chunks = []
        current_chunk = []
        current_size = 0

        for section in sections:
            if self.processing_cancelled:
                raise Exception("Processing cancelled by user")

            section_words = len(section.split())

            if current_size + section_words <= CHUNK_SIZE:
                current_chunk.append(section)
                current_size += section_words
            else:
                if current_chunk:
                    chunk_text = '. '.join(current_chunk) if '. ' in text else '\n\n'.join(current_chunk)
                    if len(chunk_text.split()) > CHUNK_SIZE:
                        sub_chunks = self.recursive_chunk_text(chunk_text, depth + 1)
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append(chunk_text)

                current_chunk = [section]
                current_size = section_words

        if current_chunk:
            chunk_text = '. '.join(current_chunk) if '. ' in text else '\n\n'.join(current_chunk)
            if len(chunk_text.split()) > CHUNK_SIZE:
                sub_chunks = self.recursive_chunk_text(chunk_text, depth + 1)
                chunks.extend(sub_chunks)
            else:
                chunks.append(chunk_text)

        return [chunk for chunk in chunks if chunk.strip()]

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks using recursive strategy."""
        try:
            text = self.preprocess_text(text)
            if not text:
                logger.warning("Text preprocessing returned empty result")
                return []

            logger.info(f"Starting text chunking for text of length {len(text)}")
            chunks = self.recursive_chunk_text(text)

            # Post-process chunks to ensure size constraints
            final_chunks = []
            for chunk in chunks:
                if self.processing_cancelled:
                    raise Exception("Processing cancelled by user")

                chunk_words = chunk.split()
                if len(chunk_words) > CHUNK_SIZE:
                    # Split into smaller chunks with overlap
                    for i in range(0, len(chunk_words), CHUNK_SIZE - CHUNK_OVERLAP):
                        sub_chunk = ' '.join(chunk_words[i:i + CHUNK_SIZE])
                        if sub_chunk and len(sub_chunk.split()) <= CHUNK_SIZE:
                            final_chunks.append(sub_chunk)
                else:
                    final_chunks.append(chunk)

            logger.info(f"Generated {len(final_chunks)} chunks")
            return final_chunks

        except Exception as e:
            logger.error(f"Error in chunk_text: {str(e)}")
            raise

    def process_chunk_batch(self, batch: List[str]) -> List[np.ndarray]:
        """Process a batch of chunks to create embeddings with caching."""
        if not batch:
            logger.warning("Empty batch provided for processing")
            return []

        if self.processing_cancelled:
            raise Exception("Processing cancelled by user")

        try:
            # Check cache first
            cached_embeddings = []
            uncached_chunks = []

            for chunk in batch:
                cached_embedding = self.cache_manager.get_embedding_cache(chunk)
                if cached_embedding is not None:
                    cached_embeddings.append(cached_embedding)
                else:
                    uncached_chunks.append(chunk)

            if not uncached_chunks:
                logger.info(f"Using {len(cached_embeddings)} cached embeddings")
                return cached_embeddings

            # Process uncached chunks
            response = self.client.embeddings(
                model=EMBEDDING_MODEL,
                input=uncached_chunks
            )
            new_embeddings = [embed_data.embedding for embed_data in response.data]

            # Cache new embeddings
            for chunk, embedding in zip(uncached_chunks, new_embeddings):
                self.cache_manager.cache_embedding(chunk, embedding)

            # Combine cached and new embeddings
            return cached_embeddings + new_embeddings

        except Exception as e:
            logger.error(f"Error in process_chunk_batch: {str(e)}")
            if "Too many tokens" in str(e):
                # Process individually if batch is too large
                logger.warning("Batch too large, processing chunks individually")
                results = []
                for chunk in batch:
                    if self.processing_cancelled:
                        raise Exception("Processing cancelled by user")

                    try:
                        cached_embedding = self.cache_manager.get_embedding_cache(chunk)
                        if cached_embedding is not None:
                            results.append(cached_embedding)
                            continue

                        response = self.client.embeddings(
                            model=EMBEDDING_MODEL,
                            input=[chunk]
                        )
                        embedding = response.data[0].embedding
                        self.cache_manager.cache_embedding(chunk, embedding)
                        results.append(embedding)
                    except Exception as chunk_error:
                        logger.error(f"Error processing individual chunk: {str(chunk_error)}")
                        continue

                return results
            raise

    def create_embeddings(self, chunks: List[str], progress_callback: Optional[Callable[[float, str], None]] = None) -> np.ndarray:
        """Generate embeddings using parallel processing and progress tracking."""
        if not chunks:
            logger.warning("No chunks provided for embedding creation")
            return np.array([])

        try:
            all_embeddings = []
            total_batches = (len(chunks) + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE
            completed_batches = 0

            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = []

                for i in range(0, len(chunks), MAX_BATCH_SIZE):
                    if self.processing_cancelled:
                        raise Exception("Processing cancelled by user")

                    batch = chunks[i:i + MAX_BATCH_SIZE]
                    total_tokens = sum(len(chunk.split()) for chunk in batch)

                    if total_tokens > 15000:  # Mistral token limit
                        batch = [chunk for chunk in batch if len(chunk.split()) <= 2000]

                    if batch:
                        futures.append(executor.submit(self.process_chunk_batch, batch))

                for future in as_completed(futures):
                    if self.processing_cancelled:
                        raise Exception("Processing cancelled by user")

                    try:
                        batch_embeddings = future.result()
                        all_embeddings.extend(batch_embeddings)

                        completed_batches += 1
                        if progress_callback:
                            progress = completed_batches / total_batches
                            progress_callback(progress)
                    except Exception as e:
                        logger.error(f"Error processing batch: {str(e)}")
                        continue

            return np.array(all_embeddings)

        except Exception as e:
            if "Processing cancelled by user" in str(e):
                raise e
            logger.error(f"Error in create_embeddings: {str(e)}")
            raise Exception(f"Failed to create embeddings: {str(e)}")

    def process_document(self, text: str, progress_callback: Optional[Callable[[float, str], None]] = None) -> Dict:
        """Process document with progress tracking and cancellation support."""
        if not text:
            raise ValueError("Empty document provided")

        self.reset_cancel_flag()
        start_time = time.time()
        total_words = len(text.split())
        logger.info(f"Starting document processing with {total_words} words")

        try:
            if progress_callback:
                progress_callback(0.1, "Chunking document...")

            # Add maximum text length check (approximately 100k tokens)
            max_text_length = 400000  # characters (roughly 100k tokens)
            if len(text) > max_text_length:
                logger.info("Large document detected, processing in sections")
                # Split into major sections first
                sections = text.split('\n\n')
                processed_sections = []
                total_sections = len(sections)

                for i, section in enumerate(sections):
                    if self.processing_cancelled:
                        raise Exception("Document processing was cancelled")

                    if progress_callback:
                        progress = 0.1 + (0.8 * (i / total_sections))
                        progress_callback(progress, f"Processing section {i+1}/{total_sections}...")

                    # Process each section individually
                    section_chunks = self.chunk_text(section)
                    if section_chunks:
                        section_embeddings = self.create_embeddings(
                            section_chunks,
                            lambda p: progress_callback(progress + (0.8/total_sections) * p, 
                            f"Generating embeddings for section {i+1}...")
                        )
                        processed_sections.append({
                            'chunks': section_chunks,
                            'embeddings': section_embeddings
                        })

                # Combine results
                all_chunks = []
                all_embeddings = []
                for section in processed_sections:
                    all_chunks.extend(section['chunks'])
                    if len(section['embeddings']) > 0:
                        all_embeddings.extend(section['embeddings'])

                embeddings = np.array(all_embeddings) if all_embeddings else np.array([])

            else:
                # Process normally for smaller documents
                chunks = self.chunk_text(text)
                if not chunks:
                    raise ValueError("Text chunking failed to produce any chunks")

                embeddings = self.create_embeddings(
                    chunks, 
                    lambda p: progress_callback(0.3 + 0.6 * p, "Processing embeddings...")
                )

            processing_time = time.time() - start_time

            # Calculate document statistics
            doc_stats = {
                "total_chunks": len(all_chunks) if "all_chunks" in locals() else len(chunks),
                "avg_chunk_size": sum(len(chunk.split()) for chunk in (all_chunks if "all_chunks" in locals() else chunks)) / len(all_chunks if "all_chunks" in locals() else chunks) if (all_chunks if "all_chunks" in locals() else chunks) else 0,
                "total_words": total_words,
                "processing_time": processing_time
            }

            if progress_callback:
                progress_callback(1.0, "Processing complete!")

            logger.info(f"Document processing completed in {processing_time:.2f} seconds")
            return {
                "chunks": all_chunks if "all_chunks" in locals() else chunks,
                "embeddings": embeddings,
                "stats": doc_stats
            }

        except Exception as e:
            logger.error(f"Error in process_document: {str(e)}")
            if "Processing cancelled by user" in str(e):
                raise Exception("Document processing was cancelled")
            raise

    def clear_cache(self):
        """Clear all cached data."""
        self.cache_manager.clear_cache()

    def clean_expired_cache(self):
        """Remove expired cache entries."""
        self.cache_manager.clean_expired_entries()