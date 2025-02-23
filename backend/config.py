import os

# API Keys
MISTRAL_API_KEY = "eCWGXf3wRyVRqIG1t5VkHBHxrG9pTkSw"

# Vector Store Configuration
VECTOR_DIMENSION = 1024  # Mistral's embedding dimension
CHUNK_SIZE = 2000  # Reduced to avoid token limits
CHUNK_OVERLAP = 400  # Reduced proportionally
MAX_BATCH_SIZE = 5  # Reduced batch size for better token management
MAX_RECURSIVE_CHUNKS = 50  # Maximum chunks per recursive split

# Model Configuration
EMBEDDING_MODEL = "mistral-embed"  # Mistral's embedding model
LLM_MODEL = "mistral-large-latest"  # Mistral's latest large model

# System prompt for RAG
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context. 

When responding:
1. Start directly with your answer in clear, professional English
2. Structure your response in a clear, organized manner
3. Support your answers with relevant quotes from the provided context when appropriate
4. If you can't find enough information in the context to answer the question fully, say so clearly

Remember:
- Base your answers solely on the provided context
- Maintain a professional, clear writing style
- Do not add any prefixes or special characters to your responses
- Do not mention that you are an AI or assistant in your responses
"""
