"""
Embedding Module

Handles text embedding generation using Azure OpenAI's text-embedding-3-small model.
Provides caching and batch processing for efficient embedding operations.
"""

import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

from openai import AzureOpenAI

logger = logging.getLogger(__name__)
load_dotenv()


class EmbeddingClient:
    """Manages text embeddings via Azure OpenAI service."""
    
    def __init__(self):
        """Initialize the Azure OpenAI embedding client."""
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
        
        if not self.api_key or not self.endpoint:
            raise ValueError("Missing AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT in environment")
        
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version="2024-08-01-preview",
            azure_endpoint=self.endpoint
        )
        
        logger.info(f"EmbeddingClient initialized with deployment: {self.embedding_deployment}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector (list of floats)
        """
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.embedding_deployment
            )
            
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding with dimension {len(embedding)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def embed_batch(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with batching.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to embed in each batch
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.embedding_deployment
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                logger.info(f"Embedded batch {i//batch_size + 1} ({len(batch)} texts)")
                
            except Exception as e:
                logger.error(f"Error embedding batch: {e}")
                raise
        
        logger.info(f"Successfully embedded {len(embeddings)} texts")
        return embeddings
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add embeddings to document chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'content' key
            
        Returns:
            Chunks with added 'embedding' key
        """
        chunk_texts = [chunk["content"] for chunk in chunks]
        embeddings = self.embed_batch(chunk_texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding
        
        logger.info(f"Added embeddings to {len(chunks)} chunks")
        return chunks


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    client = EmbeddingClient()
    
    # Example: Embed a single text
    # embedding = client.embed_text("This is a sample document about waste management.")
    # print(f"Embedding dimension: {len(embedding)}")
    
    # Example: Embed multiple texts
    # texts = ["Sample text 1", "Sample text 2", "Sample text 3"]
    # embeddings = client.embed_batch(texts)
    # print(f"Generated {len(embeddings)} embeddings")
