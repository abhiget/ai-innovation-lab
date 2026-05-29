"""Azure Cognitive RAG Engine - Retrieval-Augmented Generation System"""

__version__ = "0.1.0"
__author__ = "Your Name"

from src.chunking import DocumentChunker
from src.embedding import EmbeddingClient
from src.indexing import SearchIndexer

__all__ = [
    "DocumentChunker",
    "EmbeddingClient",
    "SearchIndexer",
]
