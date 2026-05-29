"""
Indexing Module

Handles indexing of document chunks and their embeddings into Azure AI Search.
Provides index creation, update, and query capabilities for the RAG pipeline.
"""

import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch
)
from azure.search.documents.models import VectorizedQuery

logger = logging.getLogger(__name__)
load_dotenv()


class SearchIndexer:
    """Manages document indexing and search operations in Azure AI Search."""
    
    def __init__(self):
        """Initialize Azure AI Search client."""
        self.endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        self.api_key = os.getenv("AZURE_SEARCH_API_KEY")
        self.index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "rag-index")
        
        if not self.endpoint or not self.api_key:
            raise ValueError("Missing AZURE_SEARCH_ENDPOINT or AZURE_SEARCH_API_KEY in environment")
        
        self.credential = AzureKeyCredential(self.api_key)
        self.index_client = SearchIndexClient(self.endpoint, self.credential)
        self.search_client = SearchClient(self.endpoint, self.index_name, self.credential)
        
        logger.info(f"SearchIndexer initialized with index: {self.index_name}")
    
    def delete_index(self) -> None:
        """Delete the search index if it exists."""
        try:
            self.index_client.delete_index(self.index_name)
            logger.info(f"Deleted index: {self.index_name}")
        except Exception as e:
            if "not found" in str(e).lower():
                logger.info(f"Index does not exist: {self.index_name}")
            else:
                logger.error(f"Error deleting index: {e}")
    
    def create_index(self) -> None:
        """Create the search index with vector and semantic search configuration."""
        try:
            # Define fields
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SearchableField(name="document_id", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True),
                SearchableField(name="content", type=SearchFieldDataType.String),
                SearchField(
                    name="content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=3072,  # text-embedding-3-large dimension
                    vector_search_profile_name="myHnswProfile"
                ),
                SimpleField(name="character_count", type=SearchFieldDataType.Int32, filterable=True),
            ]
            
            # Configure vector search
            vector_search_config = VectorSearch(
                algorithms=[HnswAlgorithmConfiguration(name="myHnsw")],
                profiles=[
                    VectorSearchProfile(
                        name="myHnswProfile",
                        algorithm_configuration_name="myHnsw"
                    )
                ]
            )
            
            # Configure semantic search
            semantic_config = SemanticConfiguration(
                name="my-semantic-config",
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[SemanticField(field_name="content")],
                    keywords_fields=[SemanticField(field_name="document_id")]
                )
            )
            
            semantic_search_config = SemanticSearch(configurations=[semantic_config])
            
            # Create index
            index = SearchIndex(
                name=self.index_name,
                fields=fields,
                vector_search=vector_search_config,
                semantic_search=semantic_search_config
            )
            
            self.index_client.create_index(index)
            logger.info(f"Successfully created index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise
    
    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Upload documents to the search index.
        
        Args:
            documents: List of documents with id, content, and embedding
        """
        try:
            # Prepare documents for indexing
            indexed_docs = []
            for idx, doc in enumerate(documents):
                indexed_docs.append({
                    "id": f"{doc.get('document_id', 'doc')}_chunk_{idx}",
                    "document_id": doc.get("document_id"),
                    "chunk_index": doc.get("chunk_index", 0),
                    "content": doc.get("content", ""),
                    "content_vector": doc.get("embedding", []),
                    "character_count": doc.get("character_count", 0)
                })
            
            # Batch upload (max 1000 docs per batch)
            for i in range(0, len(indexed_docs), 1000):
                batch = indexed_docs[i:i+1000]
                self.search_client.upload_documents(batch)
                logger.info(f"Uploaded batch {i//1000 + 1} ({len(batch)} documents)")
            
            logger.info(f"Successfully indexed {len(indexed_docs)} documents")
            
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            raise
    
    def hybrid_search(self, query: str, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector and full-text search.
        
        Args:
            query: Text query
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of search results with scores
        """
        try:
            # Create vectorized query for the newer API
            vector_query = VectorizedQuery(vector=query_embedding, k_nearest_neighbors=k, fields="content_vector")
            
            # Vector search
            vector_search_results = list(self.search_client.search(
                search_text="",
                vector_queries=[vector_query],
                top=k
            ))
            
            # Full-text search
            text_search_results = list(self.search_client.search(
                search_text=query,
                top=k
            ))
            
            # Combine and deduplicate results
            results_dict = {}
            for doc in vector_search_results:
                results_dict[doc["id"]] = {
                    "id": doc["id"],
                    "content": doc["content"],
                    "document_id": doc.get("document_id", "unknown"),
                    "score": doc["@search.score"],
                    "search_type": "vector"
                }
            
            for doc in text_search_results:
                if doc["id"] in results_dict:
                    # Increase score if found in both searches
                    results_dict[doc["id"]]["score"] += doc["@search.score"]
                    results_dict[doc["id"]]["search_type"] = "hybrid"
                else:
                    results_dict[doc["id"]] = {
                        "id": doc["id"],
                        "content": doc["content"],
                        "document_id": doc.get("document_id", "unknown"),
                        "score": doc["@search.score"],
                        "search_type": "text"
                    }
            
            # Sort by score and return top k
            sorted_results = sorted(results_dict.values(), key=lambda x: x["score"], reverse=True)[:k]
            logger.info(f"Hybrid search returned {len(sorted_results)} results")
            
            return sorted_results
            
        except Exception as e:
            logger.error(f"Error performing search: {e}")
            raise


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    indexer = SearchIndexer()
    
    # Example: Create index
    # indexer.create_index()
    
    # Example: Index documents
    # documents = [...]  # Prepare documents with embeddings
    # indexer.index_documents(documents)
    
    # Example: Search
    # results = indexer.hybrid_search("query text", query_embedding, k=5)
