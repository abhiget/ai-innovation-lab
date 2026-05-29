"""
Document Indexing Script

Processes all PDFs in data/sample_documents/, generates embeddings, and indexes them.
Run this script to populate the Azure AI Search index with your documents.
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add src to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from chunking import DocumentChunker
from embedding import EmbeddingClient
from indexing import SearchIndexer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def main():
    """Main indexing pipeline."""
    try:
        logger.info("=" * 60)
        logger.info("Starting Document Indexing Pipeline")
        logger.info("=" * 60)
        
        # Initialize components
        logger.info("Initializing RAG components...")
        chunker = DocumentChunker(
            chunk_size=int(os.getenv("MAX_CHUNK_SIZE", 2000)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200))
        )
        embedding_client = EmbeddingClient()
        search_indexer = SearchIndexer()
        
        # Step 1: Create index
        logger.info("\nStep 1: Creating Azure AI Search index...")
        try:
            # Delete existing index first
            search_indexer.delete_index()
            
            # Create fresh index
            search_indexer.create_index()
            logger.info("✓ Index created successfully")
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise
        
        # Step 2: Process documents
        logger.info("\nStep 2: Processing PDF documents...")
        documents_dir = "data/sample_documents"
        
        if not os.path.exists(documents_dir):
            logger.error(f"Directory not found: {documents_dir}")
            return False
        
        pdf_files = list(Path(documents_dir).glob("*.pdf"))
        if not pdf_files:
            logger.error(f"No PDF files found in {documents_dir}")
            return False
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        all_chunks = []
        for pdf_path in pdf_files:
            logger.info(f"  Processing: {pdf_path.name}")
            try:
                text = chunker.extract_text_from_pdf(str(pdf_path))
                chunks = chunker.chunk_document(text, document_id=pdf_path.stem)
                all_chunks.extend(chunks)
                logger.info(f"    → Created {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"    ✗ Error: {e}")
                continue
        
        if not all_chunks:
            logger.error("No chunks created from documents")
            return False
        
        logger.info(f"✓ Total chunks created: {len(all_chunks)}")
        
        # Step 3: Generate embeddings
        logger.info("\nStep 3: Generating embeddings...")
        chunk_texts = [chunk["content"] for chunk in all_chunks]
        logger.info(f"Generating embeddings for {len(chunk_texts)} chunks...")
        
        embeddings = embedding_client.embed_batch(chunk_texts)
        
        for chunk, embedding in zip(all_chunks, embeddings):
            chunk["embedding"] = embedding
        
        logger.info(f"✓ Generated {len(embeddings)} embeddings")
        
        # Step 4: Index documents
        logger.info("\nStep 4: Indexing documents in Azure AI Search...")
        search_indexer.index_documents(all_chunks)
        logger.info(f"✓ Indexed {len(all_chunks)} documents")
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ Indexing Pipeline Completed Successfully!")
        logger.info("=" * 60)
        logger.info(f"\nSummary:")
        logger.info(f"  - Documents processed: {len(pdf_files)}")
        logger.info(f"  - Total chunks created: {len(all_chunks)}")
        logger.info(f"  - Index name: {os.getenv('AZURE_SEARCH_INDEX_NAME')}")
        logger.info(f"\nYou can now query your documents in the Streamlit app!")
        
        return True
        
    except Exception as e:
        logger.error(f"\n✗ Error during indexing: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
