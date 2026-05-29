"""
Document Chunking Module

Handles PDF parsing and intelligent text segmentation using PyPDF and LangChain.
Splits documents into manageable chunks with configurable overlap for RAG context windows.
"""

import os
from pathlib import Path
from typing import List, Dict, Any
import logging

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Handles PDF parsing and text chunking for RAG pipeline."""
    
    def __init__(
        self,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        separators: List[str] = None
    ):
        """
        Initialize the document chunker.
        
        Args:
            chunk_size: Maximum number of characters per chunk
            chunk_overlap: Number of overlapping characters between chunks
            separators: Text separators for intelligent splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators
        )
        
        logger.info(f"DocumentChunker initialized with chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        try:
            reader = PdfReader(pdf_path)
            text = ""
            
            for page_num, page in enumerate(reader.pages):
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.extract_text()
            
            logger.info(f"Extracted {len(text)} characters from {pdf_path}")
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            raise
    
    def chunk_document(self, text: str, document_id: str = None) -> List[Dict[str, Any]]:
        """
        Split document text into semantic chunks.
        
        Args:
            text: Document text to chunk
            document_id: Optional identifier for the source document
            
        Returns:
            List of chunk dictionaries with content and metadata
        """
        chunks = self.splitter.split_text(text)
        
        chunk_list = []
        for idx, chunk in enumerate(chunks):
            chunk_list.append({
                "content": chunk,
                "document_id": document_id,
                "chunk_index": idx,
                "chunk_count": len(chunks),
                "character_count": len(chunk)
            })
        
        logger.info(f"Created {len(chunk_list)} chunks from document {document_id}")
        return chunk_list
    
    def process_pdf_directory(self, directory: str) -> List[Dict[str, Any]]:
        """
        Process all PDF files in a directory.
        
        Args:
            directory: Path to directory containing PDF files
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        pdf_files = list(Path(directory).glob("*.pdf"))
        
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
        
        for pdf_path in pdf_files:
            try:
                text = self.extract_text_from_pdf(str(pdf_path))
                chunks = self.chunk_document(text, document_id=pdf_path.stem)
                all_chunks.extend(chunks)
                
            except Exception as e:
                logger.warning(f"Skipping {pdf_path}: {e}")
                continue
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    chunker = DocumentChunker(chunk_size=2000, chunk_overlap=200)
    
    # Example: Process a single PDF
    # chunks = chunker.chunk_document(
    #     text=chunker.extract_text_from_pdf("data/sample_documents/example.pdf")
    # )
    # print(f"Created {len(chunks)} chunks")
