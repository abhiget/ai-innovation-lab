"""
Main Application Module

Streamlit-based chatbot application that orchestrates the RAG pipeline.
Provides conversational interface for querying indexed documents using GPT-4o.
"""

import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

import streamlit as st
from openai import AzureOpenAI

from chunking import DocumentChunker
from embedding import EmbeddingClient
from indexing import SearchIndexer

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class RAGChatbot:
    """Orchestrates the RAG pipeline for conversational document retrieval."""
    
    def __init__(self):
        """Initialize RAG chatbot components."""
        self.chunker = DocumentChunker(
            chunk_size=int(os.getenv("MAX_CHUNK_SIZE", 2000)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200))
        )
        self.embedding_client = EmbeddingClient()
        self.search_indexer = SearchIndexer()
        
        # Initialize Azure OpenAI for chat
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.chat_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
        
        self.chat_client = AzureOpenAI(
            api_key=self.api_key,
            api_version="2024-08-01-preview",
            azure_endpoint=self.endpoint
        )
        
        logger.info("RAGChatbot initialized")
    
    def retrieve_context(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks for the query.
        
        Args:
            query: User query
            k: Number of results to retrieve
            
        Returns:
            List of relevant document chunks
        """
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_client.embed_text(query)
            
            # Perform hybrid search
            results = self.search_indexer.hybrid_search(query, query_embedding, k=k)
            
            logger.info(f"Retrieved {len(results)} context documents")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    def generate_response(self, query: str, context: List[Dict[str, Any]]) -> str:
        """
        Generate response using GPT-4o with retrieved context.
        
        Args:
            query: User query
            context: Retrieved document chunks
            
        Returns:
            Generated response
        """
        try:
            # Format context for the prompt
            context_text = "\n\n".join([
                f"[Source: {doc['document_id']}]\n{doc['content']}"
                for doc in context
            ])
            
            # System prompt
            system_prompt = """You are a helpful assistant that answers questions based on provided documents.
Answer the user's question using only the information from the provided context.
If the context doesn't contain relevant information, say so clearly.
Always cite the source document when providing information."""
            
            # User message
            user_message = f"""Context Documents:
{context_text}

User Question: {query}

Please provide a helpful answer based on the context above."""
            
            # Generate response
            response = self.chat_client.chat.completions.create(
                model=self.chat_deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            logger.info(f"Generated response ({len(answer)} characters)")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    def chat(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve and generate.
        
        Args:
            query: User query
            k: Number of context documents to retrieve
            
        Returns:
            Dictionary with answer and sources
        """
        # Retrieve context
        context = self.retrieve_context(query, k=k)
        
        if not context:
            return {
                "answer": "No relevant documents found for your query.",
                "sources": []
            }
        
        # Generate response
        answer = self.generate_response(query, context)
        
        return {
            "answer": answer,
            "sources": [
                {"document": doc["document_id"], "score": doc["score"]}
                for doc in context
            ]
        }


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Azure Cognitive RAG Engine",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Azure Cognitive RAG Engine")
    st.write("Ask questions about your documents using AI-powered retrieval and generation")
    
    # Initialize session state
    if "chatbot" not in st.session_state:
        try:
            st.session_state.chatbot = RAGChatbot()
            st.session_state.messages = []
        except Exception as e:
            st.error(f"Failed to initialize chatbot: {str(e)}")
            st.stop()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Settings")
        k_results = st.slider(
            "Number of documents to retrieve",
            min_value=1,
            max_value=10,
            value=5
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.write("""
        This RAG engine combines:
        - **Document chunking** with intelligent text splitting
        - **Vector embeddings** via Azure OpenAI
        - **Hybrid search** on Azure AI Search
        - **Generative AI** with GPT-4o responses
        """)
    
    # Chat interface
    st.subheader("💬 Chat")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📄 Sources"):
                    for source in message["sources"]:
                        st.write(f"- {source['document']} (score: {source['score']:.2f})")
    
    # User input
    if query := st.chat_input("Ask a question about your documents..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": query})
        
        with st.chat_message("user"):
            st.markdown(query)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Retrieving documents and generating response..."):
                result = st.session_state.chatbot.chat(query, k=k_results)
            
            st.markdown(result["answer"])
            
            if result["sources"]:
                with st.expander("📄 Sources"):
                    for source in result["sources"]:
                        st.write(f"- {source['document']} (score: {source['score']:.2f})")
        
        # Add assistant message to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"]
        })


if __name__ == "__main__":
    main()
