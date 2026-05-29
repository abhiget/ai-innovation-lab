# Azure Cognitive RAG Engine - Beginner's Complete Guide

## 📚 Table of Contents
1. [Why We Created This Project](#why-project)
2. [What is the `src` Folder](#src-folder)
3. [Understanding Each File in `src`](#file-breakdown)
4. [Environment Variables (.env) Explained](#env-variables)
5. [How Streamlit Creates the Web Interface](#streamlit-web)
6. [Chat Flow Explained](#chat-flow)
7. [Complete Beginner Example](#beginner-example)

---

## Why We Created This Project {#why-project}

### Real-World Problem
Imagine you have **2 large PDF documents** (158,000+ characters each) about waste management, and you want to ask questions like:
- "What does SWM mean?"
- "How is waste segregated?"
- "What are disposal methods?"

**Without our app:**
- You'd have to manually open PDFs
- Search through hundreds of pages
- Read everything to find the answer

**With our RAG app:**
- Type a question in a chat window
- AI instantly finds the relevant parts of your documents
- AI explains the answer in your own words
- Shows you exactly which document sections it used

### Why Not Just Ask ChatGPT?
ChatGPT doesn't know about **your specific documents**. It only knows general knowledge trained on the internet. Our app teaches it about YOUR documents and answers questions based only on that information.

---

## What is the `src` Folder? {#src-folder}

### Folder Purpose
**`src`** = "Source code" - This is where all the main Python code lives.

### Why We Organize Code This Way

```
project-root/
├── src/                 ← All our custom code goes here
│   ├── __init__.py      ← Makes src a "package"
│   ├── chunking.py      ← Break PDFs into pieces
│   ├── embedding.py     ← Convert text to vectors
│   ├── indexing.py      ← Store in search engine
│   └── app.py           ← Create web interface
│
├── data/                ← Store PDF files
├── index_documents.py   ← Script to index (run once)
├── .env                 ← Secret keys
└── requirements.txt     ← List of libraries
```

### Good Practice
- Keeps project organized
- Easy to find code
- Professional structure
- Reusable code (can import from src)

---

## Understanding Each File in `src` {#file-breakdown}

### 1. `__init__.py` - Package Initialization

**What is it?**
- An empty Python file (or with imports) that tells Python "this folder is a package"

**Simple Explanation:**
- When Python sees `__init__.py`, it treats the folder as a package
- Allows us to do: `from src import DocumentChunker`

**What's Inside:**
```python
"""Azure Cognitive RAG Engine - Retrieval-Augmented Generation System"""

__version__ = "0.1.0"

from src.chunking import DocumentChunker
from src.embedding import EmbeddingClient
from src.indexing import SearchIndexer

__all__ = [
    "DocumentChunker",
    "EmbeddingClient", 
    "SearchIndexer",
]
```

**Why Need It?**
- Without it, Python won't recognize src as a package
- Other files couldn't import from src folder

---

### 2. `chunking.py` - Breaking PDFs into Pieces

**Purpose:** Read PDF files and split them into smaller chunks

**Real-World Analogy:**
```
Imagine you have a 500-page book:
- Instead of memorizing all 500 pages
- Split it into 50 chapters (chunks)
- Each chapter is easier to remember
- But keep a few sentences overlap between chapters
  (so context isn't lost)
```

**How It Works - Step by Step:**

#### Step 1: Read PDF File
```python
def extract_text_from_pdf(pdf_path):
    # Input: "data/sample_documents/SWM01_data.pdf"
    # What happens:
    #   1. Open the PDF file
    #   2. Extract all text from every page
    #   3. Return all text as one big string
    # Output: "SWM stands for Solid Waste... [158,189 characters]"
```

**Example:**
```
PDF File: SWM01_data.pdf (2 pages)
├─ Page 1: "SWM stands for Solid Waste Management.
│           It includes waste segregation..."
│
└─ Page 2: "Processing methods are incineration,
           composting, and landfill disposal..."

After extraction (one big text):
"SWM stands for Solid Waste Management. It includes... 
 Processing methods are incineration..."
```

#### Step 2: Split into Chunks
```python
def chunk_document(text, document_id):
    # Input: 158,189 character text
    # Configuration:
    #   - Max chunk size: 2,000 characters
    #   - Overlap: 200 characters
    #   - Separators: \n\n, \n, " ", ""
    #
    # What happens:
    #   1. Split at paragraph breaks (\n\n) first
    #   2. If still too long, split at line breaks (\n)
    #   3. If still too long, split at spaces
    #   4. Last resort: split any character
    #   5. Add 200-char overlap for context
    #
    # Output: 89 chunks (for SWM01)
    #         50 chunks (for SWM02)
    #         Total: 139 chunks
```

**Visual Example:**
```
Original Text (5000 chars):
"A B C D E F G H I J K L M N O P Q R S T U V W X Y Z..."

With chunk_size=2000, overlap=200:

Chunk 1: "A B C D E F G... X Y" (2000 chars)
                              ↓ (200 char overlap)
Chunk 2: "X Y Z ... AA BB CC" (2000 chars)
                              ↓ (200 char overlap)
Chunk 3: "CC DD EE... end" (remaining chars)
```

**Why Overlap?**
- Imagine a sentence spans two chunks
- Chunk 1 ends mid-sentence
- Chunk 2 starts with the end of that sentence
- 200-char overlap ensures we don't lose meaning

**Output Structure:**
```python
chunks = [
    {
        "content": "Waste segregation is the process of...",
        "document_id": "SWM01_data",
        "chunk_index": 0,
        "chunk_count": 89,
        "character_count": 1856
    },
    {
        "content": "Composting is a biological process...",
        "document_id": "SWM01_data",
        "chunk_index": 1,
        "chunk_count": 89,
        "character_count": 2000
    },
    # ... 87 more chunks
]
```

**Key Methods:**
```python
DocumentChunker
├─ extract_text_from_pdf()    # PDF file → raw text
├─ chunk_document()            # Text → chunks with metadata
└─ process_pdf_directory()     # All PDFs → all chunks
```

---

### 3. `embedding.py` - Converting Text to Vectors

**Purpose:** Convert text into numbers (vectors) that computers can compare

**Real-World Analogy:**
```
Imagine you assign each word a coordinate:
- "waste" = (1, 5, 2)
- "management" = (1.2, 4.8, 1.9)
- "segregation" = (1.1, 5.1, 2.1)

Similar words have similar coordinates!

"waste" and "trash" are close coordinates
"waste" and "dog" are far apart

This lets computers find similar chunks!
```

**How It Works:**

#### Step 1: Send Text to Azure OpenAI
```python
def embed_text(text):
    # Input: "Waste segregation is the process of..."
    # Process:
    #   1. Send text to Azure OpenAI
    #   2. Use model: text-embedding-3-large
    #   3. Azure returns 3,072 numbers
    #
    # Output: [0.123, -0.456, 0.789, ..., 2.341] (3072 numbers)
    #
    # These 3,072 numbers represent the MEANING of the text!
```

**What is text-embedding-3-large?**
- A special AI model that understands language
- It assigns coordinates (numbers) to text
- Similar meaning = similar coordinates
- 3,072 dimensions (very detailed!)

**Batching - Processing Multiple Texts:**
```python
def embed_batch(texts):
    # Input: 139 chunks to embed
    # We don't send all 139 at once (too slow)
    # Instead: Send 10 at a time
    #
    # Batch 1: chunks 0-9 → 10 vectors
    # Batch 2: chunks 10-19 → 10 vectors
    # ...
    # Batch 14: chunks 130-138 → 9 vectors
    #
    # Total: 139 vectors (3,072 dims each)
    #
    # This is MUCH faster than 139 individual requests!
```

**Output Structure:**
```python
embeddings = [
    [0.123, -0.456, 0.789, ...],  # Vector for chunk 0
    [0.234, -0.567, 0.890, ...],  # Vector for chunk 1
    [0.345, -0.678, 0.901, ...],  # Vector for chunk 2
    # ... 136 more vectors
]
```

**Key Methods:**
```python
EmbeddingClient
├─ embed_text()        # Single text → 3072-dim vector
├─ embed_batch()       # Multiple texts → vectors (batch processing)
└─ embed_chunks()      # Add vectors to chunk dictionaries
```

**Important:** We generate embeddings **twice**:
1. **During Indexing:** Embed all 139 document chunks once
2. **During Query:** Embed user's question to find similar chunks

---

### 4. `indexing.py` - Storing & Searching

**Purpose:** Store chunks + vectors in Azure AI Search and retrieve similar ones

**Real-World Analogy:**
```
Library with Two Search Methods:

1. KEYWORD SEARCH (Full-Text)
   "Find books about WASTE"
   → Returns all books with word "waste"
   
2. SEMANTIC SEARCH (Vector)
   "Find books semantically similar to GARBAGE"
   → Returns books about trash, refuse, etc.
   → Even if they don't use word "waste"

Our system uses BOTH!
```

**Part 1: Creating the Index**

```python
def create_index():
    # This creates the database structure in Azure AI Search
    #
    # Field Structure:
    index = {
        "id": {
            type: "String",
            key: True  # Unique identifier
        },
        "document_id": {
            type: "String",
            searchable: True,   # Can search this field
            filterable: True    # Can filter by this
        },
        "chunk_index": {
            type: "Int32",
            filterable: True
        },
        "content": {
            type: "String",
            searchable: True  # Full-text search happens here
        },
        "content_vector": {
            type: "Float array (3072 elements)",
            searchable: True,
            vector_search: True,  # Vector search happens here
            dimensions: 3072
        },
        "character_count": {
            type: "Int32",
            filterable: True
        }
    }
```

**What Each Field Does:**

| Field | Type | Purpose |
|-------|------|---------|
| `id` | String | Unique ID for chunk (e.g., "SWM01_chunk_0") |
| `document_id` | String | Source document (e.g., "SWM01_data") |
| `chunk_index` | Number | Position in document (0, 1, 2, ...) |
| `content` | Text | Actual text of chunk (what we search) |
| `content_vector` | 3072 numbers | Embedding vector (for semantic search) |
| `character_count` | Number | Size of chunk |

**Part 2: Uploading Chunks to Index**

```python
def index_documents(documents):
    # Input: 139 chunks with embeddings
    # Process:
    #   1. Prepare each chunk with all fields
    #   2. Upload to Azure AI Search
    #   3. Azure stores them in the index
    #
    # After this: Can search all 139 chunks instantly!
```

**Part 3: Searching (Hybrid)**

```python
def hybrid_search(query, query_embedding):
    # Two searches happen simultaneously:
    
    # SEARCH 1: VECTOR SEARCH (Semantic)
    vector_results = search_in_vectors(query_embedding)
    # Finds chunks with similar vectors
    # Example: User asks "garbage disposal"
    #         → Also finds chunks with "waste disposal"
    #         → Because they have similar vectors!
    
    # SEARCH 2: FULL-TEXT SEARCH (Keywords)
    text_results = search_in_text(query)
    # Traditional keyword matching
    # Example: User asks "waste segregation"
    #         → Finds chunks containing both words
    
    # MERGE RESULTS
    combined = merge_and_deduplicate(
        vector_results + text_results
    )
    # If same chunk in both searches → boost score
    
    # RETURN TOP 5
    return combined.sort_by_score()[:5]
```

**Scoring Example:**
```
Query: "waste segregation"
Query Vector: [0.123, -0.456, ...]

Chunk 1: "Waste segregation is..."
├─ Vector similarity score: 3.79 (high!)
├─ Full-text match: YES ("waste" and "segregation" both present)
└─ Final score: 3.79 + text boost = 3.79

Chunk 2: "Disposal of waste includes..."
├─ Vector similarity score: 1.43 (lower)
├─ Full-text match: PARTIAL ("waste" yes, "segregation" no)
└─ Final score: 1.43

Result: Chunk 1 ranked higher!
```

**Key Methods:**
```python
SearchIndexer
├─ create_index()        # Set up database structure
├─ delete_index()        # Clean up old index
├─ index_documents()     # Upload chunks to Azure
└─ hybrid_search()       # Vector + text search combined
```

---

### 5. `app.py` - The Web Interface & Chat

**Purpose:** Create interactive web interface and orchestrate the entire RAG system

**Two Main Components:**

#### Component A: RAGChatbot Class
```python
class RAGChatbot:
    # This is the "brain" - handles AI logic
    
    def __init__(self):
        # Load all the pieces:
        self.chunker = DocumentChunker()           # PDF splitter
        self.embedding_client = EmbeddingClient()  # Text → vectors
        self.search_indexer = SearchIndexer()      # Search engine
        self.chat_client = AzureOpenAI()           # GPT-4 LLM
    
    def chat(self, user_question):
        # Main method:
        # 1. Retrieve relevant chunks
        # 2. Generate answer
        # Return: {answer, sources}
```

**How RAGChatbot Works:**

```python
def chat(self, query):
    # STEP 1: GET CONTEXT
    context = self.retrieve_context(query)
    # └─ Embed query
    # └─ Search Azure AI Search
    # └─ Get top 5 chunks with sources
    
    # STEP 2: BUILD PROMPT
    prompt = f"""
    System: Answer using only provided context.
    
    Context Documents:
    {context}
    
    User Question: {query}
    """
    
    # STEP 3: CALL GPT-4
    response = self.chat_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7  # Balanced creativity
    )
    
    # STEP 4: RETURN ANSWER
    return {
        "answer": response.choices[0].message.content,
        "sources": context_sources
    }
```

#### Component B: Streamlit Web Interface

**What is Streamlit?**
- Python library that creates web apps WITHOUT HTML/CSS/JavaScript
- You write Python → Streamlit converts to web interface
- Runs on localhost (your computer)

**How Streamlit Creates Web Page:**

```python
import streamlit as st

def main():
    # This code runs in browser!
    
    st.set_page_config(
        page_title="Azure Cognitive RAG Engine",
        page_icon="🤖",
        layout="wide"  # Full width layout
    )
    
    st.title("🤖 Azure Cognitive RAG Engine")
    # Creates: <h1>🤖 Azure Cognitive RAG Engine</h1>
    
    st.write("Ask questions about your documents...")
    # Creates: <p>Ask questions about your documents...</p>
    
    # SIDEBAR (left panel)
    with st.sidebar:
        st.header("⚙️ Settings")
        k_results = st.slider(
            "Number of documents to retrieve",
            min_value=1,
            max_value=10,
            value=5
        )
        # Creates slider control
    
    # MAIN CONTENT (right panel)
    st.subheader("💬 Chat")
    
    # Chat input box
    if user_message := st.chat_input("Ask a question..."):
        # When user types and presses Enter:
        
        # Show user message
        with st.chat_message("user"):
            st.write(user_message)
        
        # Generate and show assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = chatbot.chat(user_message)
            
            st.write(result["answer"])
            
            # Show sources
            with st.expander("📄 Sources"):
                for source in result["sources"]:
                    st.write(f"- {source['document']}")

if __name__ == "__main__":
    main()
```

**Streamlit Behind the Scenes:**
```
Python Code (app.py):
├─ st.title()
├─ st.write()
├─ st.slider()
├─ st.chat_input()
└─ st.chat_message()
    │
    └─> Converts to HTML/CSS/JavaScript
        │
        └─> Renders in browser
            │
            └─> User sees web page!
```

**Session State (Remembering Chat History):**
```python
# When user refreshes page, chat disappears!
# Streamlit fixes this with session state:

if "messages" not in st.session_state:
    st.session_state.messages = []

# Now messages persist across page refreshes!
```

---

## Environment Variables (.env) Explained {#env-variables}

**What are environment variables?**
- Settings stored in a special file
- Not in your code (safer!)
- Your code reads them when it runs

**Why we need them:**
- API keys are SECRET (don't commit to GitHub!)
- Configuration might change (dev vs production)
- Different computers might have different settings

### The .env File Structure

```env
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=CJnYLoNBFIZd9Q3Zp4Htlo1E96VSYRyfbjPpm21aIcjDgexpcs4xJQQJ99CEAC77bzfXJ3w3AAABACOGnKCO
# └─ Secret key to access Azure OpenAI
# └─ NEVER share this!
# └─ NEVER commit to GitHub!

AZURE_OPENAI_ENDPOINT=https://openairagabhi.openai.azure.com/
# └─ Web address of your Azure OpenAI service
# └─ Format: https://{resource-name}.openai.azure.com/

AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4.1-mini
# └─ Name of the chat model you deployed in Azure
# └─ This is the model that generates responses

AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large
# └─ Name of the embedding model you deployed
# └─ This converts text to 3,072-dimensional vectors


# Azure AI Search Configuration
AZURE_SEARCH_ENDPOINT=https://aisearch-ragabhi.search.windows.net
# └─ Web address of your Azure AI Search service
# └─ Format: https://{service-name}.search.windows.net

AZURE_SEARCH_API_KEY=RwNYa7ztYj1Ew59MSux8dTcn58m4EXBSKBSLAWOu6rAzSeBSwMN8
# └─ Secret key to access Azure AI Search
# └─ NEVER share this!

AZURE_SEARCH_INDEX_NAME=abhiindex
# └─ Name of your search index (database)
# └─ Where we store all 139 chunks
# └─ We named ours "abhiindex"


# Application Settings
LOG_LEVEL=INFO
# └─ How much logging detail to show
# └─ INFO = normal, DEBUG = detailed, WARNING = errors only

MAX_CHUNK_SIZE=2000
# └─ Maximum characters per chunk
# └─ Smaller = more chunks, but lose context
# └─ Larger = fewer chunks, but might be too big for LLM

CHUNK_OVERLAP=200
# └─ Characters to overlap between chunks
# └─ Prevents meaning from being split
```

### How Python Reads These Values

```python
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Read values
api_key = os.getenv("AZURE_OPENAI_API_KEY")
# Returns: "CJnYLoNBFIZd9Q3Zp4Htlo1E96VSYRyfbjPpm21aIcjDgexpcs4xJQQJ99CEAC77bzfXJ3w3AAABACOGnKCO"

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
# Returns: "https://openairagabhi.openai.azure.com/"

# With defaults (if not in .env)
chunk_size = int(os.getenv("MAX_CHUNK_SIZE", 2000))
# If MAX_CHUNK_SIZE not set, uses 2000
```

### Security Best Practice

```
❌ DON'T DO THIS:
api_key = "CJnYLoNBFIZd9Q3Zp4Htlo1E96VSYRyfbjPpm21aIcjDgexpcs4xJQQJ99CEAC77bzfXJ3w3AAABACOGnKCO"

✅ DO THIS:
api_key = os.getenv("AZURE_OPENAI_API_KEY")

WHY?
- Hard-coded keys get leaked to GitHub
- Anyone can see your keys
- They can access your Azure resources!
```

### .gitignore Protects Your .env

```
# File: .gitignore
.env

# This tells Git: "Never commit .env file"
# So your keys stay safe!
```

---

## How Streamlit Creates the Web Interface {#streamlit-web}

### What Happens When You Run `streamlit run src/app.py`

```
Step 1: Terminal Command
$ streamlit run src/app.py

Step 2: Streamlit Starts Server
Streamlit server started on 0.0.0.0:8501
│
└─> Creates local web server on port 8501

Step 3: Open Browser
http://localhost:8501
│
└─> "localhost" = Your computer
    "8501" = Port number (address on computer)

Step 4: Page Loads
Browser connects to Streamlit server
│
Server runs app.py code
│
App generates HTML/CSS/JavaScript
│
Browser displays web page!
```

### Architecture Diagram

```
YOUR COMPUTER
│
├─ Terminal (PowerShell)
│  └─ python runs app.py
│
├─ Streamlit Server (localhost:8501)
│  ├─ Loads RAGChatbot
│  ├─ Loads environment variables
│  ├─ Renders Streamlit components
│  └─ Serves web page
│
└─ Browser (Chrome/Firefox/Edge)
   └─ Shows web page from server
```

### File Organization for Web Serving

```python
# In app.py

import streamlit as st  # Web framework

# This code runs on SERVER (Streamlit)
st.title("Azure Cognitive RAG Engine")
st.write("Ask questions...")

# This code runs on SERVER (Streamlit)
chatbot = RAGChatbot()

# When user interacts with web page:
user_query = st.chat_input("Ask...")
# User's input travels from browser → server

# Process on SERVER
result = chatbot.chat(user_query)

# Response travels back to browser
st.write(result["answer"])
# Display in browser
```

### Data Flow in Web Interface

```
BROWSER (Client):
User types: "What is SWM?"
│
└─ Sends to Streamlit Server

STREAMLIT SERVER:
│
├─ Receives message
├─ Runs: chatbot.chat(user_query)
│
├─ Calls Azure OpenAI (embedding)
├─ Calls Azure AI Search
├─ Calls Azure OpenAI (GPT-4 response)
│
└─ Sends answer back to browser

BROWSER:
Displays: "SWM stands for Solid Waste Management..."
And sources
```

### Session State (Persistent Memory)

**Problem:**
```python
st.chat_input("Ask...")
# Every time user refreshes page, this forgets everything!
```

**Solution:**
```python
import streamlit as st

# Check if this is first time
if "messages" not in st.session_state:
    st.session_state.messages = []

# Now, even after refresh, messages stay!
st.session_state.messages.append({
    "role": "user",
    "content": "What is SWM?"
})

# Messages persist until session ends!
```

---

## Chat Flow Explained {#chat-flow}

### Complete Journey of a Message

```
STEP 0: Setup (Once when app starts)
├─ Load all libraries
├─ Read .env file
├─ Create RAGChatbot instance
│  ├─ Initialize DocumentChunker
│  ├─ Initialize EmbeddingClient
│  ├─ Initialize SearchIndexer
│  └─ Initialize AzureOpenAI client
└─ Create Streamlit interface

───────────────────────────────────────────────────

STEP 1: User Types Question
┌──────────────────────────┐
│ Streamlit Web Page       │
│ ┌────────────────────┐   │
│ │ Input: "SWM means?"│   │
│ │ [Send Button]      │   │
│ └────────────────────┘   │
└──────────────────┬───────┘
                   │
                   ▼
         Streamlit Server receives
         user_query = "SWM means?"

───────────────────────────────────────────────────

STEP 2: Generate Query Embedding
         user_query = "SWM means?"
                 │
                 ▼
    embedding_client.embed_text(query)
                 │
                 ├─ Send to Azure OpenAI
                 │  Model: text-embedding-3-large
                 │
                 └─ Receive 3,072-dim vector
                    query_vector = [0.123, -0.456, ...]

───────────────────────────────────────────────────

STEP 3: Search Azure AI Search Index
         search_indexer.hybrid_search(query, vector)
                 │
                 ├─ VECTOR SEARCH
                 │  └─ Find 5 chunks with closest vectors
                 │     (chunks 0-4 from SWM01, SWM02)
                 │
                 ├─ FULL-TEXT SEARCH
                 │  └─ Find chunks with "SWM" or "means"
                 │     (might be different chunks)
                 │
                 └─ MERGE RESULTS
                    └─ Combine, deduplicate, score
                       Result: Top 5 chunks with scores
                       
                    Result example:
                    [
                      {"content": "SWM stands for...", score: 3.79},
                      {"content": "SWM includes...", score: 3.79},
                      {"content": "Processing includes...", score: 1.43},
                      ...
                    ]

───────────────────────────────────────────────────

STEP 4: Format Context for LLM
         contexts = [chunk1, chunk2, ..., chunk5]
                 │
                 ▼
    Format as string:
    
    "[Source: SWM01_data]
     SWM stands for Solid Waste Management...
     
     [Source: SWM02_data]
     SWM includes waste segregation..."

───────────────────────────────────────────────────

STEP 5: Build Prompt
         Combine everything for GPT-4:
         
    System Prompt:
    "You are a helpful assistant.
     Answer using only the provided context.
     If info not in context, say so."
    
    + User Message:
    "Context Documents:
     [Source: SWM01_data] ...
     [Source: SWM02_data] ...
     
     User Question: SWM means?"

───────────────────────────────────────────────────

STEP 6: Call Azure OpenAI GPT-4
         chat_client.chat.completions.create(
             model="gpt-4.1-mini",
             messages=[system_msg, user_msg],
             temperature=0.7,
             max_tokens=1000
         )
                 │
                 ├─ Sent to: https://openairagabhi.openai.azure.com/
                 │
                 ├─ GPT-4 reads context + question
                 │
                 └─ Returns generated response:
                    "SWM stands for Solid Waste Management,
                     as indicated by the context in the
                     documents discussing various aspects
                     of managing municipal solid waste..."

───────────────────────────────────────────────────

STEP 7: Format Response
         response = {
             "answer": "SWM stands for...",
             "sources": [
                 {"document": "SWM01_data", "score": 3.79},
                 {"document": "SWM02_data", "score": 1.43},
                 ...
             ]
         }

───────────────────────────────────────────────────

STEP 8: Display in Web Browser
         ┌──────────────────────────┐
         │ Streamlit Web Page       │
         │                          │
         │ Q: SWM means?            │
         │                          │
         │ 🤖 Assistant:            │
         │ SWM stands for Solid ... │
         │                          │
         │ 📄 Sources:              │
         │ • SWM01_data (3.79)      │
         │ • SWM02_data (1.43)      │
         │                          │
         │ [Input box for next Q]   │
         └──────────────────────────┘

───────────────────────────────────────────────────

STEP 9: Store in Chat History
         st.session_state.messages.append({
             "role": "user",
             "content": "SWM means?"
         })
         
         st.session_state.messages.append({
             "role": "assistant",
             "content": "SWM stands for...",
             "sources": [...]
         })
         
         (Messages persist until browser closed)
```

---

## Complete Beginner Example {#beginner-example}

### Real Scenario: "What doc is about tel me briefly"

**ACTUAL INPUT TO OUTPUT JOURNEY:**

```
═══════════════════════════════════════════════════════════════════════════════
USER INTERACTION (What you see on screen)
═══════════════════════════════════════════════════════════════════════════════

1. You type in chat box:
   Input: "what doc is about tel me briefly"
   [Press Enter]

2. You see "Thinking..." message

3. Response appears:
   Output: "Our documents contain information about..."
   Sources: SWM01_data, SWM02_data


═══════════════════════════════════════════════════════════════════════════════
BEHIND THE SCENES (What happens in background)
═══════════════════════════════════════════════════════════════════════════════

PHASE 1: EMBEDDING THE QUERY
──────────────────────────────
1. Query received: "what doc is about tel me briefly"

2. Call embedding function:
   embedding_client.embed_text(
       "what doc is about tel me briefly"
   )

3. Sent to Azure OpenAI:
   POST https://openairagabhi.openai.azure.com/
       /openai/deployments/text-embedding-3-large/
       embeddings?api-version=2024-08-01-preview
   
   Headers:
       Authorization: "Bearer {AZURE_OPENAI_API_KEY}"
   
   Body:
       {
           "input": "what doc is about tel me briefly"
       }

4. Azure OpenAI responds:
   {
       "data": [
           {
               "embedding": [
                   0.0234,
                   -0.0456,
                   0.0789,
                   ...(3069 more numbers)
               ]
           }
       ]
   }

5. Extract vector:
   query_vector = [0.0234, -0.0456, 0.0789, ...]
   Vector Length: 3,072 dimensions


PHASE 2: HYBRID SEARCH IN AZURE AI SEARCH
──────────────────────────────────────────
1. Call search with:
   - query_text: "what doc is about tel me briefly"
   - query_vector: [0.0234, -0.0456, ...]
   - k: 5 (top 5 results)

2. Vector Search (Semantic):
   Find chunks with SIMILAR VECTORS
   
   Azure compares query_vector against all 139 chunk vectors:
   ├─ Chunk 0: Distance = 0.05 (similar!) score: 4.2
   ├─ Chunk 1: Distance = 0.23 (not similar) score: 1.8
   ├─ Chunk 5: Distance = 0.04 (very similar!) score: 4.5
   ├─ Chunk 89: Distance = 0.30 (different) score: 0.9
   └─ ... (139 chunks compared)
   
   Top 5 vector results:
   ├─ Chunk 5 (SWM01_data): score 4.5
   ├─ Chunk 0 (SWM01_data): score 4.2
   ├─ Chunk 12 (SWM02_data): score 3.1
   ├─ Chunk 23 (SWM01_data): score 2.9
   └─ Chunk 67 (SWM02_data): score 2.7

3. Full-Text Search (Keywords):
   Find chunks containing words: "doc", "about"
   
   ├─ Chunk 2: Contains "document" and "about" score: 2.3
   ├─ Chunk 50: Contains "about" score: 1.5
   ├─ Chunk 15: Contains "documentation" score: 1.2
   └─ ...
   
   Top text results:
   ├─ Chunk 2 (SWM01_data): score 2.3
   ├─ Chunk 50 (SWM01_data): score 1.5
   ├─ Chunk 80 (SWM02_data): score 1.1
   └─ ...

4. Merge & Deduplicate:
   Combined = {
       Chunk 5: score 4.5 (vector only)
       Chunk 0: score 4.2 (vector only)
       Chunk 2: score 2.3 + 0.0 (text only, no vector match)
       Chunk 12: score 3.1 (vector only)
       Chunk 23: score 2.9 (vector only)
       ...
   }

5. Sort by score and return top 5:
   Final Results:
   [
       {
           "id": "SWM01_data_chunk_5",
           "document_id": "SWM01_data",
           "content": "Waste management includes segregation...",
           "score": 4.5
       },
       {
           "id": "SWM01_data_chunk_0",
           "document_id": "SWM01_data",
           "content": "SWM stands for Solid Waste Management...",
           "score": 4.2
       },
       {
           "id": "SWM02_data_chunk_12",
           "document_id": "SWM02_data",
           "content": "Documents cover various disposal methods...",
           "score": 3.1
       },
       {
           "id": "SWM01_data_chunk_23",
           "document_id": "SWM01_data",
           "content": "Processing methods include incineration...",
           "score": 2.9
       },
       {
           "id": "SWM02_data_chunk_67",
           "document_id": "SWM02_data",
           "content": "Environmental considerations in waste...",
           "score": 2.7
       }
   ]


PHASE 3: FORMAT CONTEXT
───────────────────────
Combine top 5 chunks into one text:

context_text = """
[Source: SWM01_data]
Waste management includes segregation...

[Source: SWM01_data]
SWM stands for Solid Waste Management...

[Source: SWM02_data]
Documents cover various disposal methods...

[Source: SWM01_data]
Processing methods include incineration...

[Source: SWM02_data]
Environmental considerations in waste...
"""


PHASE 4: BUILD PROMPT FOR GPT-4
────────────────────────────────
Combine system instruction + context + question:

full_prompt = """
System Instruction:
"You are a helpful assistant that answers questions based on 
provided documents. Answer using only the information from the 
provided context. If the context doesn't contain relevant 
information, say so clearly. Always cite the source document 
when providing information."

Context Documents:
[Source: SWM01_data]
Waste management includes segregation...

[Source: SWM01_data]
SWM stands for Solid Waste Management...

[Source: SWM02_data]
Documents cover various disposal methods...

[Source: SWM01_data]
Processing methods include incineration...

[Source: SWM02_data]
Environmental considerations in waste...

User Question:
"what doc is about tel me briefly"
"""


PHASE 5: CALL GPT-4
──────────────────
Send to Azure OpenAI:

POST https://openairagabhi.openai.azure.com/
    /openai/deployments/gpt-4.1-mini/
    chat/completions?api-version=2024-08-01-preview

Headers:
    Authorization: "Bearer {AZURE_OPENAI_API_KEY}"
    Content-Type: "application/json"

Body:
{
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant..."
        },
        {
            "role": "user",
            "content": "[full_prompt above]"
        }
    ],
    "temperature": 0.7,
    "max_tokens": 1000,
    "model": "gpt-4.1-mini"
}

GPT-4 reads everything and generates response token by token:

Token 1: "Our"
Token 2: "documents"
Token 3: "contain"
...
Token 87: "disposal"
Token 88: "."

Azure sends full response:

{
    "choices": [
        {
            "message": {
                "content": "Our documents contain information about Solid Waste Management (SWM), including waste segregation practices, processing methods such as incineration, composting, and landfill disposal, as well as environmental considerations. [Source: SWM01_data, SWM02_data]."
            }
        }
    ]
}


PHASE 6: EXTRACT & FORMAT RESPONSE
──────────────────────────────────
Parse response:

answer = "Our documents contain information about Solid Waste Management..."

sources = [
    {"document": "SWM01_data", "score": 4.5},
    {"document": "SWM01_data", "score": 4.2},
    {"document": "SWM02_data", "score": 3.1},
    {"document": "SWM01_data", "score": 2.9},
    {"document": "SWM02_data", "score": 2.7}
]

response_dict = {
    "answer": "Our documents contain information...",
    "sources": sources
}


PHASE 7: DISPLAY IN STREAMLIT
──────────────────────────────
Browser receives and displays:

┌────────────────────────────────────────────┐
│                                            │
│  Settings     Number of documents to...    │
│                                            │
│  Chat                                      │
│                                            │
│  Q: what doc is about tel me briefly       │
│                                            │
│  🤖 Assistant:                             │
│  Our documents contain information about   │
│  Solid Waste Management (SWM), including   │
│  waste segregation practices, processing   │
│  methods such as incineration, composting, │
│  and landfill disposal, as well as...      │
│                                            │
│  📄 Sources  ▼                             │
│  • SWM01_data (score: 4.5)                 │
│  • SWM01_data (score: 4.2)                 │
│  • SWM02_data (score: 3.1)                 │
│  • SWM01_data (score: 2.9)                 │
│  • SWM02_data (score: 2.7)                 │
│                                            │
│  [Ask a question about your documents...] │
│                                            │
└────────────────────────────────────────────┘


PHASE 8: SAVE TO SESSION MEMORY
────────────────────────────────
Chat history saved so messages don't disappear on refresh:

st.session_state.messages = [
    {
        "role": "user",
        "content": "what doc is about tel me briefly"
    },
    {
        "role": "assistant",
        "content": "Our documents contain information...",
        "sources": [...]
    }
]


═══════════════════════════════════════════════════════════════════════════════
TIMING SUMMARY
═══════════════════════════════════════════════════════════════════════════════

Query Embedding:        ~0.5 seconds
Hybrid Search:          ~1.0 second
Prompt Building:        ~0.1 seconds
GPT-4 Response:         ~2.0 seconds
Display in Browser:     ~0.2 seconds
───────────────────────────────────
TOTAL:                  ~3.8 seconds

(All happens invisibly while you see "Thinking...")
```

---

## Key Takeaways for Beginners

### 1. **Project Structure**
```
SRC folder = All custom code
├─ chunking.py = Break PDFs into pieces
├─ embedding.py = Text → numbers (vectors)
├─ indexing.py = Store & search data
├─ app.py = Web interface + orchestration
└─ __init__.py = Makes src a "package"
```

### 2. **Data Transformation**
```
PDFs → Chunks → Vectors → Indexed → Searchable
```

### 3. **Key Azure Services**
```
Azure OpenAI = Generate embeddings + responses
Azure AI Search = Store & search vectors + text
```

### 4. **Web Interface**
```
Streamlit = Python → Web App (no HTML/CSS/JS needed!)
localhost:8501 = Your computer serving the web page
```

### 5. **Chat Flow**
```
User Query → Embed → Search → Format → GPT-4 → Display → Save
```

### 6. **Security**
```
.env file = Secrets (never commit!)
.gitignore = Tells Git to ignore .env
```

---

## Quick Reference: Files & Their Jobs

| File | Purpose | Key Functions |
|------|---------|---|
| `__init__.py` | Package marker | None (can have imports) |
| `chunking.py` | PDF → Chunks | `extract_text_from_pdf()`, `chunk_document()` |
| `embedding.py` | Text → Vectors | `embed_text()`, `embed_batch()` |
| `indexing.py` | Store & Search | `create_index()`, `index_documents()`, `hybrid_search()` |
| `app.py` | Web + AI Logic | `RAGChatbot` class, Streamlit UI |
| `.env` | Configuration | API keys, endpoints, settings |
| `requirements.txt` | Dependencies | Package list |

---

This guide should help you understand every single piece! Happy learning! 🚀
