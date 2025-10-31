# Local RAG Chatbot with Image Understanding 🤖

A complete, production-ready RAG (Retrieval-Augmented Generation) chatbot that runs **100% locally** on your Mac Mini M4 with support for PDF documents containing images.

## ✨ Key Features

- ✅ **100% Local**: No cloud APIs, no external dependencies
- ✅ **Image Understanding**: Extracts and analyzes images from PDFs using LLaVA vision model
- ✅ **No RAGatouille**: Direct Jina ColBERT v2 implementation (no dependency conflicts!)
- ✅ **Hybrid Retrieval**: BM25s (lexical) + ColBERT (semantic) + RRF fusion + reranking
- ✅ **Markdown-Aware Chunking**: Respects document structure (headings, sections, tables)
- ✅ **Fast**: ~300ms retrieval, 2-4s total response time
- ✅ **SQLite Storage**: Simple, portable database

## 🏗️ Architecture

```
PDF → PyMuPDF4LLM → Markdown
                       ↓
              Extract Images → LLaVA Analysis
                       ↓           ↓
              Markdown-Aware → Add Image
                 Chunking      Descriptions
                       ↓
              ┌────────┴────────┐
              ↓                 ↓
        BM25s Index      ColBERT Index
              ↓                 ↓
        Lexical Search   Semantic Search
              └────────┬────────┘
                       ↓
              RRF Fusion (Top 50)
                       ↓
           ColBERT Reranking (Top 10)
                       ↓
              Ollama Generation
                       ↓
            Response + Citations
```

## 📦 What's Included

### Files
- **local_rag_complete.py** - Complete implementation (1,400+ lines)
- **requirements.txt** - Python dependencies
- **README.md** - This file

### Key Components

1. **DocumentProcessor**
   - PDF → Markdown conversion (PyMuPDF4LLM)
   - Image extraction and storage
   - Vision model analysis (LLaVA)
   - Chunk enrichment with image context

2. **MarkdownSemanticChunker**
   - Respects H1/H2/H3 hierarchy
   - Merges small sections
   - Splits large sections intelligently
   - Preserves tables, lists, code blocks

3. **JinaColBERTRetriever**
   - Direct ColBERT v2 implementation
   - Token-level embeddings
   - MaxSim scoring
   - No RAGatouille dependency

4. **HybridRetriever**
   - Stage 1: BM25s (top 100)
   - Stage 2: ColBERT (top 100)
   - Fusion: RRF (top 50)
   - Stage 3: ColBERT reranking (top 10)

5. **RAGChatbot**
   - Ollama integration
   - Conversation history
   - Context-aware responses
   - Source citations

## 🚀 Quick Start

### Prerequisites

**Hardware:**
- Mac Mini M4 (or any Mac with M-series chip)
- 16GB RAM minimum
- 10GB free disk space

**Software:**
- macOS 12+
- Python 3.10+
- Ollama

### Step 1: Install Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama server (keep this running in a terminal)
ollama serve
```

In a **new terminal**, pull the models:

```bash
# Text generation model (~3.5GB)
ollama pull llama3.2:3b

# Vision model for image analysis (~4.5GB)
ollama pull llava:7b
```

### Step 2: Set Up Python Environment

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Upload Your First PDF

```bash
python local_rag_complete.py --upload /path/to/document.pdf
```

**Example output:**
```
============================================================
Processing: technical_manual.pdf
============================================================

[Step 1/5] Converting PDF to Markdown... ✓ 3.24s
  • Extracted 245,123 characters

[Step 2/5] Extracting and analyzing images...
    Analyzing image 1 on page 5... ✓ (14.8s)
    Analyzing image 2 on page 8... ✓ (15.2s)
    Analyzing image 3 on page 12... ✓ (14.9s)
  ✓ Completed in 45.18s
  • Extracted 3 images
  • Vision analysis: ✓

[Step 3/5] Markdown-aware semantic chunking... ✓ 0.23s
  • Created 287 semantic chunks

[Step 4/5] Enriching chunks with image context... ✓ 0.12s
  • 15 chunks enriched with image context

[Step 5/5] Saving chunks to database... ✓ 0.08s

[BM25s] Building lexical search index... ✓ 0.54s
[ColBERT] Building semantic search index... ✓ 12.87s

✅ Document indexed successfully!
```

### Step 4: Start Chatting

```bash
python local_rag_complete.py --chat
```

**Example conversation:**

```
You: What does the architecture diagram show?

🔍 Retrieving relevant chunks...
   • BM25s: 0.032s
   • ColBERT: 0.189s
   • Fusion: 0.001s
   • Fetch: 0.004s
   • Rerank: 0.095s
   ✓ Total retrieval: 0.321s

🤖 Generating response... ✓ 1.9s
⏱️  Total: 2.2s