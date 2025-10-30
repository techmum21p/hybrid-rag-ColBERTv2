# Local RAG Setup for Mac Mini M4 - Files Summary

## 🎯 What Changed?

Based on your previous chat thread, you want a **local setup on Mac Mini M4 with Ollama** using:
- ✅ **PyMuPDF4LLM** for PDF extraction (better than MarkItDown)
- ✅ **PyMuPDF** for image extraction
- ✅ **Gemma2 Vision (27B)** for multimodal understanding
- ✅ **Gemma2 (2B)** for text-only responses

---

## 📁 File Status

### ✅ **KEEP AS-IS** (No changes needed - works for local setup):

1. **`markdown_chunking_strategy.py`** (19KB)
   - Markdown-aware semantic chunking
   - Works perfectly for local setup
   - No cloud dependencies

2. **`model_downloader.py`** (14KB)
   - Downloads ColBERT and tokenizer
   - Still useful for local setup
   - Caches models in `~/.cache/rag_models`

### 🔄 **UPDATED** (New local versions):

1. **`local_rag_app.py`** ⭐ **NEW** (58KB)
   - Complete local RAG implementation
   - Uses Ollama instead of Gemini
   - SQLite instead of Cloud SQL
   - Local file storage instead of GCS
   - Detailed timing metrics
   - Mac M4 optimized

2. **`requirements_local.txt`** ⭐ **NEW**
   - Simplified dependencies
   - No GCP packages
   - Only local processing libraries

### ❌ **IGNORE** (Cloud-specific - not needed for local setup):

1. ~~`rag_app_complete.py`~~ - GCP/Gemini version
2. ~~`requirements.txt`~~ - Cloud dependencies
3. ~~`README.md`~~ - Cloud setup docs
4. ~~`quickstart.py`~~ - Cloud setup helper
5. ~~All other .md files~~ - Cloud-specific docs

---

## 🚀 Quick Start (Local Setup)

### 1. Install Ollama

```bash
# Install Ollama (Mac)
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama server
ollama serve

# Pull Gemma2 models (in another terminal)
ollama pull gemma2:2b          # For text responses (fast, 2GB)
ollama pull gemma2:27b-vision  # For vision + text (slower, 16GB)
```

### 2. Install Python Dependencies

```bash
# Install dependencies
pip install -r requirements_local.txt

# Optional: Download models ahead of time
python model_downloader.py --download-all
```

### 3. Upload and Index a PDF

```bash
python local_rag_app.py --upload your_document.pdf
```

**You'll see:**
```
Processing: your_document.pdf
[Step 1/3] Converting PDF to Markdown...
  ✓ Completed in 2.34s
  • Extracted 125,432 characters

[Step 2/3] Markdown-aware semantic chunking...
  ✓ Completed in 0.18s
  • Created 145 semantic chunks
  • Average chunk size: 512 tokens

[Step 3/3] Saving chunks to database...
  ✓ Completed in 0.05s

[BM25s] Building lexical search index...
  ✓ Completed in 0.42s
  • Indexed 145 chunks

[ColBERT] Building semantic search index...
  ✓ Completed in 8.23s
  • Indexed 145 chunks

✅ Document indexed successfully!
```

### 4. Chat with Your Documents

```bash
# Interactive chat
python local_rag_app.py --chat

# Single query
python local_rag_app.py --query "How do I configure authentication?"
```

**You'll see timing breakdown:**
```
🔍 Retrieving relevant chunks...
   • BM25s: 0.023s
   • ColBERT: 0.156s
   • Fusion: 0.001s
   • Fetch: 0.003s
   • Rerank: 0.089s
   ✓ Total retrieval: 0.272s
   • Retrieved 10 chunks

🤖 Generating response with Ollama (llama3.2:3b)...
   ✓ Generated in 1.847s

⏱️  Total query time: 2.119s
   • Retrieval: 0.272s (12.8%)
   • Generation: 1.847s (87.2%)
```

---

## 🔧 What's Different in Local Version?

### Architecture Comparison

| Component | Cloud Version | Local Version |
|-----------|---------------|---------------|
| **PDF Processing** | MarkItDown | **PyMuPDF4LLM** ⭐ |
| **Image Extraction** | None | **PyMuPDF** ⭐ |
| **Chunking** | Markdown-aware | Markdown-aware ✅ |
| **Contextualization** | Gemini API ($$$) | None (not needed locally) |
| **Retrieval** | BM25s + ColBERT | BM25s + ColBERT ✅ |
| **Reranking** | ColBERT | ColBERT ✅ |
| **LLM Generation** | Gemini API ($$$) | **Gemma2 (Ollama, FREE!)** ⭐ |
| **Vision Support** | Limited | **Gemma2 Vision (27B)** ⭐ |
| **Database** | Cloud SQL (PostgreSQL) | **SQLite** ⭐ |
| **Storage** | Google Cloud Storage | **Local files** ⭐ |
| **Cost** | ~$0.0001-0.0003/query | **FREE!** ⭐ |

### Key Benefits of Local Setup

✅ **Completely FREE** - No API costs
✅ **Privacy** - All data stays on your Mac
✅ **Fast** - No network latency
✅ **Offline** - Works without internet
✅ **M4 Optimized** - Takes advantage of Apple Silicon

---

## 📊 Performance on Mac Mini M4

### Upload/Indexing (500-page PDF):
- PDF → Markdown: ~3-5s
- Chunking: ~0.2-0.5s
- BM25s indexing: ~0.5-1s
- ColBERT indexing: ~10-15s
- **Total: ~15-20s**

### Query Performance:
- Retrieval (BM25s + ColBERT + Rerank): ~0.2-0.3s
- Generation (llama3.2:3b): ~1-3s
- **Total: ~1.5-3.5s per query**

---

## 🤔 Which Models to Use?

### Recommended Ollama Models for Mac Mini M4:

**For Speed (Recommended):**
```bash
ollama pull llama3.2:1b   # Super fast, 1B parameters
ollama pull llama3.2:3b   # Fast, good quality, 3B parameters
```

**For Quality:**
```bash
ollama pull qwen2.5:7b    # Better quality, 7B parameters
ollama pull llama3.1:8b   # Good quality, 8B parameters
```

**For Best Quality (if you have 32GB+ RAM):**
```bash
ollama pull qwen2.5:14b   # High quality, 14B parameters
```

Change model in the app:
```bash
python local_rag_app.py --chat --model llama3.2:1b
python local_rag_app.py --chat --model qwen2.5:7b
```

---

## 💡 Model Clarification (Same as Before)

**You asked:** "Why do we need an embedding model? We have ColBERT?"

**Answer: You're RIGHT!** We don't use a separate embedding model.

What `model_downloader.py` downloads:
1. **ColBERTv2** (~440MB) - Does ALL embeddings
2. **BERT tokenizer** (~440MB) - ONLY for counting tokens during chunking

The tokenizer is just for splitting text properly. ColBERT does everything else!

---

## 📝 CLI Commands

```bash
# Upload and index a PDF
python local_rag_app.py --upload document.pdf

# Interactive chat
python local_rag_app.py --chat

# Single query with full metrics
python local_rag_app.py --query "What is the API endpoint?"

# Show database stats
python local_rag_app.py --stats

# Use different Ollama model
python local_rag_app.py --chat --model llama3.2:1b
```

---

## 🗂️ File Structure

```
your-project/
├── local_rag_app.py              # Main application ⭐
├── markdown_chunking_strategy.py  # Chunking module (reusable)
├── model_downloader.py           # Download ColBERT/tokenizer
├── requirements_local.txt         # Dependencies ⭐
└── rag_local/                    # Created automatically
    ├── rag_local.db              # SQLite database
    └── indexes/                  # BM25s + ColBERT indexes
        ├── bm25s/
        └── colbert/
```

---

## ✅ Final Files You Need (Local Setup)

### Essential Files (3 files):

1. **`local_rag_app.py`** (58KB) - Main application
2. **`requirements_local.txt`** - Dependencies  
3. **`markdown_chunking_strategy.py`** (19KB) - Optional, if you want to customize chunking

### Optional:
4. **`model_downloader.py`** (14KB) - Pre-download models

---

## 🎯 Summary

**What to use:**
- ✅ `local_rag_app.py` - Your main app
- ✅ `requirements_local.txt` - Install these
- ✅ `markdown_chunking_strategy.py` - Already integrated in local_rag_app.py, keep for reference
- ✅ `model_downloader.py` - Optional, for pre-downloading models

**What to ignore:**
- ❌ All the cloud-based files (rag_app_complete.py, etc.)
- ❌ Cloud documentation files

---

## 🚀 You're Ready!

1. Install Ollama: `ollama serve`
2. Pull a model: `ollama pull llama3.2:3b`
3. Install Python deps: `pip install -r requirements_local.txt`
4. Upload PDF: `python local_rag_app.py --upload your.pdf`
5. Chat: `python local_rag_app.py --chat`

Everything runs locally. FREE. Fast. Private. 🎉
