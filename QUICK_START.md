# Quick Start Guide - Streamlit RAG Chatbot

## Three Ways to Run the App

### Option 1: Use the Launcher Script (Easiest!) 🚀

The launcher script checks everything for you and starts the app.

```bash
# Python launcher (cross-platform)
python3 start_rag.py

# Or bash launcher (Mac/Linux)
./start_rag.sh
```

**Features:**
- ✅ Checks if Ollama is running
- ✅ Verifies required models are installed
- ✅ Shows data status (database, indexes)
- ✅ Automatically opens browser

---

### Option 2: Manual Start (Two Terminals)

**Terminal 1 - Start Ollama:**
```bash
ollama serve
```

**Terminal 2 - Start Streamlit:**
```bash
streamlit run streamlit_rag.py
```

---

### Option 3: Auto-start Ollama (Bash only)

```bash
./start_rag.sh --start-ollama
```

This starts Ollama in the background automatically.

---

## First Time Setup

### 1. Install Required Models

```bash
ollama pull llama3.2:3b    # Chat model
ollama pull gemma3:4b      # Vision model for images
```

### 2. Start the App

Use any of the methods above.

### 3. Upload Your First Document

1. Click "Upload PDF" in the sidebar
2. Select a PDF file
3. Click "📥 Index Document"
4. Wait for processing (may take a few minutes)
5. Chatbot auto-initializes when done!

---

## What Happens When You Run the App

### First Time (No Data)
```
1. Database is created
2. Upload PDF → Processing starts
3. Text extraction + Image analysis
4. Index building (BM25 + ColBERT)
5. Chatbot ready!
```

### Subsequent Times (Data Exists)
```
1. Database loads
2. Indexes load
3. Chatbot auto-initializes
4. Ready to chat immediately!
```

### Adding More Documents
```
1. Upload another PDF
2. Checks for duplicates (warns if exists)
3. Processes new document
4. Rebuilds indexes with ALL documents
5. Chatbot reinitializes
6. All documents searchable!
```

---

## App Features

- ✅ **Hybrid Search**: BM25 + ColBERT + Reranking
- ✅ **Image Understanding**: Gemma3 vision model
- ✅ **OCR Text Extraction**: All image text searchable
- ✅ **Persistence**: Database and indexes saved between sessions
- ✅ **Duplicate Detection**: Warns and overwrites existing files
- ✅ **Auto-initialization**: Chatbot loads automatically if data exists
- ✅ **Multiple Documents**: Add documents without losing existing ones

---

## Troubleshooting

### "Ollama is not running"
```bash
# Start Ollama in another terminal
ollama serve

# Or check if it's already running
curl http://localhost:11434/api/tags
```

### "Model not found"
```bash
# Install required models
ollama pull llama3.2:3b
ollama pull gemma3:4b
```

### "Failed to initialize chatbot"
```bash
# Re-upload your documents to rebuild indexes
# Or check terminal for error messages
```

### Database locked / I/O errors
```bash
# Stop any running instances
# Delete database and restart:
rm rag_local.db
rm -rf indexes/
# Then re-upload documents
```

### App is slow / stuck
- First upload takes time (image analysis with vision model)
- Subsequent uploads are faster
- Check terminal for progress messages
- Ensure Ollama has enough memory

---

## File Locations

```
hybrid-rag-ColBERTv2/
├── rag_local.db              # SQLite database (documents, chunks, images)
├── indexes/
│   ├── bm25s/                # BM25 lexical search index
│   ├── colbert/              # ColBERT semantic search index
│   └── corpus_mapping.pkl    # Chunk ID mapping
├── extracted_images/         # Images extracted from PDFs
└── streamlit_rag.py         # Main app
```

---

## Tips

1. **Use descriptive filenames** - They appear in the UI
2. **PDFs with images work best** - Vision model analyzes them
3. **Check terminal output** - Shows progress and timing info
4. **Ask specific questions** - More focused = better results
5. **Check sources** - Expand to see images and metadata

---

## Keyboard Shortcuts

- `Ctrl+C` in terminal - Stop the app
- Click "🗑️ Clear Conversation" - Reset chat history
- Refresh browser - Reload app (keeps chatbot state)

---

## Need Help?

Check the detailed documentation:
- [STREAMLIT_IMPROVEMENTS.md](STREAMLIT_IMPROVEMENTS.md) - Technical details
- Terminal output - Shows progress and errors
- Streamlit UI - Shows warnings and status messages

---

## Quick Commands Reference

```bash
# Start everything
python3 start_rag.py

# Check Ollama
curl http://localhost:11434/api/tags

# List models
ollama list

# View logs (if using auto-start)
tail -f /tmp/ollama.log

# Clean slate (delete everything)
rm rag_local.db
rm -rf indexes/ extracted_images/
```

---

**Ready to start? Run:** `python3 start_rag.py` 🚀
