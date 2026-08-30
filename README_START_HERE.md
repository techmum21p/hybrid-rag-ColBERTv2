# 🚀 Start Here - Streamlit RAG Chatbot

## The Simplest Way to Run the App

### Just run this:

```bash
python3 start_rag.py
```

That's it! The launcher will:
- ✅ Check if Ollama is running
- ✅ Verify models are installed
- ✅ Show your data status
- ✅ Start the Streamlit app
- ✅ Open it in your browser

---

## What You Need

**Before running the launcher:**

1. **Start Ollama** (in another terminal):
   ```bash
   ollama serve
   ```

2. **Have the models installed** (one-time):
   ```bash
   ollama pull llama3.2:3b    # For chat
   ollama pull gemma3:4b      # For images
   ```

---

## Complete Workflow

### First Time

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Start the app
python3 start_rag.py
```

Then in the Streamlit UI:
1. Upload a PDF
2. Click "Index Document"
3. Wait for processing (2-5 minutes)
4. Start chatting!

### Next Times

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Start the app
python3 start_rag.py
```

The app auto-loads your data and chatbot is ready immediately! No need to re-upload.

---

## Quick FAQs

**Q: Do I need to manually start Ollama?**
A: Yes, in a separate terminal run `ollama serve`

**Q: What if I forgot to start Ollama?**
A: The launcher will remind you and exit gracefully

**Q: Will my data persist between sessions?**
A: Yes! Database and indexes are saved automatically

**Q: Can I add more documents later?**
A: Yes! Just upload another PDF and it rebuilds indexes with all documents

**Q: What if I upload a duplicate file?**
A: It warns you and automatically overwrites the old version

---

## Alternative Methods

If you prefer not to use the launcher:

**Manual (Two Terminals):**
```bash
# Terminal 1
ollama serve

# Terminal 2
streamlit run streamlit_rag.py
```

**Auto-start Ollama (Mac/Linux only):**
```bash
./start_rag.sh --start-ollama
```

---

## Need More Details?

- [QUICK_START.md](QUICK_START.md) - Full guide with troubleshooting
- [STREAMLIT_IMPROVEMENTS.md](STREAMLIT_IMPROVEMENTS.md) - Technical documentation

---

**Ready? Let's go!** → `python3 start_rag.py` 🎯
