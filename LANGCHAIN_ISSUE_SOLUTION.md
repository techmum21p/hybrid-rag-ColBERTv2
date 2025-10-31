# RAGatouille LangChain Dependency Issue - SOLUTION

## The Problem

When running `model_downloader.py`, you get:
```
❌ Error downloading Jina ColBERT: No module named 'langchain.retrievers'
```

Even after installing `langchain-community`, the error persists.

## Root Cause Analysis

**RAGatouille has OPTIONAL LangChain integration** that tries to import `from langchain.retrievers import ...`. This old import path doesn't work with newer LangChain versions (1.0+) which split functionality into separate packages.

The issue happens even though:
- We don't need LangChain for our RAG system
- RAGatouille can work perfectly fine without it
- The error occurs during model download, not during actual use

## Why langchain-community Didn't Fix It

The correct import in modern LangChain is:
```python
from langchain_community.retrievers import BM25Retriever  # ✅ Correct
```

But RAGatouille's optional code still uses:
```python
from langchain.retrievers import ...  # ❌ Old path
```

This causes the import to fail even when langchain-community is installed.

## Solution: Install RAGatouille Without LangChain

We have **3 options**:

### Option 1: Use Simplified Downloader (RECOMMENDED)
Use the provided `model_downloader_simplified.py`:

```bash
python model_downloader_simplified.py --download-all
```

**What it does:**
- Installs RAGatouille with `--no-deps` flag
- Manually installs only needed dependencies
- Skips LangChain entirely
- Still downloads Jina ColBERT v2 model
- All core RAG functionality works

### Option 2: Manual Installation
Do it step by step:

```bash
# 1. Install base dependencies
pip install numpy scipy torch transformers sentence-transformers

# 2. Install RAG libraries
pip install "bm25s[full]" PyStemmer

# 3. Install document processing
pip install pymupdf4llm markitdown

# 4. Install RAGatouille WITHOUT dependencies
pip install --no-deps ragatouille

# 5. Install RAGatouille's core dependencies (skip langchain)
pip install tqdm datasets python-dotenv

# 6. Download model in Python
python -c "from ragatouille import RAGPretrainedModel; RAGPretrainedModel.from_pretrained('jinaai/jina-colbert-v2')"
```

### Option 3: Install Old LangChain (NOT RECOMMENDED)
If you really need RAGatouille's LangChain integration:

```bash
pip install "langchain<1.0.0"  # Use old version
```

**Why not recommended:**
- Uses outdated LangChain version
- Conflicts with modern LangChain features
- We don't need LangChain for our RAG system

## Do We Need LangChain?

**NO!** Here's why:

### What RAGatouille's LangChain Integration Provides:
- `as_langchain_retriever()` - Convert RAGatouille to LangChain retriever
- `as_langchain_document_compressor()` - Use as reranker in LangChain

### What Our RAG System Uses:
- **Direct RAGatouille API** for ColBERT indexing and search
- **BM25s standalone** for lexical search  
- **PyMuPDF4LLM** for PDF processing
- **Custom fusion logic** for hybrid retrieval

**We never call LangChain integration methods!**

## Verification

After using the simplified installer, verify everything works:

```bash
python model_downloader_simplified.py --test-only
```

Expected output:
```
[Test 1] Testing BM25s...
  ✓ BM25s working correctly

[Test 2] Testing RAGatouille (standalone)...
  ✓ RAGatouille import successful
  ✓ Can be used for indexing and search

[Test 3] Testing PyMuPDF4LLM...
  ✓ PyMuPDF4LLM import successful

[Test 4] Testing Transformers...
  ✓ Transformers import successful

✅ ALL TESTS PASSED!
```

## Using RAGatouille Without LangChain

Here's how to use RAGatouille in standalone mode:

```python
from ragatouille import RAGPretrainedModel

# Load model (no LangChain needed)
model = RAGPretrainedModel.from_pretrained("jinaai/jina-colbert-v2")

# Index documents (no LangChain needed)
model.index(
    collection=["Document 1 text", "Document 2 text"],
    index_name="my_index"
)

# Search (no LangChain needed)
results = model.search("query text", k=10)

# Rerank (no LangChain needed)
reranked = model.rerank(
    query="query text",
    documents=["doc 1", "doc 2"],
    k=5
)
```

**Everything works perfectly without LangChain!**

## Package Versions After Fix

```
ragatouille==0.0.9 (without LangChain)
bm25s==0.2.6
transformers==4.45.0+
torch==2.0.0+
pymupdf4llm==0.0.17+
```

**NO langchain packages needed!**

## FAQ

### Q: Will this affect my RAG system?
**A:** No! Your RAG system doesn't use RAGatouille's LangChain integration.

### Q: Can I still use LangChain elsewhere?
**A:** Yes! Just don't install it with RAGatouille. Install separately if needed.

### Q: What about the PyLate migration warning?
**A:** That's unrelated. RAGatouille 0.0.9 works great with ColBERT. The warning is just informational about future versions.

### Q: Is this a permanent fix?
**A:** Yes. This is the correct way to use RAGatouille when you don't need LangChain integration.

### Q: Will RAGatouille still download models?
**A:** Yes! Model downloading doesn't need LangChain at all.

## Next Steps

1. **Run the simplified installer:**
   ```bash
   python model_downloader_simplified.py --download-all
   ```

2. **Verify it worked:**
   ```bash
   python model_downloader_simplified.py --test-only
   ```

3. **Continue with your RAG system:**
   - Your hybrid retrieval will work perfectly
   - BM25s + ColBERT fusion works
   - No LangChain needed!

## Summary

- ✅ **Problem:** RAGatouille's optional LangChain integration uses old import path
- ✅ **Solution:** Install RAGatouille without LangChain dependencies  
- ✅ **Impact:** None - we don't use LangChain integration anyway
- ✅ **Result:** All RAG functionality works perfectly

---

**File:** model_downloader_simplified.py  
**Status:** Ready to use  
**Dependencies:** Only core RAG libraries (no LangChain)
