# Quick Fix Reference Card

## THE ERROR
```
❌ Error downloading Jina ColBERT: No module named 'langchain.retrievers'
```

## THE FIX (3 Simple Steps)

### 1. Download the fixed installer
Use: `model_downloader_simplified.py`

### 2. Run it
```bash
python model_downloader_simplified.py --download-all
```

### 3. Verify
```bash
python model_downloader_simplified.py --test-only
```

## WHY THIS WORKS

| Issue | Explanation |
|-------|-------------|
| **Problem** | RAGatouille tries to import from old LangChain path |
| **Why** | Optional LangChain integration uses outdated imports |
| **Solution** | Install RAGatouille without LangChain dependencies |
| **Impact** | Zero - we don't use LangChain features anyway |

## WHAT GETS INSTALLED

```
✅ BM25s 0.2.6          (lexical search)
✅ RAGatouille 0.0.9    (ColBERT, standalone)
✅ Jina ColBERT v2      (model, ~500MB)
✅ PyMuPDF4LLM 0.0.17   (PDF processing)
✅ Transformers 4.45.0  (ML models)

❌ LangChain            (not needed!)
```

## ONE-LINER ALTERNATIVE

If you prefer manual installation:
```bash
pip install numpy scipy torch transformers sentence-transformers "bm25s[full]" PyStemmer pymupdf4llm markitdown --no-deps ragatouille && pip install tqdm datasets python-dotenv
```

Then download model:
```bash
python -c "from ragatouille import RAGPretrainedModel; RAGPretrainedModel.from_pretrained('jinaai/jina-colbert-v2')"
```

## VERIFICATION CHECKLIST

After installation, you should see:
- [x] BM25s works
- [x] RAGatouille imports successfully  
- [x] PyMuPDF4LLM available
- [x] Transformers available
- [x] Jina ColBERT v2 cached locally

## FILES PROVIDED

1. **model_downloader_simplified.py** - Fixed installer (USE THIS)
2. **LANGCHAIN_ISSUE_SOLUTION.md** - Detailed explanation
3. **DEPENDENCY_FIX_README.md** - Version info & background
4. **This file** - Quick reference

## REMEMBER

- ✅ You DON'T need LangChain for your RAG system
- ✅ RAGatouille works perfectly in standalone mode
- ✅ All core functionality is available
- ✅ The PyLate warning is just informational
- ✅ Your hybrid BM25s + ColBERT system will work great!

---
**Quick Start:** Run `python model_downloader_simplified.py --download-all`
