# Fixes Applied - Nov 1, 2025 (Latest Updates)

## Recent Fixes (6:56am - 7:00am)

### Issue 1: Images Saving to Wrong Location ✅ FIXED
**Problem**: When running in notebook, images were saved to `notebooks/extracted_images/` instead of project root `extracted_images/`

**Root Cause**: Relative paths in config were resolved relative to current working directory (notebook folder)

**Solution**:
- Modified `RAGConfig` to use absolute paths
- Added `base_dir` field that points to project root
- Added `__post_init__` method to set absolute paths:
  ```python
  base_dir: str = os.path.abspath(os.path.dirname(__file__))
  
  def __post_init__(self):
      if self.db_path is None:
          self.db_path = os.path.join(self.base_dir, "rag_local.db")
      if self.images_dir is None:
          self.images_dir = os.path.join(self.base_dir, "extracted_images")
  ```

**Result**: All images, indexes, and database now save to project root regardless of where script is run from

---

### Issue 2: Database Tables "Not Found" ✅ VERIFIED
**Problem**: User reported database has "no tables"

**Investigation**: Database actually DOES have tables (verified with sqlite3):
- `chunks` table ✓
- `documents` table ✓  
- `images` table ✓

**Root Cause**: User may have been checking wrong database file (there are two: one in notebooks/, one in project root)

**Solution**:
- Enhanced `print_stats()` to show database path and sample data
- Now displays:
  - Database file location
  - Images directory location
  - Recent documents with timestamps
  
**Action Required**: User should verify they're checking the correct database file

---

### Issue 3: Ollama Timeout During Chat ✅ FIXED
**Problem**: Chat requests timing out after 120 seconds with no response

**Root Cause**: 
- Default timeout (120s) too short for slower models
- No specific timeout error handling
- Model may be stuck or overloaded

**Solution**:
1. Increased timeout from 120s to 300s (5 minutes)
2. Added `ollama_timeout` config parameter
3. Improved error handling:
   ```python
   try:
       response = requests.post(url, json=payload, timeout=self.config.ollama_timeout)
       ...
   except requests.exceptions.Timeout:
       print(f"❌ Ollama timeout after {self.config.ollama_timeout}s - model may be too slow or stuck")
       return ""
   ```

**Recommendations**:
- Check if Ollama is running: `ollama ps`
- Try a faster model: `llama3.2:3b` instead of larger models
- Monitor Ollama logs for issues
- Restart Ollama if stuck: `pkill ollama && ollama serve`

---

### Issue 4: Database Column Name Mismatch ✅ FIXED
**Problem**: Code was using `metadata` but database schema has `chunk_metadata`

**Solution**: Fixed column name in chunk creation:
```python
chunk_record = Chunk(
    ...
    chunk_metadata=json.dumps({...})  # Was: metadata=...
)
```

---

### Issue 5: Tokenizers Parallelism Warning ✅ FIXED
**Problem**: Warning about forking after parallelism when building BM25s index

**Solution**: Added environment variable at script start:
```python
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

---

# Previous Fixes - Image Extraction and ColBERT Search Issues

## Issues Identified

### 1. Image Extraction Problem
**Symptom**: Images were not being saved correctly to disk. The `extracted_images` folder only contained `.DS_Store` file.

**Root Cause**: The code was writing raw image bytes directly to disk without proper format conversion. PyMuPDF's `extract_image()` returns raw image data that needs to be converted to a proper image format using PIL.

**Fix Applied**:
- Modified `extract_images_from_pdf()` method in both notebook and `local_rag_complete.py`
- Fixed naming conflict: Renamed PIL import to `PILImage` to avoid conflict with SQLAlchemy `Image` model
  ```python
  from PIL import Image as PILImage  # Avoid conflict with database model
  ```
- Now uses PIL (Pillow) to properly convert and save images:
  ```python
  pil_image = PILImage.open(io.BytesIO(image_bytes))
  if pil_image.mode == 'RGBA':
      pil_image = pil_image.convert('RGB')
  pil_image.save(image_path, 'PNG')
  ```
- Added error handling to catch and report image save failures

### 2. ColBERT Search Error
**Symptom**: Error during chat: "k of 50 is larger than the number of available scores, which is 1 (corpus size should be larger than top-k)"

**Root Cause**: The corpus only had 22 chunks, but the configuration was trying to retrieve `k=50` chunks. This happens when:
- Small documents are indexed
- The retrieval k parameter exceeds the actual corpus size

**Fix Applied**:
- Modified `retrieve()` method in `HybridRetriever` class
- Added dynamic k adjustment based on corpus size:
  ```python
  corpus_size = len(self.indexer.colbert_retriever.corpus)
  bm25_k = min(self.config.bm25_top_k, corpus_size)
  colbert_k = min(self.config.colbert_top_k, corpus_size)
  ```
- Added safety checks throughout the retrieval pipeline:
  - Limits fusion candidates to available results
  - Adjusts final_k based on candidate chunks
  - Added empty chunks check in rerank function

## Files Modified

1. **`/notebooks/00-doc-processor.ipynb`**
   - Cell 6: Updated `DocumentProcessor.extract_images_from_pdf()`
   - Cell 9: Updated `HybridRetriever.retrieve()` and `_colbert_rerank()`

2. **`/local_rag_complete.py`**
   - Lines 488-529: Updated `DocumentProcessor.extract_images_from_pdf()`
   - Lines 894-954: Updated `HybridRetriever.retrieve()`
   - Lines 996-1014: Updated `HybridRetriever._colbert_rerank()`

## Testing Instructions

### To verify image extraction fix:
1. Delete the old `extracted_images` folder (if it exists)
2. Re-run the document processing cell in the notebook
3. Check that images are properly saved in `extracted_images/` folder
4. Images should now be viewable as valid PNG files

### To verify ColBERT search fix:
1. Load the existing indexes (no need to re-index)
2. Run a chat query
3. You should see output like:
   ```
   🔍 Retrieving relevant chunks...
      • Corpus size: 22, using k=22 for retrieval
      • BM25s: 0.XXXs (22 results)
      • ColBERT: 0.XXXs (22 results)
      • Fusion: 0.XXXs (22 candidates)
      • Fetch: 0.XXXs (22 chunks)
      • Rerank: 0.XXXs (top 10)
   ```
4. No more "k is larger than corpus size" errors

### 3. Tensor Dimension Error (Single-Item Corpus)
**Symptom**: Error during chat: "Dimension out of range (expected to be in range of [-1, 0], but got 1)"

**Root Cause**: The `_maxsim_score` function in `JinaColBERTRetriever` was using `.squeeze()` which caused dimension issues when:
- Corpus has only 1 chunk
- Single document reranking
- Tensor operations expected specific dimensions

**Fix Applied**:
- Completely rewrote `_maxsim_score()` to handle various tensor dimensions properly
- Added special handling for single-item corpus in `search()` and `rerank()` methods
- Improved tensor dimension management:
  ```python
  # Handle single item corpus
  if len(self.corpus) == 1:
      return [{
          'document_id': 0,
          'score': float(scores.item() if scores.dim() == 0 else scores[0]),
          'text': self.corpus[0]
      }]
  ```
- Used proper tensor normalization and matrix multiplication for cosine similarity
- Added conditional squeezing based on tensor size

## Additional Improvements

- Added informative logging showing corpus size and actual k values used
- Added result counts at each stage of retrieval pipeline
- Better error handling for image processing failures
- More robust handling of edge cases (empty results, small corpora, single-item corpus)
- Improved tensor dimension handling for all corpus sizes

## Next Steps

If you want to re-process the document with proper image extraction:
1. Delete the database: `rm notebooks/rag_local.db`
2. Delete the indexes: `rm -rf notebooks/indexes/`
3. Delete the old images: `rm -rf notebooks/extracted_images/`
4. Re-run the document processing cells

The chat functionality should now work correctly with the existing indexes.
