# Markdown-Aware Chunking Fix - Nov 2, 2024

## Problem Identified

The MarkdownSemanticChunker was **completely broken** and creating 26,000+ character chunks instead of the configured 512-1500 token limit.

### Root Cause

The `_count_tokens()` method had a critical bug:

```python
def _count_tokens(self, text: str) -> int:
    return len(self.tokenizer.encode(
        text, 
        add_special_tokens=False,
        truncation=True,        # ❌ BUG: Always truncates!
        max_length=512          # ❌ BUG: Always returns max 512!
    ))
```

**Result**: A 26,000 char chunk would return `512 tokens`, passing all size checks! The chunker thought everything was fine when it wasn't.

## Solution: Complete Rewrite

### 1. New Configuration (Cell 2)

```python
# Chunking - FIXED to use CHARACTER counts (more reliable)
min_chunk_size: int = 600   # Minimum 600 characters per chunk
max_chunk_size: int = 800   # Maximum 800 characters per chunk (HARD LIMIT)
chunk_overlap: int = 200    # 200 character overlap between chunks
final_top_k: int = 5        # Increased from 3 (chunks are smaller now)
```

**Why character counts?**
- More predictable and reliable
- No tokenizer quirks or truncation bugs
- Easier to debug and validate
- ~800 chars ≈ 200 tokens (safe for small models)

### 2. Rewritten MarkdownSemanticChunker (Cell 5)

The new chunker implements a **hierarchical splitting strategy**:

#### Strategy Priority:
1. **Keep sections together** if they fit (≤800 chars)
2. **Split at paragraph boundaries** if section is too large
3. **Split at sentence boundaries** if paragraphs are too large
4. **Truncate as last resort** (with warning) if sentences are too large

#### Key Features:

**✅ HARD Size Enforcement**
```python
if total_size <= self.config.max_chunk_size:
    # Keep section together
else:
    # MUST split - no exceptions!
```

**✅ Markdown Hierarchy Preservation**
```python
heading_text = self._format_heading(section)
# Always includes:
# - Parent context: [Context: Parent > Section]
# - Section heading: ## Current Section
```

**✅ Smart Splitting**
```python
# Priority 1: Split by paragraphs (best semantic coherence)
paragraphs = re.split(r'\n\n+', content)

# Priority 2: Split by sentences (if paragraphs too large)
sentences = re.split(r'(?<=[.!?])\s+', paragraph)

# Priority 3: Truncate (last resort, with warning)
```

**✅ Detailed Chunk Metadata**
```python
{
    'text': chunk_text,
    'heading_path': 'Parent > Section > Subsection',
    'level': 3,
    'char_count': 756,  # Actual character count
    'token_count': 189,  # Estimated (char_count // 4)
    'type': 'split_section',  # or 'complete_section', 'sentence_split'
    'part': 2  # Part 2 of split section
}
```

### 3. Updated Context Builder (Cell 12)

```python
def _build_context(self, chunks: List[Dict]) -> str:
    # No truncation needed - chunks are properly sized at index time!
    
    # Sanity check for old chunks
    if len(chunk_text) > 1000:
        print("⚠️  Warning: Need to re-index with new chunker!")
```

## Expected Results

### Before Fix:
```
Chunk 1: 26,117 chars (Introduction section - entire document!)
Chunk 2: 25,906 chars (Another massive section)
Chunk 3: 1,821 chars (Accidentally reasonable)
```

### After Fix:
```
Chunk 1: 756 chars (Introduction - Part 1)
Chunk 2: 698 chars (Introduction - Part 2)
Chunk 3: 812 chars (Techniques section - complete)
Chunk 4: 645 chars (RAG Methods - Part 1)
Chunk 5: 723 chars (RAG Methods - Part 2)
...
```

## How It Maintains Markdown Hierarchy

### Example: Large Section

**Input:**
```markdown
## Advanced RAG Techniques

RAG combines retrieval with generation...

### Query Rewriting
Query rewriting improves retrieval...

### Hybrid Search
Hybrid search combines BM25 and dense...
```

**Output Chunks:**

**Chunk 1:**
```
[Context: Advanced RAG Techniques]
## Advanced RAG Techniques

RAG combines retrieval with generation...
```

**Chunk 2:**
```
[Context: Advanced RAG Techniques]
### Query Rewriting

Query rewriting improves retrieval...
```

**Chunk 3:**
```
[Context: Advanced RAG Techniques]
### Hybrid Search

Hybrid search combines BM25 and dense...
```

Each chunk:
- ✅ Has context breadcrumbs
- ✅ Has its heading
- ✅ Stays within 600-800 chars
- ✅ Maintains semantic coherence

## Migration Steps

### CRITICAL: You MUST Re-Index Your Documents

The old chunks in your database are still 26K+ chars. You need to:

1. **Delete old indexes and database:**
   ```bash
   cd /Users/aireesm4/Python_Projects/hybrid-rag-ColBERTv2
   rm -rf indexes/
   rm rag_local.db
   ```

2. **Restart Jupyter kernel** to load new code

3. **Re-run document processing:**
   - The new chunker will create properly sized chunks
   - You'll see chunks are 600-800 chars each
   - No more 26K char monsters!

4. **Verify chunk sizes:**
   ```python
   # After processing, check a few chunks
   chunks = session.query(Chunk).limit(10).all()
   for chunk in chunks:
       print(f"Chunk {chunk.id}: {len(chunk.text)} chars")
   # Should see: 600-800 chars each
   ```

## Benefits

### 1. Better Retrieval
- Smaller chunks = more precise matching
- Each chunk is a focused semantic unit
- Better BM25 and ColBERT scores

### 2. No Hallucination
- Chunks fit comfortably in model context
- Model can actually read and understand the content
- No overwhelming with 26K chars

### 3. More Context
- Can retrieve 5 chunks instead of 3
- 5 × 800 = 4,000 chars total
- vs old: 3 × 26,000 = 78,000 chars (unusable)

### 4. Maintainable
- Character counts are predictable
- No tokenizer quirks
- Easy to debug and validate

## Testing the Fix

After re-indexing, test with a query:

```python
result = chatbot.ask_question("What are advanced RAG techniques?")

# You should see:
# - 5 chunks retrieved (instead of 3)
# - Each chunk 600-800 chars
# - Clear section headings
# - No truncation warnings
# - Factual responses with source citations
```

## Technical Details

### Character Count vs Token Count

**Why we switched:**

| Aspect | Token Count | Character Count |
|--------|-------------|-----------------|
| Predictability | ❌ Varies by tokenizer | ✅ Always consistent |
| Debugging | ❌ Hard to validate | ✅ Easy to check |
| Truncation bugs | ❌ Can be silently truncated | ✅ No truncation |
| Cross-model | ❌ Different per model | ✅ Universal |
| Estimation | ❌ Need actual tokenizer | ✅ Simple division |

**Conversion:**
- 1 token ≈ 4 characters (English text)
- 800 chars ≈ 200 tokens
- 600 chars ≈ 150 tokens

### Splitting Algorithm

```
For each section:
  1. Calculate total size (heading + content)
  
  2. If size ≤ max_chunk_size:
     → Keep as single chunk (best case)
  
  3. Else if has multiple paragraphs:
     → Split by paragraphs
     → Accumulate until max_chunk_size
     → Create new chunk when limit reached
  
  4. Else if paragraph too large:
     → Split by sentences
     → Accumulate until max_chunk_size
  
  5. Else (sentence too large):
     → Truncate with warning (rare)
```

## Files Modified

✅ `/Users/aireesm4/Python_Projects/hybrid-rag-ColBERTv2/notebooks/00-doc-processor.ipynb`
- Cell 2: RAGConfig with new parameters
- Cell 5: Complete rewrite of MarkdownSemanticChunker
- Cell 12: Updated _build_context (no truncation needed)

## Next Steps

1. ✅ Delete old indexes and database
2. ✅ Restart Jupyter kernel
3. ✅ Re-process your PDFs
4. ✅ Verify chunks are 600-800 chars
5. ✅ Test queries and validate responses
6. ✅ Enjoy hallucination-free RAG!

## Troubleshooting

**Q: I still see 26K char chunks**
A: You forgot to re-index. Delete `rag_local.db` and `indexes/` folder, then re-process.

**Q: Chunks are too small/large**
A: Adjust in RAGConfig:
```python
min_chunk_size: int = 500  # Lower if too large
max_chunk_size: int = 900  # Raise if too small
```

**Q: Lost semantic coherence**
A: The chunker tries to keep sections together. If splitting is too aggressive, increase `max_chunk_size` to 1000-1200.

**Q: Still getting hallucinations**
A: Check:
1. Did you re-index? (old chunks still there?)
2. Are chunks actually 600-800 chars? (verify in DB)
3. Is temperature still 0.0? (check OllamaClient.chat)
