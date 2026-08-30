# Chunking Strategy for PDFs with Charts/Screenshots

## Recommended Approach: Structure-Preserving Late Chunking

### Why This Works for User Manuals

1. **Preserves visual-text relationships** - Charts stay with their references
2. **Maintains document structure** - Sections, headings stay coherent
3. **No extra cost** - You're already using ColBERT v2
4. **Better context** - Embeddings understand full section context

## Implementation Plan

### Option A: Late Chunking with ColBERT (Recommended)

**How it works:**
```
1. Create larger "super-chunks" (e.g., full sections = 1000-2000 tokens)
2. Embed these super-chunks with ColBERT
3. At retrieval time, return relevant super-chunks
4. Optionally: break into smaller display chunks for LLM context
```

**Benefits:**
- Image descriptions stay with their sections
- Better semantic understanding (full section context)
- Works with your existing Jina ColBERT v2 model
- Minimal code changes

**Changes needed:**
- Increase `max_chunk_size` from 512 to 1500-2000 tokens
- Adjust `min_chunk_size` accordingly
- Keep section-based chunking (already doing this!)
- Don't split sections at paragraph boundaries unless >2000 tokens

### Option B: Agentic Chunking (Middle Ground)

**How it works:**
```
1. Use a small, fast LLM (llama3.2:3b) to identify:
   - Where images are referenced ("See Figure X", "shown below")
   - Start/end of logical sections
2. Create chunk boundaries based on these markers
3. Only call LLM on document structure, not every chunk
```

**Benefits:**
- More intelligent than pure structural
- Much cheaper than full LLM-based chunking
- Preserves image-text relationships

**Cons:**
- Still some LLM cost (but minimal)
- More complex implementation

### Option C: Enhanced Structure-Aware (Easiest)

**Improve your current approach:**

1. **Increase chunk sizes** for user manuals:
   ```python
   min_chunk_size: int = 512   # from 256
   max_chunk_size: int = 1500  # from 512
   chunk_overlap: int = 200    # from 128
   ```

2. **Add image-aware boundaries:**
   - Don't split chunks within N paragraphs of an image
   - Keep captions with their images
   - Detect patterns like "Figure X:", "Table Y:", etc.

3. **Section-based chunking** (you're already doing this!):
   - Keep entire subsections together when possible
   - Only split on major section boundaries

## Comparison Table

| Strategy | Complexity | Cost | Quality for Manuals | Your Codebase Fit |
|----------|-----------|------|---------------------|-------------------|
| Late Chunking (A) | Low | None | ⭐⭐⭐⭐⭐ | Perfect (using ColBERT) |
| Agentic (B) | Medium | Low | ⭐⭐⭐⭐ | Good (have Ollama) |
| Enhanced Structure (C) | Very Low | None | ⭐⭐⭐⭐ | Perfect (small tweaks) |
| LLM-based | High | High | ⭐⭐⭐⭐⭐ | Overkill |
| Semantic only | Medium | Medium | ⭐⭐ | Poor fit |

## Recommendation

**Start with Option C** (Enhanced Structure-Aware):
1. Increase your chunk sizes to 1500 tokens
2. Add more chunk overlap (200 tokens)
3. Ensure sections stay together

**Then optionally add Option A** (Late Chunking):
1. Your ColBERT model already supports this naturally
2. Just change how you create chunks (larger sections)
3. The embeddings will handle the rest

**Skip:**
- Pure semantic chunking (breaks structure)
- Full LLM-based chunking (too expensive)

## Code Changes Needed

### Quick Win (5 minutes):
```python
@dataclass
class RAGConfig:
    # Chunking - UPDATED for user manuals with images
    min_chunk_size: int = 512    # Increased from 256
    max_chunk_size: int = 1500   # Increased from 512
    chunk_overlap: int = 200     # Increased from 128
```

### Medium Effort (30 minutes):
Add image-proximity detection to `MarkdownSemanticChunker`:
- Detect image references in text
- Don't split chunks near images
- Keep captions with figures

### Long-term (if needed):
Implement proper late chunking by creating section-level chunks instead of arbitrary-size chunks.

## References

- Late Chunking: https://weaviate.io/blog/late-chunking
- ColBERT paper: https://arxiv.org/abs/2004.12832
- Jina ColBERT v2: https://huggingface.co/jinaai/jina-colbert-v2