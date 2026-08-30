# Hallucination Fix - Applied Nov 2, 2024

## Problem Identified

The LLM was hallucinating despite retrieving relevant chunks because:

1. **Chunks were TOO LARGE** - Retrieved chunks contained 26,000+ characters each
2. **Context overload** - Small 3B parameter models can't handle massive context windows
3. **Weak grounding** - System prompt wasn't strong enough to prevent hallucination
4. **High temperature** - Temperature of 0.1 still allowed creative responses

## Root Cause

Your chunks showed:
```
Chunk 1: 26,117 total characters
Chunk 2: 25,906 total characters  
Chunk 3: 1,821 total characters
```

The display truncated these to 300 chars, but the **FULL 26K+ character chunks were being sent to the LLM**. This overwhelmed the small model, causing it to hallucinate instead of using the actual content.

## Solutions Applied

### 1. Chunk Truncation (CRITICAL)
**File**: Both `notebooks/00-doc-processor.ipynb` (Cell 12) and `local_rag_complete.py`

```python
def _build_context(self, chunks: List[Dict]) -> str:
    """Build context from retrieved chunks - IMPROVED with TRUNCATION"""
    MAX_CHUNK_CHARS = 800  # ~200 tokens per chunk
    
    for i, chunk in enumerate(chunks, 1):
        # TRUNCATE the chunk text to prevent overwhelming the model
        chunk_text = chunk['text']
        if len(chunk_text) > MAX_CHUNK_CHARS:
            chunk_text = chunk_text[:MAX_CHUNK_CHARS] + "..."
            print(f"⚠️  Truncated Source {i} from {len(chunk['text'])} to {MAX_CHUNK_CHARS} chars")
```

**Impact**: Reduces each chunk from 26K+ chars to max 800 chars (~200 tokens)

### 2. Ultra-Strong System Prompt
**File**: Both `notebooks/00-doc-processor.ipynb` (Cell 4) and `local_rag_complete.py`

```python
system_msg = f"""You are a STRICT document reader. You can ONLY read and quote from the documents below.

🚫 ABSOLUTE PROHIBITIONS:
- DO NOT use ANY knowledge from your training
- DO NOT make ANY assumptions
- DO NOT infer ANYTHING beyond what's explicitly written
- DO NOT add ANYTHING that's not in the documents

✅ WHAT YOU MUST DO:
1. Read ONLY the documents between === SOURCE X === markers
2. Find the EXACT text that answers the question
3. Quote or paraphrase ONLY that text
4. Cite the SOURCE number (e.g., "Source 1 states...")
5. If the answer is NOT in the documents, say: "This information is not in the provided documents."

📄 DOCUMENTS TO READ:
{context}

⚠️ REMEMBER: You are a READER, not a WRITER. Only extract what's already there!"""
```

**Impact**: Much stronger grounding instructions for small models

### 3. Zero Temperature
**File**: Both files

```python
"options": {
    "temperature": 0.0,  # ZERO temperature for maximum factuality!
    "top_p": 0.8,  # Reduced for more focused responses
    "top_k": 20,  # Limit vocabulary to most likely tokens
    "repeat_penalty": 1.2,  # Increased to prevent repetition
    "num_ctx": 4096  # Ensure enough context window
}
```

**Impact**: Eliminates creative/random responses, forces factual extraction

### 4. Clear Source Boundaries
**File**: Both files

```python
source_header = f"=== SOURCE {i} ==="
source_footer = f"=== END SOURCE {i} ==="
```

**Impact**: Helps model understand where each source begins/ends

### 5. Context-Only Queries
**File**: Both files

```python
if context:
    # For RAG queries: Only send the LATEST user question
    # This prevents hallucination from mixing old context with new queries
    chat_messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": messages[-1]["content"]}  # Only current question!
    ]
```

**Impact**: Prevents confusion from previous conversation history

## Files Modified

1. ✅ `/Users/aireesm4/Python_Projects/hybrid-rag-ColBERTv2/notebooks/00-doc-processor.ipynb`
   - Cell 4: OllamaClient.chat() with ultra-strong grounding
   - Cell 12: _build_context() with truncation

2. ✅ `/Users/aireesm4/Python_Projects/hybrid-rag-ColBERTv2/local_rag_complete.py`
   - OllamaClient.chat() method (lines 269-339)
   - RAGChatbot._build_context() method (lines 1419-1446)

## Testing Instructions

1. **Restart your Jupyter notebook kernel** to load the new code
2. Run a query that previously hallucinated
3. You should now see:
   - `⚠️  Truncated Source X from YYYY to 800 chars` messages
   - Responses that cite sources (e.g., "Source 1 states...")
   - Responses that say "This information is not in the provided documents" when appropriate
   - NO hallucinated information

## Expected Behavior

**Before Fix**:
- LLM receives 26K+ character chunks
- Gets overwhelmed and hallucinates
- Makes up information not in the documents

**After Fix**:
- LLM receives max 800 chars per chunk (3 chunks = 2400 chars total)
- Ultra-strong prompt forces grounding
- Temperature 0.0 eliminates creativity
- Responses are factual extractions from sources

## Why This Works

Small models (3B parameters) have limited:
1. **Context window capacity** - Can't process 26K chars effectively
2. **Instruction following** - Need very explicit, forceful prompts
3. **Grounding ability** - Easily drift into training data without strong constraints

By truncating chunks + ultra-strong prompts + zero temperature, we force the model to stay grounded in the provided text.

## Next Steps

If hallucination persists:
1. Reduce `MAX_CHUNK_CHARS` further (try 500 or 600)
2. Reduce `final_top_k` from 3 to 2 chunks
3. Consider using a larger model (7B or 13B parameters)
4. Add explicit examples in the system prompt

## Notes

- The truncation happens at **query time**, not indexing time
- Original chunks in the database remain unchanged
- You can adjust `MAX_CHUNK_CHARS` based on your model's capabilities
- Larger models (7B+) can handle more context (try 1200-1500 chars)
