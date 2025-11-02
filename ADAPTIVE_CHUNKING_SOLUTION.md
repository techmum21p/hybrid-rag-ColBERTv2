# Adaptive Chunking Solution - Balancing Precision & Recall

## The Problem You Identified

**With 5 chunks**: Missing information from other relevant chunks (low recall)  
**With 20 chunks**: Model overwhelmed, only reads first few (low precision)

This is the classic **precision vs. recall tradeoff** in RAG systems.

## The Solution: Adaptive Top-K + Smart Context Management

### 1. Query Complexity Detection

The system now automatically detects if a query needs comprehensive answers:

**Simple queries** (e.g., "What is RAG?")
- Uses **7 chunks** (~5,600 chars)
- Model reads thoroughly, good precision

**Complex queries** (e.g., "List all chunking strategies")
- Uses **10 chunks** (~8,000 chars)  
- More comprehensive, better recall

**Keywords that trigger comprehensive mode:**
- "all", "list", "different", "various", "types of"
- "compare", "contrast", "similarities"
- "explain", "describe in detail"
- "multiple", "several", "many"

### 2. Smart Context Window Management

**Max context limit**: 6,000 characters (configurable)

The system:
1. Retrieves top-k chunks (7 or 10)
2. Adds chunks one by one
3. Stops if adding next chunk would exceed 6,000 chars
4. Warns you if chunks were dropped

### 3. Configuration Changes

```python
# Old (fixed):
final_top_k: int = 5  # Always 5 chunks

# New (adaptive):
final_top_k_min: int = 5      # Minimum (not currently used)
final_top_k_max: int = 10     # For complex queries
final_top_k_default: int = 7  # For normal queries
max_context_chars: int = 6000 # Hard limit on context size
```

## How It Works

### Example 1: Simple Query
```
Query: "What is semantic chunking?"

💡 Query complexity analysis: Using 7 chunks
🔍 Retrieving relevant chunks...
   • Rerank: 2.4s (top 7)

Context: 5,600 chars (all 7 chunks fit)
Answer: [Focused answer about semantic chunking]
```

### Example 2: Complex Query
```
Query: "List all the different chunking strategies in RAG"

💡 Query complexity analysis: Using 10 chunks
🔍 Retrieving relevant chunks...
   • Rerank: 2.8s (top 10)

Context: 6,000 chars (8 chunks fit, 2 dropped)
⚠️  Note: 2 additional chunks retrieved but not sent to LLM
    (exceeded max_context_chars limit of 6000)

Answer: [Comprehensive list from 8 chunks]
```

### Example 3: Manual Override
```python
# Force specific number of chunks
result = chatbot.chat("Your question", top_k=15)
```

## Benefits

### ✅ Better Recall
- Complex queries get more chunks (10 vs 5)
- Captures information spread across multiple sections
- Answers "list all..." questions completely

### ✅ Maintained Precision  
- Simple queries stay focused (7 chunks)
- Context limit prevents overwhelming model
- Model still reads all chunks thoroughly

### ✅ Transparency
- Shows you which strategy was chosen
- Warns if chunks were dropped
- Displays actual chunks used vs retrieved

### ✅ Flexibility
- Can manually override with `top_k` parameter
- Adjust `max_context_chars` for your model
- Tune keyword detection for your use case

## Configuration Tuning

### For Larger Models (7B+)
```python
final_top_k_default: int = 10   # More chunks for normal queries
final_top_k_max: int = 15       # Even more for complex queries
max_context_chars: int = 10000  # Larger models can handle more
```

### For Smaller Models (1.5B-3B)
```python
final_top_k_default: int = 5    # Fewer chunks for normal queries
final_top_k_max: int = 7        # Conservative for complex queries
max_context_chars: int = 4000   # Keep context small
```

### For Your Current Setup (3B model)
```python
final_top_k_default: int = 7    # Good balance
final_top_k_max: int = 10       # Comprehensive when needed
max_context_chars: int = 6000   # ~7-8 chunks × 800 chars
```

## Usage

### Automatic (Recommended)
```python
# System detects query complexity automatically
result = chatbot.chat("What are the different chunking strategies?")
# → Uses 10 chunks (detected "different")

result = chatbot.chat("What is semantic chunking?")
# → Uses 7 chunks (simple query)
```

### Manual Override
```python
# Force specific number for testing
result = chatbot.chat("Your question", top_k=5)   # Conservative
result = chatbot.chat("Your question", top_k=15)  # Comprehensive
```

## Expected Output

```
💡 Query complexity analysis: Using 10 chunks

🔍 Retrieving relevant chunks...
   • BM25s: 0.065s (33 results)
   • ColBERT: 0.162s (33 results)
   • Fusion: 0.000s (33 candidates)
   • Fetch: 0.006s (33 chunks)
   • Rerank: 2.449s (top 10)
   ✓ Total retrieval: 2.682s

============================================================
🐛 DEBUG: Context being sent to LLM
============================================================
Context length: 5847 characters
Chunks retrieved: 10
Chunks actually used: 7
...
============================================================

======================================================================
💬 Question: What are the different chunking strategies in RAG?
======================================================================

🤖 Answer:

There are several chunking strategies in RAG including fixed-size 
chunking, overlapping chunking, recursive chunking, semantic chunking,
and document-structure-aware chunking. Fixed-size chunking splits text
into predetermined sizes...

──────────────────────────────────────────────────────────────────────
⏱️  10.9s | 📚 7 chunks | 📝 5847 chars | 🎯 top_k=10
──────────────────────────────────────────────────────────────────────

📖 Sources Used:
──────────────────────────────────────────────────────────────────────
  1. [0.5583] Introduction... (792 chars)
  2. [0.5083] Introduction... (795 chars)
  3. [0.5041] Introduction... (732 chars)
  4. [0.4808] Introduction... (798 chars)
  5. [0.4772] Introduction... (730 chars)
  6. [0.4590] Introduction... (541 chars)
  7. [0.4330] Introduction... (791 chars)

  ⚠️  Note: 3 additional chunks retrieved but not sent to LLM
     (exceeded max_context_chars limit of 6000)
──────────────────────────────────────────────────────────────────────
```

## Validation Strategy

To validate if this solves your recall problem:

1. **Test comprehensive queries**:
   ```python
   chatbot.chat("List all the different chunking strategies mentioned")
   chatbot.chat("What are the various types of RAG techniques?")
   ```

2. **Compare with manual override**:
   ```python
   # Test with different top_k values
   result_5 = chatbot.chat("Your question", top_k=5)
   result_10 = chatbot.chat("Your question", top_k=10)
   result_15 = chatbot.chat("Your question", top_k=15)
   ```

3. **Check the source table**:
   - Are all expected sections represented?
   - Are chunks being dropped due to context limit?
   - Do you need to increase `max_context_chars`?

## Next Steps

1. **Restart Jupyter kernel** to load new code
2. **Re-run configuration cell** (Cell 2)
3. **Re-run chatbot cell** (Cell 10)
4. **Re-initialize your chatbot**
5. **Test with comprehensive queries**

## Troubleshooting

**Q: Still missing information**
- Increase `final_top_k_max` to 15
- Increase `max_context_chars` to 8000
- Check if relevant chunks are being retrieved at all

**Q: Model still overwhelmed**
- Decrease `max_context_chars` to 4000
- Decrease `final_top_k_default` to 5
- Your model might be too small for complex queries

**Q: Wrong complexity detection**
- Add/remove keywords in `comprehensive_keywords` list
- Use manual override: `chatbot.chat(query, top_k=10)`

## Summary

This adaptive approach gives you:
- **Better recall** for comprehensive queries (10 chunks)
- **Better precision** for simple queries (7 chunks)
- **Safety net** with max_context_chars (6000 limit)
- **Transparency** about what's being sent to LLM
- **Flexibility** to override when needed

The system now balances precision and recall automatically based on query complexity!
