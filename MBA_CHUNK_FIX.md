# MBA Attack Fix: Chunk-Level Membership Testing

## Problem Identified

The MBA attack was **fundamentally flawed** because it was testing membership of **full documents** when the vector store only contains **chunks**.

### What Was Wrong

1. **Ingestion Process** ([ingest_data.py](scripts/ingest_data.py)):
   - Documents are **chunked** into 512-character pieces (with 50-char overlap)
   - Only **chunks** are embedded and stored in ChromaDB
   - Each document → multiple chunks in vector store

2. **Original MBA Attack**:
   - Testing membership of **full concatenated documents** (4000+ characters)
   - Querying RAG with full masked document
   - RAG retrieves **chunks** from vector store (not the full document)
   - Mismatch: testing full docs vs. stored chunks

3. **Why It Failed**:
   - Full document never existed in vector store
   - RAG couldn't match full documents → retrieved random chunks
   - Attack accuracy was random (~30-60%) instead of high (~85-95%)
   - Large prompts printed because RAG retrieved multiple unrelated chunks

### The Fix

Modified `run_mba_experiments.py` to test **chunk-level membership**:

```python
# Create chunks from member and non-member passages
chunk_size = config["retrieval"]["chunk_size"]  # 512
chunk_overlap = config["retrieval"]["chunk_overlap"]  # 50

member_chunks = []
for qa in member_qa_pairs:
    for passage in qa.gold_passages:
        chunks = chunk_text(passage, chunk_size, chunk_overlap)
        for chunk in chunks:
            member_chunks.append({'text': chunk, 'pair_id': qa.pair_id})

# Test individual chunks (not full documents)
for chunk_data in test_member_chunks:
    target_chunk = chunk_data['text']
    attack_result = mba.attack(target_chunk, rag_system)
```

### Configuration Updates

Reduced parameters for smaller chunks ([config.yaml](config/config.yaml)):
- `M: 5` (was 10) - fewer masks for smaller chunks
- `max_document_words: 200` (was 500) - chunks are shorter
- Comments updated to reflect chunk-level testing

## Expected Results

With chunk-level testing:
- ✅ **Member chunks** should achieve ~85-95% accuracy (high similarity)
- ✅ **Non-member chunks** should achieve ~10-30% accuracy (low/random)
- ✅ Attack should correctly classify membership
- ✅ Smaller, focused prompts (one chunk + retrieved similar chunks)

## Why This Matters

Membership inference attacks must match the **granularity of storage**:
- If vector store has chunks → test chunk membership
- If vector store has full documents → test document membership
- Mismatch = invalid attack results

## Running the Fixed Attack

```bash
./run_mba_experiments.sh nq
```

Results will now test chunk membership correctly.
