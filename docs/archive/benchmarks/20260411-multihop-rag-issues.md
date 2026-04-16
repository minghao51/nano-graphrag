# Multihop RAG Benchmark Issues

> **Note:** These issues from April 2026 informed the implementation of entity-grounded RAG. See [`../architecture/entity-grounded-rag-design.md`](../architecture/entity-grounded-rag-design.md) for the implemented solution.

**Date:** 2026-04-11
**Dataset:** multihop_rag_small
**Config:** openrouter/@preset/cheap-fast (LLM), openrouter/qwen/qwen3-embedding-8b (embeddings)
**Results:** 0% exact match, 1.07% token F1

## Overview

Initial benchmark run on the multihop_rag dataset revealed significant performance issues. This document catalogs the key problems observed and provides guidance for future investigation.

## Key Issues

### 1. Verbose Response Format

**Problem:** The model produces excessively verbose, structured responses when concise answers are expected.

**Example:**
- **Question:** "Who is the individual associated with the cryptocurrency industry facing a criminal trial...?"
- **Expected Answer:** "Sam Bankman-Fried"
- **Actual Response:** A multi-paragraph structured essay with sections like "### Individual Identification and Criminal Charges", "### Prosecution Arguments and Defense Strategy", etc.

**Impact:** This dramatically reduces token F1 scores because the model generates hundreds of tokens when only a few are needed.

**Root Cause:** The LLM (openrouter/@preset/cheap-fast) may be fine-tuned for explanation rather than direct answering. The prompt format may also encourage verbosity.

**Potential Solutions:**
- Use more direct prompts that explicitly request concise answers
- Try models optimized for question answering rather than general chat
- Add explicit format instructions (e.g., "Answer in 5 words or less")
- Consider using the cheap-fast preset with a different base model

### 2. Missed Entity Connections

**Problem:** The model fails to connect entities across different sources in the corpus.

**Example:**
- **Question:** "Which individual is implicated in both inflating the value of a Manhattan apartment...?"
- **Expected Answer:** "Donald Trump"
- **Actual Response:** The model incorrectly identifies "Sam Bankman-Fried" and provides detailed legal context that doesn't match the question's specific real estate context.

**Impact:** Wrong entity identification leads to complete failure on multi-hop reasoning tasks.

**Root Cause:** The graph construction may not be capturing cross-document entity relationships effectively. The retrieval may also be missing relevant documents.

**Potential Solutions:**
- Investigate entity extraction quality - are entities being correctly identified across documents?
- Check if community detection is working properly
- Verify that the local retrieval is pulling relevant entities
- Consider increasing top_k for retrieval
- Review entity extraction prompts for multihop scenarios

### 3. Hallucination and Source Attribution Issues

**Problem:** The model generates confident responses that don't match the provided context, particularly around media outlet attributions.

**Example:**
- **Question:** "Do the TechCrunch article on software companies and the Hacker News article on The Epoch Times both report an increase in revenue...?"
- **Expected Answer:** "Yes"
- **Actual Response:** The model claims no information is available about TechCrunch or Hacker News, despite having relevant information in the corpus.

**Impact:** Even when the model has access to relevant information, it fails to retrieve and synthesize it correctly.

**Root Cause:** The retrieval may not be pulling the most relevant documents, or the model may be overly cautious about source attribution.

**Potential Solutions:**
- Investigate retrieval quality - are relevant documents being ranked highly?
- Consider using hybrid retrieval (dense + sparse)
- Review the prompts used for local query mode
- Test with different retrieval parameters (top_k, etc.)

### 4. Media Outlet Attribution Failures

**Problem:** The model consistently fails to attribute information to specific media outlets mentioned in questions.

**Pattern:** Across multiple questions, when asked about reporting from specific outlets (TechCrunch, Fortune, Sporting News, etc.), the model either:
1. Claims no information is available about those outlets
2. Correctly identifies the entity but notes the outlet isn't in the data

**Impact:** This suggests a disconnect between how questions are framed and how the corpus is indexed/retrieved.

**Root Cause:** The corpus may not index media outlets as first-class entities, or the retrieval doesn't prioritize document source information.

**Potential Solutions:**
- Ensure media outlets are extracted as entities during graph construction
- Consider adding document metadata to entity relationships
- Review how source attribution is handled in the retrieval pipeline

## Configuration Issues

### LLM Model Choice

The `openrouter/@preset/cheap-fast` preset routes to various low-cost models. These models may not be optimal for:
- Precise entity extraction
- Concise answer generation
- Multi-hop reasoning

**Recommendation:** Test with higher-quality models like:
- `openrouter/anthropic/claude-3.5-sonnet`
- `openrouter/meta-llama/llama-3.1-70b`

### Embedding Model

The `openrouter/qwen/qwen3-embedding-8b` embedding model was used. While Qwen is generally capable, different embedding models may perform better on this dataset.

**Recommendation:** Test with:
- `openrouter/text-embedding-3-small` (OpenAI via OpenRouter)
- `openrouter/sentence-transformers/all-mpnet-base-v2`

## Resolution

These issues led to the implementation of **entity-grounded RAG**, which addresses the core problems identified above:
- Entity canonicalization and alias resolution
- Answer validation against retrieved entities
- Improved cross-document entity connections
- More concise answer generation

See [`../architecture/entity-grounded-rag-design.md`](../architecture/entity-grounded-rag-design.md) for details on the implemented solution.

## Original Next Steps (Historical)

1. **Re-run with higher-quality LLM** - Test if performance improves with Claude 3.5 Sonnet
2. **Investigate entity extraction** - Review what entities are being extracted from the corpus
3. **Review retrieval quality** - Check if relevant documents are being retrieved
4. **Experiment with prompts** - Try more concise prompt formats
5. **Consider hybrid retrieval** - Add sparse retrieval to complement dense embeddings

## References

- Benchmark results: `benchmark_results/multihop_rag_small/multihop_rag_small_2026-04-02T16-48-34.096878.json`
- OpenRouter models: https://openrouter.ai/models
- Dataset location: `./workdirs/multihop_rag_small/datasets/multihoprag/`
