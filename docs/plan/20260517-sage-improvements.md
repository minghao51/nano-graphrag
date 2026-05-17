# Session Handoff: SAGE-Inspired Improvements for nano-graphrag

## 1. Primary Request and Intent

Analyze the SAGE paper (arXiv:2605.12061) and its reference codebase (anonymous.4open.science/r/Unified-Representation-A9D9/) to identify concrete improvements for nano-graphrag. Produce a phased implementation plan (4 phases) + a separate writeup on what remains for GFM/GNN/training work.

## 2. Key Technical Concepts

### SAGE Architecture

- **Memory Writer** (`memwriter/`): Policy-based graph construction via GRPO. Reward = deducibility + recall/precision of retrieved evidence + answer F1. Repetition penalty prevents triple duplication.
- **Memory Reader** (`memory_reader/`): 5-stage pipeline: Query Rewriting → NER+EL → GNN message passing over full KG → Document Ranking → Prompt Building.
- **Structured Query Planning** (`rewriter.py`): Two-stage (extractor + inferrer). Extracts explicit entities, aliases, relation clues, constraints, answer types + generates pseudo-queries with confidence scores.
- **Soft Entity Addressing**: Weighted entity mask from exact matches + alias matches + NER mentions + pseudo-query matches.
- **Structurally Gated Propagation**: Edge confidence/weight gates message propagation. Structural features (degree, PageRank, betweenness, clustering coefficient) control signal flow as "synaptic weights."
- **Query-conditioned Subgraph Selection** (Appendix I): Contrastive regularizer (query-subgraph alignment) + size regularizer (compactness) + connectivity smoothing regularizer.
- **GFM Pretraining**: KGC objective (BCELoss with adversarial temperature) + ListCELoss for ranking. Multi-graph pretraining for transferable structural priors.
- **Doc Rankers** (4 variants): `SimpleRanker` (sparse mmul entity→doc), `IDFWeightedRanker`, `TopKRanker`, `IDFWeightedTopKRanker`.

### nano-graphrag Current State

- **Extraction**: LLM-based, per-chunk entity/relationship via Pydantic structured output or legacy prompt parsing. Batched (N chunks/LLM call). Alias enrichment via second LLM pass. Concurrency: doc-level (4) × chunk-level (16).
- **Query** (4 modes):
  - `local`: Vector search → entities → one-hop neighbor expansion → community reports + text units + edges → token-budget truncation → CSV context → LLM.
  - `global`: Community schema → sorted by occurrence × rating → map-reduce over community reports.
  - `naive`: Vector search over chunks → direct LLM.
  - `entity_grounded`: Vector search → entity context → constrained generation + validation.
- **Graph**: NetworkX MultiGraph. Node attrs: entity_name, entity_type, description, source_id, aliases, clusters. Edge attrs: description, weight, confidence, relation_type, temporal fields.
- **Refinement pipeline**: merge (semantic dedup >=0.93), enrich (thin desc expansion, 3-layer validation), infer (co-occurrence + LLM, rejection cache).
- **Storage**: SQLite KV (documents, chunks, reports, manifests), HNSWVectorDB (entities, chunks), NetworkX (graph), all async.
- **Zero-shot**: No model training. Everything via LLM calls.

### Key Gaps

| Area | nano-graphrag | SAGE |
|------|--------------|------|
| Query understanding | Vector search only | LLM analysis + pseudo-queries |
| Entity activation | Hard top-k from vector DB | Soft weighted mask from multiple signals |
| Graph traversal | Uniform one-hop expansion | Confidence-gated propagation + structural features |
| Subgraph selection | Token-budget truncation only | Query-conditioned regularized selection |
| Feedback loop | None | Writer-reader GRPO loop |
| Training | Zero-shot | GFM pretrain + finetune + GRPO |
| Retrieval model | N/A | GNN with NBFNet message passing |

## 3. nano-graphrag Files to Modify

### Core query pipeline
- `src/nano_graphrag/_ops/query.py` (647 lines): `_build_local_query_context`, `_find_most_related_text_unit_from_entities`, `_find_most_related_edges_from_entities` — targets for Phases 1-3.
- `src/nano_graphrag/graphrag_query.py` (162 lines): Query dispatcher — plumb new params.
- `src/nano_graphrag/base.py` (389 lines): `QueryParam` dataclass — add fields for each phase.

### Extraction pipeline
- `src/nano_graphrag/_ops/extraction_writeback.py` (~300 lines): Entity writeback to graph + VDB — target for Phase 2 structural features storage.
- `src/nano_graphrag/_ops/extraction.py` (~150 lines): Extraction orchestrator — target for Phase 4 feedback integration.
- `src/nano_graphrag/_ops/extraction_structured.py` (275 lines): Batched structured extraction.

### Support files
- `src/nano_graphrag/_storage/gdb_networkx.py` (~400 lines): NetworkX backend — may need feature storage helpers.
- `src/nano_graphrag/_schemas.py` (222 lines): Pydantic models — may need new output models for query planner.
- `src/nano_graphrag/_entity_grounded_query.py` (398 lines): Partially overlaps with Phase 1 query planning.

## 4. Phase 1: Structured Query Planning

**Effort**: Low. **Impact**: High.

**New file**: `src/nano_graphrag/_query_planner.py`

**Files to modify**: `_ops/query.py`, `graphrag_query.py`, `base.py`

### Implementation details

1. **`QueryAnalysis` dataclass**:
```python
@dataclass
class QueryAnalysis:
    entities: list[tuple[str, list[str]]]  # (name, [aliases])
    relation_clues: list[str]
    constraints: dict[str, str]  # temporal, spatial, etc.
    answer_type: str | None
    pseudo_queries: list[tuple[str, float]]  # (query, confidence)
```

2. **`_analyze_query()` function**: Single LLM call with structured output:
   - Prompt based on SAGE's `rewriter.py` extractor stage
   - Output: Pydantic model mirroring `QueryAnalysis`
   - Cheap model function (consistent with SAGE's use of smaller LLM for this)

3. **`SoftEntityMask` class**:
```python
@dataclass
class SoftEntityMask:
    scores: dict[str, float]  # entity_id -> score [0, 1]
    signals: dict[str, list[tuple[str, float]]]  # entity_id -> [(signal_type, weight)]
```
   - Exact name match -> 1.0
   - Alias match -> 0.8
   - Query NER span match -> 0.6
   - Pseudo-query entity -> 0.4 * pseudo_confidence

4. **Integration into `_build_local_query_context()`**:
   - If `enable_query_planning=True`:
     - Run `_analyze_query()` first
     - Build `SoftEntityMask` from analysis
     - Run vector search as before for diversity
     - Re-rank results: `final_score = 0.6 * vector_sim + 0.4 * mask_score`
     - Fallback: pure vector search if analysis fails
   - entity_grounded mode also benefits from soft mask

5. **New `QueryParam` field**: `enable_query_planning: bool = True`

### Verification
- `uv run ruff check src/nano_graphrag/`
- Compare query results with/without planning on a known multi-hop question

## 5. Phase 2: Topological Structural Features

**Effort**: Low. **Impact**: Medium.

**New file**: `src/nano_graphrag/_ops/structural_features.py`

**Files to modify**: `_ops/extraction_writeback.py`, `_ops/query.py`, `base.py`

### Implementation details

1. **`compute_structural_features()` function**:
```python
async def compute_structural_features(
    graph: nx.MultiGraph
) -> dict[str, dict[str, float]]:
```
   - `degree_centrality`: nx.degree_centrality(G)
   - `pagerank`: nx.pagerank(G, alpha=0.85)
   - `betweenness_centrality`: nx.betweenness_centrality(G, k=int(sqrt(V)), normalized=True)
   - `clustering`: nx.clustering(G)
   - Normalize each to [0,1] per graph

2. **Storage**: Store as node attributes in NetworkX graph during writeback phase (in `_write_extraction_manifest` or after flush). Key: `structural_features`.

3. **Query-time composite scoring**: Replace single `rank` (degree) in `_build_local_query_context`:
   `composite_score = w1 * vector_sim + w2 * pagerank + w3 * clustering + w4 * degree`
   - Default: `w=[0.5, 0.2, 0.1, 0.2]`, configurable

4. **Edge gating signal**: In `_find_most_related_edges_from_entities`, compute `gate = confidence * weight * (source_pagerank + target_pagerank) / 2`. Filter edges with `gate < threshold`.

5. **New `QueryParam` fields**:
   - `structural_feature_weights: list[float] = [0.5, 0.2, 0.1, 0.2]`
   - `edge_gate_threshold: float = 0.0`

### Cache invalidation
- Features recomputed lazily after each extraction flush (not per-chunk)
- `_invalidate_structural_features()` flag on graph storage

## 6. Phase 3: Structurally-Gated Propagation

**Effort**: Medium. **Impact**: High.

**Files to modify**: `_ops/query.py`, `base.py`

No new files. This phase modifies `_find_most_related_text_unit_from_entities` and `_find_most_related_edges_from_entities`.

### Implementation details

1. **Multi-hop propagation config**:
   - New `QueryParam` field: `propagation_hops: int = 1`
   - When `hops > 1`: iterative expansion from initial entities
   - Each hop: query entity_vdb with previous-hop entity descriptions as query embedding

2. **Confidence-gated traversal** (synapse-inspired):
   - In `_find_most_related_edges_from_entities`, on top of existing `confidence_threshold` filter:
   - `gate_score = edge["confidence"] * (edge["weight"] / 10.0)`
   - `if gate_score < param.edge_gate_threshold / hop: continue`
   - The `1/hop` decay mimics SAGE's structural SNR gating

3. **Query-conditioned subgraph pruning** (inspired by SAGE Appendix I):
   After expansion in `_find_most_related_text_unit_from_entities`:
   - Re-rank all text units by combined score:
     `unit_score = w1 * relation_count + w2 * query_similarity(chunk_content)`
   - Prune bottom-percentile units before token-budget truncation
   - `query_similarity` via cheap embedding call

4. **Combined flow** for `_build_local_query_context`:
   ```
   seed_entities = vector_search(query, top_k) + soft_mask(query) [Phase 1 + Phase 2]
   for hop in 1..propagation_hops:
       neighbors = expand(seed_entities, gate=confidence*weight/hop)
       context = collect(neighbors, prune_by_query_similarity)
       seed_entities = neighbors  # for next hop
   ```

5. **New `QueryParam` fields**:
   - `propagation_hops: int = 1`
   - `subgraph_prune_ratio: float = 0.0` (0 = no pruning)
   - `edge_gate_decay: float = 1.0`

## 7. Phase 4: Retrieval Feedback Loop

**Effort**: Medium. **Impact**: Medium.

**New file**: `src/nano_graphrag/_ops/retrieval_feedback.py`

**Files to modify**: `_ops/extraction.py`, `_ops/query.py` (add logging), `graphrag_insert.py` (re-extraction trigger)

### Implementation details

1. **`RetrievalFeedbackTracker` dataclass**:
```python
@dataclass
class RetrievalFeedback:
    query: str
    doc_ids_retrieved: list[str]
    doc_ids_relevant: list[str]  # gold evidence (if known)
    answer_correct: bool | None   # from user feedback or LLM judge
    recall: float
    precision: float
    deducible: bool
```

2. **After each query**, if `enable_retrieval_feedback=True`:
   - Compute `r_rec = |retrieved ∩ relevant| / |relevant|` (coverage)
   - Compute `r_pre = |retrieved ∩ relevant| / |retrieved|` (precision)
   - LLM judge: "Can the answer be deduced from the following context?" -> `r_ded`
   - Log all to `document_index` KV under a `retrieval_feedback` key

3. **Document-level aggregation**:
   - Track per-document: `total_queries`, `avg_recall`, `avg_precision`, `avg_deducible`
   - Store in `document_index` under `feedback_stats`

4. **Re-extraction trigger** in `_ainsert_documents`:
   - After delta detection, check `feedback_stats` for existing docs
   - If `avg_recall < 0.5` and `queries > 3`: flag doc for re-extraction even if content hash unchanged
   - Force re-extraction by removing old manifest

5. **New `QueryParam` / config fields**:
   - `enable_retrieval_feedback: bool = False`
   - `re_extraction_recall_threshold: float = 0.5`

### Edge cases
- Can't compute precision/recall without gold evidence — use `r_ded` as fallback
- Rate-limit re-extraction to avoid thrashing (1 re-extraction per doc per hour max)

## 8. Appendix: Future GFM/GNN Work

### What remains outside the 4 phases

The following require adding PyTorch / PyTorch Geometric as dependencies, plus model training infrastructure. These are separate from the zero-shot LLM-only approach of nano-graphrag.

### 8a. GNN Entity Scorer

Replace the vector-search + heuristic ranking in local query with a learned GNN:

- **Architecture**: SAGE's `GNNRetriever` or `SAGERetriever` from `models.py`
  - Query embedding -> MLP -> initial node features
  - NBFNet message passing over KG (from `ultra/` library)
  - Output: entity relevance scores over all KG nodes
- **Key difference from current**: Scores ALL entities via graph propagation, not just top-k from vector search
- **Training**: Contrastive loss (ListCELoss from `losses.py`) on query-answer pairs from HotpotQA / MuSiQue
- **Inference**: ~O(V*E) per query vs O(V*log k) for vector search. SAGE uses Exphormer (sparse attention) for scalability.
- **Dependencies**: torch, torch-geometric, torch-sparse

### 8b. GFM Pretraining (KGC Objective)

Before finetuning the entity scorer, pretrain on KG completion:

- **Objective**: Score `(h, r, t)` triples via `(h * r * t).sum(dim=-1)` with BCELoss
- **Adversarial temperature** (SAGE `losses.py`): reweight negative samples via softmax over scaled logits
- **Augmentation**: GraphCL-style (node dropping, edge perturbation, attribute masking) for contrastive views
- **Data**: The same KG built by nano-graphrag's extraction pipeline, using existing entity/relationship data
- **Benefit**: Transferable structural priors across domains. The GFM learns what "hub" nodes look like, what "bridge" edges are, etc.

### 8c. GRPO Writer Training

Full RL-based graph construction as in SAGE's `memwriter/`:

- **Setup**: For each HotpotQA/MuSiQue example, LLM constructs KG from supporting docs
- **Reward**: `r_ded` (judge: can KG deduce answer?) + `r_rec`/`r_pre` (evidence recovery) + `r_ans` (answer F1)
- **Penalty**: `lambda_rep * repetition_rate` to prevent triple duplication
- **Algorithm**: GRPO (group-relative policy optimization) — standard clipped PPO variant
- **Training data pipeline**: `memwriter/data_preparation.py` (731 lines) — constructs system_prompt + user_prompt + reward_model_data
- **Challenge**: Requires training infrastructure (distributed RL, reward server)

### 8d. Exphormer Attention for Scaling

For graphs > 100k nodes, SAGE uses Exphormer (expander graph attention):

- Random-d-2 expander edges + virtual node connections + original KG edges
- Linear attention complexity instead of quadratic
- Enables full-graph message passing at scale
- Implementation: `ExphormerAttention` / `ExphormerEncoder` in `models.py` (PyG-based)

### Integration cost

| Component | New deps | Training data | Estimated effort |
|-----------|----------|---------------|-----------------|
| GNN Scorer | torch, torch-geometric | 1k-10k QA pairs | 2-3 weeks |
| GFM Pretrain | torch-geometric | Unlabelled KG | 1-2 weeks |
| GRPO Writer | trl/verl + reward server | 10k-50k KG examples | 3-4 weeks |
| Exphormer | torch-geometric + expander | Same as GNN | 1-2 weeks |

Each component can be added independently as an alternative backend, with the zero-shot LLM path as fallback.
