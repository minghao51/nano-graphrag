# Benchmark Configurations - Complete ✅

**Date:** 2026-03-24
**Status:** Ready to run

---

## What Was Created

### 📁 Benchmark Configurations (4 datasets)

All configurations are in `experiments/`:

1. **`benchmark_multihop_rag.yaml`** - MultiHop-RAG dataset (2,556 samples)
2. **`benchmark_musique.yaml`** - MuSiQue dataset (2,417 samples)
3. **`benchmark_hotpotqa.yaml`** - HotpotQA dataset (7,405 samples)
4. **`benchmark_2wiki.yaml`** - 2WikiMultiHopQA dataset (12,576 samples)

### 🛠️ Utility Scripts

1. **`run_all_benchmarks.sh`** - Run all benchmarks sequentially
   - Supports `--quick` mode for testing (50 samples)
   - Tracks duration and reports results
   - Creates results directory automatically

2. **`compare_results.py`** - Compare and visualize results
   - Prints comparison tables
   - Shows improvement vs baseline
   - Calculates aggregate statistics

3. **`validate_setup.py`** - Validate environment setup
   - Checks dependencies
   - Validates config files
   - Creates required directories

### 📖 Documentation

**`experiments/README.md`** - Comprehensive guide covering:
- Dataset descriptions
- Quick start guide
- Configuration options
- Expected results
- Cost estimation
- Troubleshooting

---

## Configuration Features

### ✅ All Configs Include

- **Auto-download**: Datasets download automatically on first run
- **Caching**: LLM response caching enabled (saves money)
- **Three modes**: naive, local, multihop
- **Metrics**: Exact Match and Token F1
- **Sample control**: Start with 50 samples, scale up

### 📊 Query Modes Tested

Each benchmark evaluates three retrieval strategies:

1. **Naive** - Baseline vector search
2. **Local** - GraphRAG local entity retrieval
3. **MultiHop** - Iterative decomposition with entity carry-over

### ⚙️ Multi-Hop Parameters

```yaml
query:
  param_overrides:
    max_hops: 4              # Decomposition depth
    entities_per_hop: 10     # Entities per hop
    context_token_budget: 8000  # Context limit
    top_k: 60               # Retrieval count
```

---

## How to Use

### Quick Test (15-30 minutes)

```bash
cd /Users/minghao/Desktop/personal/nano-graphrag

# Test with 50 samples per dataset
./experiments/run_all_benchmarks.sh --quick

# View results
python experiments/compare_results.py
```

### Full Benchmark (2-4 hours)

```bash
# Run all datasets with full samples
./experiments/run_all_benchmarks.sh

# This will:
# - Download datasets (first time only)
# - Index ~25,000 documents
# - Run ~25,000 queries (with caching)
# - Save results to ./results/
```

### Single Dataset

```bash
# Test just one dataset
python -m bench.run --config experiments/benchmark_multihop_rag.yaml
```

---

## Validation Results

```
✅ PASS     Dependencies (pyyaml, datasets, nano-graphrag)
✅ PASS     Directories (experiments, results, workdirs, cache)
✅ PASS     Configs (all 4 configs validated)
✅ PASS     API Keys (optional, will use environment)
```

**Environment is ready!**

---

## Expected Output

### Results Location

```
./results/
├── multihop_rag_benchmark_*.json
├── musique_benchmark_*.json
├── hotpotqa_benchmark_*.json
└── 2wiki_benchmark_*.json
```

### Result Format

Each result file contains:
- `mode_results`: Metrics for naive, local, multihop
- `predictions`: Individual question predictions
- `duration_seconds`: Total runtime
- `cache_stats`: Cache hit rate

### Comparison Table

```
MULTI-HOP RAG BENCHMARK RESULTS
────────────────────────────────────────────────────────────────────────────────

📊 MULTIHOP-RAG
────────────────────────────────────────────────────────────────────────────────

Mode            Exact Match     Token F1         Delta vs Naive
────────────────────────────────────────────────────────────────────────────────
naive           0.350           0.400           baseline
local           0.420           0.480           +0.080 ✓
multihop        0.450           0.520           +0.120 ✓

⏱️  Duration: 0h 15m 30s
💾 Cache hit rate: 85.2%
```

---

## Cost Estimation

### Quick Test (50 samples × 4 datasets)
- **First run**: ~$2.00 (no cache)
- **Cached run**: ~$0.20 (90% hit rate)

### Full Benchmark (~25,000 samples)
- **First run**: ~$25-50 (no cache)
- **Cached run**: ~$5-10 (80-90% hit rate)

**Note**: Costs are for GPT-4o-mini. Adjust if using other models.

---

## Roadmap Targets

Based on `docs/technical_roadmap.md`:

| Dataset | Naive Baseline | Target Multihop | Stretch Target |
|---------|----------------|-----------------|----------------|
| MultiHop-RAG | ~0.40 F1 | 0.57+ | 0.64+ |
| MuSiQue | ~0.25 F1 | 0.42+ | 0.50+ |
| HotpotQA | ~0.45 F1 | 0.62+ | 0.68+ |
| 2WikiMHQA | ~0.38 F1 | 0.55+ | 0.62+ |

After running benchmarks, update `docs/verification/phase3_verification_report.md` with actual results.

---

## Customization

### Change Sample Size

Edit any config file:

```yaml
dataset:
  max_samples: 50   # Quick test
  # max_samples: -1  # Full dataset (all samples)
```

### Test Specific Modes

```yaml
query:
  modes:
    - local      # Only test local + multihop
    - multihop
```

### Adjust Multi-Hop Parameters

```yaml
query:
  param_overrides:
    max_hops: 6              # Deeper decomposition
    entities_per_hop: 15     # More entities per hop
    context_token_budget: 12000  # More context
```

---

## Troubleshooting

### Dataset Download Issues

```bash
# Manual download (if auto-download fails)
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('TIGER-Lab/MultiHop-RAG')
ds['validation'].to_json('~/.cache/nano-bench/multihop_rag/questions.json')
"
```

### Out of Memory

```yaml
graphrag:
  chunk_token_size: 800  # Reduce from 1200
```

### Slow Performance

1. Enable caching (already enabled)
2. Reduce sample size
3. Use cheaper model

---

## Next Steps

1. **Run quick test**: `./experiments/run_all_benchmarks.sh --quick`
2. **Review results**: `python experiments/compare_results.py`
3. **Run full benchmark** (if quick test succeeds): `./experiments/run_all_benchmarks.sh`
4. **Update verification report** with actual F1 scores
5. **Compare against roadmap targets**

---

## Files Created

```
experiments/
├── README.md                        # Comprehensive guide
├── benchmark_multihop_rag.yaml     # MultiHop-RAG config
├── benchmark_musique.yaml           # MuSiQue config
├── benchmark_hotpotqa.yaml          # HotpotQA config
├── benchmark_2wiki.yaml             # 2WikiMHQA config
├── run_all_benchmarks.sh            # Run all benchmarks
├── compare_results.py               # Compare results
└── validate_setup.py                # Validate setup
```

---

## Summary

✅ **4 benchmark configurations created**
✅ **All configurations validated**
✅ **Utility scripts ready**
✅ **Documentation complete**
✅ **Environment validated**

**Ready to run benchmarks!** 🚀

---

*Created: 2026-03-24*
