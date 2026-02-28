# LLM_RAG (Pittsburgh/CMU QA)

This repo implements an end-to-end RAG pipeline:
- crawling and extraction
- chunking
- sparse/dense/hybrid retrieval
- reader generation
- experiment/analysis scripts

The current experiment defaults are based on:
- Reader: `Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4`
- Dense embedder: `Alibaba-NLP/gte-Qwen2-7B-instruct` (8-bit) or MiniLM variants
- Sparse retriever: BM25 with weighted fields + PRF
- Hybrid fusion: RRF or min-max

## 1) Environment Setup

### Requirements
- Linux with CUDA (recommended).  
- Python `>=3.13` (per `pyproject.toml`).
- [`uv`](https://docs.astral.sh/uv/) installed.
- Hugging Face access for model downloads.

### Install dependencies
```bash
cd /path/to/LLM_RAG
uv sync
```

If needed:
```bash
source .venv/bin/activate
```

## 2) Data Pipeline (Optional if data already prepared)

### Step A: Crawl
```bash
uv run src/crawl/crawl.py --config configs/crawl_config.yaml
```

Optional: crawl selected domains only
```bash
uv run src/crawl/crawl.py \
  --config configs/crawl_config.yaml \
  --domains en.wikipedia.org,www.cmu.edu,www.visitpittsburgh.com
```

### Step B: Extract text
```bash
uv run src/crawl/extract.py \
  --raw_dir data/raw \
  --manifest data/raw/manifest.jsonl \
  --baseline_dir data/raw/baseline_data \
  --out data/processed/docs.jsonl \
  --overwrite
```

### Step C: Chunk documents
```bash
uv run src/crawl/chunk.py \
  --in_docs data/processed/docs.jsonl \
  --out_chunks data/processed/chunks.jsonl \
  --overwrite
```

## 3) Build Retrieval Indexes (Optional if indexes already exist)

### BM25 sparse index
```bash
PYTHONPATH=src/process uv run python -c "
from pathlib import Path
from sparse_retrieval import SparseRetriever
r = SparseRetriever(
    index_dir=Path('indexes/bm25_v3'),
    chunks_path=Path('data/processed/chunks.jsonl'),
    title_weight=3, heading_weight=2, body_weight=1,
    add_bigrams=True, enable_prf=True, prf_k=8, prf_terms=6, prf_alpha=0.65
)
r.build()
print('Built sparse index at indexes/bm25_v3')
"
```

### Dense FAISS index (GTE-Qwen 7B, 8-bit)
```bash
PYTHONPATH=src/process uv run python -c "
from pathlib import Path
from dense_retrieval import DenseRetriever
r = DenseRetriever(
    index_dir=Path('indexes/dense_gte-Qwen2-7B-instruct_v2'),
    chunks_path=Path('data/processed/chunks.jsonl'),
    model_name='Alibaba-NLP/gte-Qwen2-7B-instruct',
    device='cuda:0',
    batch_size=8,
    normalize=True,
    quant_backend='8bit'
)
r.build(save_embeddings=False)
print('Built dense index at indexes/dense_gte-Qwen2-7B-instruct_v2')
"
```

## 4) Run RAG on Leaderboard Queries

All experiment scripts are in `/scripts`.

### Main hybrid run (RRF)
```bash
bash scripts/run_pipeline_exp4_hybrid_rrf.sh
```

### Other ablations
```bash
bash scripts/run_pipeline_exp1_closed_book.sh
bash scripts/run_pipeline_exp2_dense_only.sh
bash scripts/run_pipeline_exp3_sparse_only.sh
bash scripts/run_pipeline_exp5_hybrid_minmax.sh
bash scripts/run_pipeline_exp6_hybrid_rrf_no_mmr.sh
bash scripts/run_pipeline_exp7_dense_only_minilm.sh
bash scripts/run_pipeline_exp8_hybrid_minilm.sh
```

Outputs are written to `data/answers/*.json`.

## 5) Run RAG on Released Test Set (.txt questions)

Use the helper script:
```bash
bash scripts/run_rag_on_testset_txt.sh \
  "/Users/samchen/Downloads/test_set_day_2(1).txt" \
  "data/answers/test_set_day_2_answers.json"
```

You can pass extra `query_ppl.py` args after those two positional args, e.g.:
```bash
bash scripts/run_rag_on_testset_txt.sh \
  "/Users/samchen/Downloads/test_set_day_2(1).txt" \
  "data/answers/test_set_day_2_answers.json" \
  --reader_device cuda:1 --device cuda:0
```

Output format:
```json
{
  "1": "Answer 1",
  "2": "Answer 2; Answer 3"
}
```

## 6) Retrieval Analysis and Summaries

### Minimal analysis run (Exp A/B + Exp D prep)
```bash
bash scripts/run_analysis_minimal.sh
```

### Build summary tables/plots
```bash
bash scripts/run_analysis_summary.sh
```

Artifacts:
- `analysis/exp_ab/`
- `analysis/exp_ab_exp8_minilm/`
- `analysis/exp_d/`
- `analysis/summary_report/`

## 7) Key Entry Points

- Retrieval + generation pipeline: `src/process/query_ppl.py`
- Sparse retriever: `src/process/sparse_retrieval.py`
- Dense retriever: `src/process/dense_retrieval.py`
- Hybrid retriever: `src/process/hybrid_retrieval.py`
- Reader: `src/process/reader.py`
- Crawl/extract/chunk: `src/crawl/`
- Analysis scripts: `src/analysis/`

## 8) Notes

- If using 2 GPUs, set dense retriever on one GPU and reader on another via `--device` and `--reader_device`.
- On single GPU, reduce memory pressure with smaller `k_ctx`, lower retrieval depth, or lighter embedder/index.
- Keep index/model pair consistent (`dense_dir` must match `embed_model` used to build it).
