# SWE-RL Re-implementation: Code Repair at Small Scale

## Overview

This repository is a **student-friendly re-implementation** of the SWE-RL (Software Engineering with Reinforcement Learning) framework from the original Facebook Research paper. It provides a complete pipeline for training models to automatically repair code bugs using Reinforcement Learning.

Unlike the original paper which operates at massive scale, this implementation is designed to run on modest hardware (laptops, free cloud notebooks, or small GPU clusters) while preserving the core algorithmic ideas. The framework emphasizes **free and open tools**: local RAG retrieval, oracle-based supervision, and no dependency on paid APIs.

### What is SWE-RL?

SWE-RL is a reinforcement learning approach to code repair that:
1. **Retrieves relevant code** using semantic search (RAG)
2. **Generates repairs** as structured SEARCH/REPLACE edits
3. **Learns from rewards** that combine format correctness, patch similarity, and successful code execution
4. **Iteratively improves** through policy gradient training (GRPO)

This re-implementation captures these core ideas while making them accessible for research and coursework.

---

## Quick Start (5 minutes)

### Prerequisites

- **Python 3.10+**
- **CUDA-capable GPU** (strongly recommended; CPU-only is very slow)
- **~10-20 GB disk** for downloaded models and data
- Optional: **GitHub token** for higher API rate limits

### Installation

```bash
# Clone or navigate to the repo
cd swe_rl_reimplement

# Create and activate a virtual environment
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Environment Variables (Optional)

```bash
# Set your GitHub token for higher API rate limits
export GITHUB_TOKEN="your_free_github_token"

# Or on Windows PowerShell:
$env:GITHUB_TOKEN="your_free_github_token"
```

### Run a Complete Pipeline in Minutes

```bash
# 1. Generate tiny dataset (takes ~2 minutes)
python run.py data --config configs/data_config_student.yaml

# 2. Generate SFT supervision data (free oracle mode)
python run.py sft_data --config configs/train_config_student.yaml

# 3. Train SFT baseline (5-10 minutes on GPU)
python run.py sft_train --config configs/train_config_student.yaml

# 4. Train GRPO policy (10-20 minutes on GPU)
python run.py train --config configs/train_config_student.yaml

# 5. Run inference and generate results
python run.py infer \
  --model_path outputs/grpo_student/final \
  --config configs/train_config_student.yaml \
  --batch_size 1 \
  --max_new_tokens 128

# 6. Generate submission file
python run.py eval --mode submission --raw_output_dir outputs/eval_student
```

**Total time on a modest GPU: ~30-45 minutes**

---

## Architecture Overview

### Pipeline Stages

The SWE-RL re-implementation consists of three main stages:

#### 1. Data Pipeline

Transforms raw GitHub data into structured training triples.

```
GitHub/GHArchive
    ↓
[Fetch] Download pull request events
    ↓
[Filter] Keep high-quality bug-fix PRs (size, language, issue quality)
    ↓
[Extract] Build (issue, code_context, oracle_patch) triples
    ↓
[Index] Create FAISS retrieval index for RAG
    ↓
Train/Val splits + RAG index ready
```

**Key modules:**
- `data/fetch_gharchive.py` — Downloads GHArchive pull request events
- `data/filter_prs.py` — Filters PRs by size, language, and quality metrics
- `data/extract_triples.py` — Extracts issue descriptions and code changes
- `data/build_rag_index.py` — Builds FAISS index for semantic search
- `data/preprocess_pipeline.py` — Orchestrates all stages

#### 2. Training Pipeline

Trains two models: a Supervised Fine-Tuning (SFT) baseline and a GRPO policy.

```
Training Triples
    ↓
[SFT Data Gen] Convert triples to SEARCH/REPLACE supervision
    ↓
[SFT Train] Teach model to format patches correctly (baseline)
    ↓
[GRPO Train] Learn policy using rewards + policy gradient
    ↓
Trained models in outputs/
```

**Key modules:**
- `sft/generate_cot_data.py` — Generates SFT training data (oracle or API mode)
- `sft/sft_train.py` — Fine-tunes models with LoRA/QLoRA
- `training/grpo_train.py` — Implements GRPO training loop
- `training/rollout_utils.py` — Rollout, reward, and advantage computation
- `reward/reward_fn.py` — Combined reward (format + correctness + similarity)

#### 3. Evaluation Pipeline

Runs models on SWE-bench and generates submission-ready outputs.

```
SWE-bench Verified Test Set
    ↓
[Inference] Generate patches for each issue
    ↓
[Post-process] Format as unified diffs
    ↓
[Evaluation] Compute metrics (format accuracy, reward, etc.)
    ↓
Submission file + analysis
```

**Key modules:**
- `evaluation/run_inference.py` — Runs inference on SWE-bench with RAG retrieval
- `evaluation/evaluate.py` — Post-processing and metric computation
- `agent/prompts.py` — System/user prompts for the repair task
- `agent/retriever.py` — FAISS-backed code retrieval
- `agent/rag_context_builder.py` — Formats retrieved code into context

---

## Configuration System

Configurations are YAML files that control every aspect of the pipeline. Three tiers are provided:

### Config Tiers

1. **Student Configs** (`configs/*_student.yaml`)
   - Tiny dataset (5-10 PRs)
   - Small models (0.5-3B parameters)
   - Minimal training steps
   - **Perfect for laptops and free cloud notebooks**
   - Runtime: ~30-45 minutes total

2. **Free-tier Configs** (`configs/*_free.yaml`)
   - Medium dataset (100-300 PRs)
   - Moderate models (7-13B parameters)
   - More training steps
   - **Good for free Colab/Kaggle or small GPUs**
   - Runtime: ~2-4 hours total

3. **Full Configs** (`configs/data_config.yaml`, `configs/train_config.yaml`)
   - Larger dataset (1000+ PRs)
   - Bigger models (13B+ parameters)
   - Full training schedule
   - **Requires HPC cluster or paid cloud compute**
   - Runtime: 6-12+ hours total

### Configuration Structure

Each training config (e.g., `train_config_student.yaml`) has these sections:

```yaml
paths:
  train_file: "data/processed/train.jsonl"          # Training data location
  val_file: "data/processed/val.jsonl"              # Validation data

model:
  name_or_path: "Qwen/Qwen2.5-Coder-0.5B-Instruct" # Base model
  torch_dtype: "bfloat16"                           # Precision

sft_baseline:
  teacher:
    mode: "oracle"                                  # oracle or api
    max_records: 50                                 # Limit for testing

training:
  output_dir: "outputs/grpo_student"
  num_train_epochs: 3
  max_steps: 500                                    # Total training steps
  per_device_train_batch_size: 4

grpo:
  num_rollouts: 2                                   # Rollouts per sample
  num_mini_batches: 2

evaluation:
  output_dir: "outputs/eval_student"
  batch_size: 4
  num_repair_samples: 100                           # Eval set size

rag:
  embed_model: "sentence-transformers/all-MiniLM-L6-v2"
  top_k: 8                                          # Chunks to retrieve
```

### Important Config Parameters

| Parameter | Effect | Notes |
|-----------|--------|-------|
| `max_steps` | Total training iterations | Smaller = faster but less trained |
| `per_device_train_batch_size` | Batch size per GPU | Reduce if OOM errors |
| `num_rollouts` | GRPO rollouts per sample | Higher = more diverse rewards |
| `top_k` (RAG) | Retrieved code chunks | More = better context but slower |
| `num_repair_samples` | Eval set size | Smaller = faster validation |

---

## Reward Function

The re-implementation uses a **combined reward** that balances multiple objectives:

```
R(output) = {
    -1.0                                              if format_error
    alpha * correctness(output) + 
    (1 - alpha) * similarity(output)                 otherwise
}
```

### Reward Components

1. **Format Correctness** (binary: -1 or continues)
   - Model output must parse as valid SEARCH/REPLACE edits
   - Returns -1 if format is invalid

2. **Patch Correctness** (in [0, 1])
   - Does patch apply cleanly without conflicts?
   - Does resulting code have valid Python syntax?
   - Are there no new linting errors? (flake8)

3. **Similarity Reward** (in [0, 1])
   - Sequence similarity (jaccard) between predicted and ground-truth patch
   - Shared-word-based normalization

### Default Weighting

- `alpha = 0.3` (in `configs/train_config.yaml`)
  - Emphasizes **similarity** to oracle (70%)
  - Rewards **correctness** as tiebreaker (30%)

This weighting helps models learn the structure of correct patches while still rewarding novel solutions that achieve the same goal.

---

## Repository Structure

```
swe_rl_reimplement/
├── README.md                           # This file
├── requirements.txt                    # Dependencies
├── setup.py                            # Package metadata
├── run.py                              # Main CLI entry point
│
├── configs/                            # Configuration files
│   ├── data_config_student.yaml        # Tiny data config
│   ├── data_config_free.yaml           # Free-tier data config
│   ├── data_config.yaml                # Full data config
│   ├── train_config_student.yaml       # Tiny training config
│   ├── train_config_free.yaml          # Free-tier training config
│   └── train_config.yaml               # Full training config
│
├── data/                               # Data pipeline
│   ├── fetch_gharchive.py              # Download PR events
│   ├── filter_prs.py                   # Quality filtering
│   ├── extract_triples.py              # Extract issue/code/patch
│   ├── build_rag_index.py              # Build FAISS index
│   ├── preprocess_pipeline.py          # Orchestration
│   └── (generated)
│       ├── raw/                        # Downloaded GHArchive files
│       ├── processed/                  # Train/val triples
│       └── rag/                        # FAISS index + chunks
│
├── agent/                              # RAG agent
│   ├── prompts.py                      # Repair prompt templates
│   ├── retriever.py                    # FAISS retriever
│   └── rag_context_builder.py          # Context formatting
│
├── sft/                                # SFT training
│   ├── generate_cot_data.py            # Data generation
│   └── sft_train.py                    # Training loop
│
├── training/                           # GRPO training
│   ├── grpo_train.py                   # GRPO algorithm
│   ├── dataset.py                      # GRPO dataset
│   └── rollout_utils.py                # Rollouts + rewards
│
├── reward/                             # Reward computation
│   └── reward_fn.py                    # Combined reward
│
├── evaluation/                         # Evaluation
│   ├── run_inference.py                # Inference loop
│   └── evaluate.py                     # Post-processing
│
├── utils/                              # Utilities
│   ├── api_client.py                   # OpenAI-compatible API
│   ├── git_utils.py                    # Git operations
│   ├── token_counter.py                # Token counting
│   └── io_utils.py                     # File I/O
│
├── workflow_pipeline_notebook.ipynb    # Complete walkthrough
└── outputs/                            # Generated artifacts
    ├── grpo_student/final              # Trained model
    ├── eval_student/                   # Evaluation results
    └── notebook_figs/                  # Visualizations
```

---

## Common Commands

### Data Pipeline (Standalone Stages)

```bash
# Full pipeline (recommended first)
python run.py data --config configs/data_config_student.yaml

# Or run stages individually for debugging:
python run.py data --stage fetch --config configs/data_config_student.yaml
python run.py data --stage filter --config configs/data_config_student.yaml
python run.py data --stage extract --config configs/data_config_student.yaml
python run.py data --stage index --config configs/data_config_student.yaml
```

### Training

```bash
# Generate SFT data (oracle mode, free)
python run.py sft_data --config configs/train_config_student.yaml

# Train SFT baseline
python run.py sft_train --config configs/train_config_student.yaml

# Train GRPO policy
python run.py train --config configs/train_config_student.yaml
```

### Inference

```bash
# Standard inference
python run.py infer \
  --model_path outputs/grpo_student/final \
  --config configs/train_config_student.yaml

# Low-VRAM mode (batch size 1, fewer tokens)
python run.py infer \
  --model_path outputs/grpo_student/final \
  --config configs/train_config_student.yaml \
  --batch_size 1 \
  --max_new_tokens 128

# Custom parameters
python run.py infer \
  --model_path outputs/grpo_student/final \
  --output_dir outputs/eval_custom \
  --num_samples 200 \
  --temperature 0.7 \
  --top_k_chunks 16
```

### Evaluation

```bash
# Generate submission file
python run.py eval --mode submission --raw_output_dir outputs/eval_student

# Compute offline validation reward
python run.py eval \
  --mode val_reward \
  --raw_output_dir outputs/eval_student \
  --val_file data/processed/val.jsonl
```

---

## Hardware Recommendations

### Minimum (Laptop / Free Cloud)
- **GPU**: Any NVIDIA CUDA-capable (2GB+ VRAM)
- **CPU**: Quad-core or better
- **RAM**: 8-16 GB
- **Config**: Use `*_student.yaml` configs
- **Time**: 30-45 minutes for full pipeline

### Recommended (Small GPU / Mid-tier Cluster)
- **GPU**: RTX 3080+ or A10 (24GB VRAM)
- **CPU**: 8-16 cores
- **RAM**: 32-64 GB
- **Config**: Use `*_free.yaml` configs
- **Time**: 2-4 hours for full pipeline

### Full Scale (HPC Cluster)
- **GPU**: 8x A100 or equivalent
- **CPU**: High-core-count workstation
- **RAM**: 256+ GB
- **Config**: Use full `*.yaml` configs with `accelerate launch`
- **Time**: 6-12+ hours with multi-GPU distributed training

---

## Troubleshooting

### CUDA Out of Memory (OOM)

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions (in order of impact):**
1. Reduce batch size: `--batch_size 1`
2. Reduce max tokens: `--max_new_tokens 64`
3. Use student config: `*_student.yaml`
4. Use 8-bit quantization in training config: `load_in_8bit: true`

### Data Pipeline Errors

**Symptom**: "No data found after filtering" or empty train/val files

**Causes and fixes:**
- GitHub API rate limit hit → Set `GITHUB_TOKEN` env var
- Network timeout → Increase `fetch_timeout` in config
- Too strict filters → Check `filter_prs.py` thresholds

### Inference Hangs or Crashes

**Symptom**: Inference starts but never completes

**Solutions:**
1. Check GPU memory: `nvidia-smi`
2. Enable auto-fallback in config: `auto_retry_batch_size: true`
3. Reduce `top_k_chunks` in RAG config
4. Run with `--batch_size 1 --max_new_tokens 128` (very conservative)

### Import Errors

**Symptom**: `ModuleNotFoundError: No module named 'data'`

**Fix:**
```bash
# Install package in editable mode
pip install -e .
```

### Validation Data Missing

**Symptom**: `FileNotFoundError: data/processed/val.jsonl`

**Fix:**
```bash
# Re-run the full data pipeline
python run.py data --config configs/data_config_student.yaml
```

---

## Output Files and Artifacts

All generated artifacts are created in `outputs/` and `data/` directories. Key files:

### Data Artifacts
- `data/raw/raw_prs.jsonl` — Initial PR events from GHArchive
- `data/raw/filtered_prs.jsonl` — PRs after quality filtering
- `data/processed/train.jsonl` — Training triples (issue, code, patch)
- `data/processed/val.jsonl` — Validation triples
- `data/processed/sft_cot_data.jsonl` — SFT training data
- `data/rag/faiss.index` — FAISS embedding index
- `data/rag/chunks.jsonl` — Code chunks + metadata

### Training Artifacts
- `outputs/sft_student/final/` — Trained SFT model (checkpoint)
- `outputs/grpo_student/final/` — Trained GRPO model (checkpoint)
- `outputs/grpo_student/runs/` — Training logs (per epoch)

### Evaluation Artifacts
- `outputs/eval_student/raw_outputs.jsonl` — Model predictions (raw)
- `outputs/eval_student/all_preds.jsonl` — Unified diff format
- `outputs/eval_student/format_report.json` — Format accuracy metrics
- `outputs/eval_student/val_reward_combined_report.json` — Reward metrics

---

## Data Sources

### Raw Data
- **GHArchive**: Public GitHub event logs ([data.gharchive.org](https://data.gharchive.org))
- **GitHub API**: PR metadata, diffs, issue details
- **SWE-bench**: Evaluation set ([HuggingFace](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified))

### Models
- **Base LLM**: Qwen/Qwen2.5-Coder (HuggingFace Hub)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (local)
- **Retrieval**: FAISS (local, CPU or GPU)

---

## Free vs. Paid Approach

### Completely Free Path (Recommended for Students)

✅ **What's Free:**
- GHArchive data (public, no API key needed)
- Local RAG with FAISS and Sentence Transformers
- SFT and GRPO training
- SWE-bench evaluation dataset
- Local model inference

❌ **What Costs Money (not required):**
- OpenAI API for API-mode SFT data generation (we use oracle mode instead)
- Weights & Biases logging (we use local logging)
- Hosted vector databases like Pinecone (we use FAISS)

### Using Paid Tools (Optional)

If you want to experiment with API-mode SFT generation:

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="sk-..."

# In config, change:
# sft_baseline.teacher.mode: "api"  # instead of "oracle"
```

Or point to a self-hosted endpoint:
```bash
export OPENAI_BASE_URL="https://your-cluster:8000/v1"
```

---

## Citation and Acknowledgments

This re-implementation is based on:

**SWE-RL: Mastering Software Engineering with Reinforcement Learning** — Facebook Research (ICLR 2025)

Original code: https://github.com/facebookresearch/swe-rl

Key papers:
- SWE-RL: https://arxiv.org/abs/2501.07124
- SWE-Bench: https://arxiv.org/abs/2312.09393

---

## License and Contact

This re-implementation is provided as-is for educational purposes.

**Author**: Yuanting Fan (yfan393@gatech.edu)  
**Course**: CS 8803 (Georgia Tech)

For issues or questions, please:
1. Check the Troubleshooting section above
2. Review example configs in `configs/`
3. Check logs in `outputs/` for detailed error messages
4. Refer to original SWE-RL repo for algorithmic details

---

## FAQ

**Q: Can I run this on my laptop?**  
A: Yes! Use `*_student.yaml` configs. Expect 30-45 minutes with a GPU, or skip training and use a pre-trained model.

**Q: Do I need a GitHub token?**  
A: No, but it raises your GitHub API rate limit from 60 to 5000 requests/hour. Recommended for anything beyond student configs.

**Q: Can I use a different base model?**  
A: Yes! Change `model.name_or_path` in any config file. We've tested with Qwen, Llama, CodeLlama, and others.

**Q: Why oracle mode instead of API mode for SFT?**  
A: Oracle mode uses ground-truth patches directly from the dataset, requiring no API calls. It's free, deterministic, and teaches the same format.

**Q: How do I scale this to full SWE-RL?**  
A: Use the full `*.yaml` configs, increase `num_train_epochs` and `max_steps`, and run with `accelerate launch` for multi-GPU training.

**Q: What if inference is too slow?**  
A: Reduce `top_k_chunks`, `max_new_tokens`, or use a smaller base model. Batch inference with `batch_size > 1` if memory allows.

