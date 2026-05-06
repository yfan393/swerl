# SWE-RL: Software Engineering with Reinforcement Learning

**A unified implementation of reinforcement learning for automated code repair, with integrated support for multiple model backends including fine-tuned models and open-source Llama 3.1.**

![Status](https://img.shields.io/badge/status-active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

This is a **unified, production-ready implementation** of SWE-RL (Software Engineering with Reinforcement Learning) featuring:

- **🎯 Reinforcement Learning**: GRPO training on code repair tasks
- **📝 Supervised Fine-tuning**: SFT baseline with oracle patches  
- **🔍 Retrieval-Augmented Generation**: FAISS-based semantic code search
- **🦙 Multi-Model Support**: Fine-tuned models, OpenAI APIs, AND open-source Llama 3.1 (1B, 3B)
- **📊 Comprehensive Evaluation**: Code repair accuracy on SWE-bench
- **⚡ Production Ready**: Easy CLI, environment variable support, automatic retry logic

### What is SWE-RL?

SWE-RL is a reinforcement learning approach to code repair that:
1. **Retrieves relevant code** using semantic search (RAG)
2. **Generates repairs** as structured SEARCH/REPLACE edits
3. **Learns from rewards** combining format correctness, similarity, and correctness
4. **Iteratively improves** through policy gradient training (GRPO)

This implementation supports multiple backends so you can use proprietary (OpenAI) or open-source (Llama) models interchangeably.

---

## Quick Start (Choose Your Path)

### Path 1: Llama Models (Recommended for Open-Source)

```bash
# 1. Install
pip install -r requirements.txt
pip install vllm  # For Llama inference

# 2. Start vLLM server
export port_vllm=8000
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-3B-Instruct \
    --port 8000 &

# 3. Run inference (in another terminal)
python run.py llama_infer --port 8000 \
    --model llama_3b \
    --output_dir outputs/llama_eval \
    --num_samples 100

# 4. Evaluate
python run.py eval --mode submission \
    --raw_output_dir outputs/llama_eval

# 5. Compare Llama 1B vs 3B
python run.py llama_compare --port 8000 --num_instances 50
```

### Path 3: Compare Model Outputs on First Instance

Save and analyze the first test instance across **Trained Model**, **Llama 1B**, and **Llama 3B** to see side-by-side thinking blocks and patches:

```bash
# After running inference with all three models above
python run.py instance_compare \
  --trained_dir outputs/eval_student \
  --llama_1b_dir outputs/llama_1b_eval \
  --llama_3b_dir outputs/llama_3b_eval \
  --output_dir evaluation/instance_comparison
```

Generates:
- `instance_comparison.json` — Structured comparison data
- `instance_comparison.md` — Formatted for reports and presentations

**What you get:**
- ✓ Problem statement (the bug description)
- ✓ Thinking blocks (`<think>...</think>`) showing model reasoning
- ✓ Solution patches (`<solution>...</solution>`) with SEARCH/REPLACE edits
- ✓ Reward scores (similarity, correctness, combined)
- ✓ Format validity checks

Perfect for qualitative analysis in your research report!

### Path 2: Fine-Tuned Models (GRPO / SFT)

```bash
# 1. Prepare data
python run.py data --config configs/data_config.yaml

# 2. Train SFT baseline or GRPO
python run.py sft_train --config configs/train_config.yaml

# 3. Run inference
python run.py infer --model_path outputs/sft/final \
    --output_dir outputs/eval

# 4. Evaluate
python run.py eval --mode submission --raw_output_dir outputs/eval
```

### Prerequisites

- **Python 3.10+**
- **CUDA-capable GPU** (for vLLM: 2GB+ for 1B model, 8GB+ for 3B)
- **~20 GB disk** for models and data
- Optional: **GitHub token** for higher API rate limits
- Optional: **OpenAI API key** for GPT models

### Installation

```bash
# 1. Clone and install dependencies
pip install -r requirements.txt

# 2. Set GitHub token for higher API rate limits (IMPORTANT for data filtering speed)
export GITHUB_TOKEN="ghp_your_token_here"

# Or on Windows PowerShell:
$env:GITHUB_TOKEN="ghp_your_token_here"

# Get a free GitHub token here: https://github.com/settings/tokens/new
# (Create a "Fine-grained personal access token" with minimal permissions)
```

#### Why GitHub Token is Important

Without a GitHub token, API rate limiting is ~60 requests/minute.  
With a token, it's ~5000 requests/minute (83x faster).

**Example impact:**
- Without token: 100 PRs takes ~1-2 hours
- With token: 100 PRs takes ~5-10 minutes

The token can also be passed via command line:
```bash
python -m data.filter_prs --input_file data/raw/raw_prs.jsonl --token ghp_xxx --max_records 100
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

### Multi-Model Backend Architecture

The system is designed to work seamlessly with multiple model sources:

```
┌─────────────────────────────────────────────────────┐
│           Unified Interface (run.py)                 │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐   │
│  │  Fine-tuned  │  │   OpenAI     │  │  Llama   │   │
│  │   Models     │  │    API       │  │  (vLLM)  │   │
│  │  (GRPO/SFT)  │  │  (GPT-4)     │  │  1B/3B   │   │
│  └──────────────┘  └──────────────┘  └──────────┘   │
│        │                  │                  │       │
├────────┴──────────────────┴──────────────────┴────────┤
│              RAG Context Builder                      │
│         (retriever + prompt formatting)              │
├─────────────────────────────────────────────────────┤
│         SEARCH/REPLACE Patch Parser                  │
│         & Reward Computation                        │
├─────────────────────────────────────────────────────┤
│      Evaluation Pipeline (SWE-bench metrics)        │
└─────────────────────────────────────────────────────┘
```

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
- `data/filter_prs.py` — Filters PRs by size, language, and quality metrics (respects --token argument)
- `data/extract_triples.py` — Extracts issue descriptions and code changes
- `data/build_rag_index.py` — Builds FAISS index for semantic search

#### 2. SFT Data Generation

Converts training triples into supervised fine-tuning examples with chain-of-thought reasoning.

**Two Modes:**

| Mode | Cost | Quality | Use Case |
|------|------|---------|----------|
| **Oracle** (default) | Free | ⭐⭐⭐⭐ | Quick testing, no API needed |
| **API-based** | $ | ⭐⭐⭐⭐⭐ | Production, best quality |

**Oracle Mode (Recommended):**
```bash
python run.py sft_data --config configs/train_config_student.yaml
```
Extracts thinking + solution from oracle patches. Free, no API calls, takes ~2-3 minutes.

**API Mode (Optional):**
```yaml
sft_baseline:
  teacher:
    mode: "api"
    model: "llama-3.3-70b-versatile"  # Groq, Together AI, or OpenAI
    max_records: 500
```
Calls an LLM API to generate patches. Produces higher-quality training data but costs money.

**Key modules:**
- `sft/generate_cot_data.py` — Generates CoT training examples
- `sft/sft_train.py` — Trains model on generated data

#### 3. Reinforcement Learning (GRPO)

Trains model to generate code repairs using reward signals.

**Training stages:**
1. **SFT Baseline** — Initialize model with supervised learning
2. **GRPO Policy** — Optimize using reward feedback (format + correctness + similarity)

```bash
# SFT training
python run.py sft_train --config configs/train_config_student.yaml

# GRPO training
python run.py train --config configs/train_config_student.yaml
```

---

## Code Quality & Error Handling

This implementation includes comprehensive error handling and fixes for common issues:

**✅ Fixed Issues (May 2026):**
- Config path mismatches (reads from correct YAML sections)
- Type errors in dataset handling (proper HuggingFace Dataset conversion)
- Silent import failures (raises clear errors with install instructions)
- Missing RAG index error messages (helpful error messages with recovery steps)
- Missing directory creation (auto-creates output directories)
- Import path assumptions (works from any directory)
- Loose version constraints (pinned to prevent breaking changes)

See [COMPREHENSIVE_CODE_AUDIT.md](COMPREHENSIVE_CODE_AUDIT.md) for detailed audit of 13 identified and fixed issues.

---

## Common Issues & Solutions

### Data Filtering is Slow

**Problem:** Only getting 2-3 filtered PRs per hour  
**Cause:** GitHub API rate limiting (60 req/min without token)  
**Solution:** Set GitHub token (see Installation section above)

```bash
# Speed increases from ~1 hour to ~5 minutes for 100 PRs
python -m data.filter_prs \
    --input_file data/raw/raw_prs.jsonl \
    --token ghp_xxx \
    --max_records 100
```

### Out of Memory During Training

**Problem:** CUDA out of memory error  
**Solutions** (in order):
1. Reduce `batch_size` in config (try 1-2)
2. Reduce `max_new_tokens` (try 64-128)
3. Reduce `num_rollouts` in GRPO config
4. Use smaller base model (Qwen 1.5B instead of Llama 3B)

### Model Output Format is Invalid

**Problem:** Model generates output that doesn't parse as SEARCH/REPLACE  
**Cause:** Model wasn't fine-tuned or format wasn't in training data  
**Solution:** Ensure SFT baseline training completed first (Part 3)

### Ray or Timeout Errors

**Problem:** "Ray has shut down" or timeouts during training  
**Solution:** Reduce `num_workers` in config or run with simpler settings

---

## Documentation & Research