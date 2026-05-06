# SWE-RL Small-Scale Re-implementation

This repository is a student-friendly re-implementation of a SWE-RL style code
repair pipeline. It includes:

- A GitHub/GHArchive data pipeline for building small bug-fix datasets.
- Local RAG with Sentence Transformers and FAISS.
- SFT and GRPO training entry points.
- Offline validation metrics and SWE-bench-style inference/post-processing.
- Student configs that avoid paid APIs and hosted vector databases.

The code is meant for small experiments and course-report evidence, not for
reproducing full paper-scale results.

## Quick Start(Free Path)

```

Create and activate a virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt`

Optional environment variables:

```powershell
$env:GITHUB_TOKEN="your_free_github_token"  # optional, raises GitHub API rate limit
$env:TOKENIZER_TYPE="tiktoken"              # fast approximate token counting
```

Run the tiny zero-cost data path:

```powershell
python run.py data --config configs/data_config_student.yaml
```

Generate free SFT data from oracle before/after files:

```powershell
python run.py sft_data --config configs/train_config_student.yaml
```

Run a tiny SFT or GRPO smoke test:

```powershell
python run.py sft_train --config configs/train_config_student.yaml
python run.py train --config configs/train_config_student.yaml
```

Run local inference from your trained model:

```powershell
python run.py infer --model_path outputs/grpo_student/final --config configs/train_config_student.yaml
```

Low-VRAM friendly inference override (recommended on small GPUs):

```powershell
python run.py infer --model_path outputs/grpo_student/final --config configs/train_config_student.yaml --batch_size 1 --max_new_tokens 128
```

Prepare submission predictions after inference:

```powershell
python run.py eval --mode submission --raw_output_dir outputs/eval_student
```

## Free-Only Operation (No Paid API Required)

The student configs are designed to run without paid software:

- `sft_baseline.teacher.mode: oracle` in `configs/train_config_student.yaml` uses oracle SEARCH/REPLACE targets and does not call OpenAI or other paid APIs.
- RAG is local (`sentence-transformers` + `faiss-cpu`) and does not require hosted vector databases.
- Logging is local by default (`report_to: none`), so W&B is optional.

`openai` in `requirements.txt` is optional and used only when you explicitly switch teacher mode to `api`. If you ever use API mode, you can still point `OPENAI_BASE_URL` to a self-hosted OpenAI-compatible endpoint (for example on your own cluster), instead of paid hosted services.

## Student Run Profiles

Use one of these depending on hardware:

- Laptop / low VRAM: keep `configs/*_student.yaml` defaults.
- Free cloud notebook tier: keep student configs and reduce `max_steps` / `num_repair_samples` if needed.
- HPC cluster: start from `configs/*_free.yaml` or `configs/*.yaml`, then run with `accelerate launch` for multi-GPU training.

You can also set `evaluation.batch_size` in your train config to control inference memory without changing code.

## Repository Map

Top-level:

- `run.py`: one CLI entry point for data, SFT data generation, SFT training, GRPO training, inference, and evaluation.
- `requirements.txt`: dependencies for local VS Code trials.
- `setup.py`: optional package install metadata.
- `README.md`: this guide.

Configs:

- `configs/data_config_student.yaml`: tiny no-cost data pipeline config.
- `configs/train_config_student.yaml`: tiny no-cost training config.
- `configs/data_config_free.yaml`: larger free-tier data config.
- `configs/train_config_free.yaml`: larger free-tier training config.
- `configs/data_config.yaml`: full/default data config.
- `configs/train_config.yaml`: full/default training config.

Data pipeline:

- `data/fetch_gharchive.py`: downloads GHArchive PullRequestEvent records.
- `data/filter_prs.py`: filters/enriches PRs through the GitHub API.
- `data/extract_triples.py`: builds issue/code/oracle-patch training triples.
- `data/build_rag_index.py`: chunks code and builds the FAISS index.
- `data/preprocess_pipeline.py`: orchestrates fetch, filter, extract, and index.

Agent/RAG:

- `agent/prompts.py`: repair and SFT prompt templates.
- `agent/retriever.py`: FAISS-backed code retriever.
- `agent/rag_context_builder.py`: formats retrieved chunks into model context.

Training:

- `training/dataset.py`: GRPO dataset wrapper.
- `training/grpo_train.py`: GRPO training script.
- `training/rollout_utils.py`: rollout and advantage helpers.
- `sft/generate_cot_data.py`: SFT data generation, including free oracle mode.
- `sft/sft_train.py`: LoRA/QLoRA SFT baseline training.

Reward and evaluation:

- `reward/reward_fn.py`: format, patch-application, similarity, and correctness rewards.
- `evaluation/run_inference.py`: builds SWE-bench-style repo context and generates outputs.
- `evaluation/evaluate.py`: offline validation and unified-diff submission generation.

Utilities:

- `utils/api_client.py`: OpenAI-compatible API helper for optional teacher mode.
- `utils/git_utils.py`: temporary git repo helpers for patch application/diffing.
- `utils/token_counter.py`: configurable token-counting helper.

## Typical Outputs

Generated artifacts are intentionally ignored by git:

- `data/raw/`
- `data/processed/`
- `data/rag/`
- `data/repos/`
- `data/eval_repos/`
- `outputs/`
- `playground/`

## Data Locations

Where data comes from:

- GHArchive public event files: [https://data.gharchive.org](https://data.gharchive.org)
- GitHub API metadata (optional higher rate limits with `GITHUB_TOKEN`)
- SWE-bench evaluation split from Hugging Face: `princeton-nlp/SWE-bench_Verified`

Where data is stored locally after running the pipeline:

- `data/raw/gharchive/`: downloaded GHArchive hourly JSON files
- `data/raw/raw_prs.jsonl`: initial PR events extracted from GHArchive
- `data/raw/filtered_prs.jsonl`: PRs that passed repository/issue/diff filters
- `data/processed/train.jsonl`, `data/processed/val.jsonl`: train/val triples used for training
- `data/processed/sft_cot_data.jsonl`: SFT training records (oracle or API mode)
- `data/rag/faiss.index`, `data/rag/chunks.jsonl`: local embedding index and chunk metadata
- `data/repos/`: cached repositories used during triple extraction
- `data/eval_repos/`: cached repositories used for SWE-bench-style evaluation context

## Notes For Trials

- Start with the student configs. The default configs are much heavier.
- First runs may download models and clone repositories, so they need network access.
- If a run fails, check the console logs first. The scripts log config paths, RAG
- Inference now auto-retries with smaller generation batch sizes on CUDA OOM. If it still fails at batch size 1, reduce `max_new_tokens`, use the student model config, or move the run to your cluster GPU.
