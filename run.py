"""
run.py — Top-level orchestrator for SWE-RL Re-implementation
=============================================================
Convenience script to run any stage of the pipeline from one place.

Usage:
    # Full data pipeline
    python run.py data --config configs/data_config.yaml

    # Single data stage
    python run.py data --stage fetch --config configs/data_config.yaml
    python run.py data --stage filter --config configs/data_config.yaml
    python run.py data --stage extract --config configs/data_config.yaml
    python run.py data --stage index --config configs/data_config.yaml

    # Generate SFT chain-of-thought data
    python run.py sft_data --config configs/train_config.yaml

    # SFT baseline training
    python run.py sft_train --config configs/train_config.yaml

    # GRPO (SWE-RL) training
    python run.py train --config configs/train_config.yaml

    # Run inference on SWE-bench Verified
    python run.py infer \
        --model_path outputs/grpo/final \
        --output_dir outputs/eval/grpo \
        --num_samples 500

    # Generate submission file
    python run.py eval \
        --mode submission \
        --raw_output_dir outputs/eval/grpo \
        --output_file outputs/eval/grpo/all_preds.jsonl

    # Offline validation reward
    python run.py eval \
        --mode val_reward \
        --raw_output_dir outputs/eval/grpo \
        --val_file data/processed/val.jsonl
"""

import argparse
from pathlib import Path

from utils.io_utils import read_jsonl, read_yaml


def cmd_data(args):
    from data.preprocess_pipeline import run_pipeline
    run_pipeline(config_path=args.config, stage=getattr(args, "stage", "all"))


def cmd_sft_data(args):
    import asyncio

    from sft.generate_cot_data import generate_all, generate_oracle_all

    cfg = read_yaml(args.config)
    sft_cfg = cfg.get("sft_baseline", {})
    teacher_cfg = sft_cfg.get("teacher", {})
    train_file = cfg["paths"]["train_file"]
    output_file = sft_cfg.get("train_file", "data/processed/sft_cot_data.jsonl")
    records = read_jsonl(train_file)
    max_records = teacher_cfg.get("max_records")
    if max_records:
        records = records[:max_records]

    mode = teacher_cfg.get("mode", "oracle")
    rag_cfg = cfg.get("rag", {})
    top_k_chunks = rag_cfg.get("top_k", 8)
    if mode == "oracle":
        generate_oracle_all(
            records=records,
            output_file=output_file,
            top_k_chunks=top_k_chunks,
            rag_index_path=rag_cfg.get("index_path", "data/rag/faiss.index"),
            rag_chunk_meta_path=rag_cfg.get("chunk_meta_path", "data/rag/chunks.jsonl"),
            rag_embed_model=rag_cfg.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2"),
        )
    elif mode == "api":
        asyncio.run(
            generate_all(
                records=records,
                output_file=output_file,
                model=teacher_cfg.get("model", "local-oracle"),
                max_concurrent=teacher_cfg.get("max_concurrent", 2),
                max_tokens=teacher_cfg.get("max_tokens", 2048),
                reward_threshold=teacher_cfg.get("reward_threshold", 0.5),
                top_k_chunks=top_k_chunks,
                rag_index_path=rag_cfg.get("index_path", "data/rag/faiss.index"),
                rag_chunk_meta_path=rag_cfg.get("chunk_meta_path", "data/rag/chunks.jsonl"),
                rag_embed_model=rag_cfg.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2"),
            )
        )
    else:
        raise ValueError("sft_baseline.teacher.mode must be 'oracle' or 'api'")


def cmd_sft_train(args):
    from sft.sft_train import train_sft
    train_sft(config_path=args.config)


def cmd_train(args):
    from training.grpo_train import train
    train(config_path=args.config)


def cmd_infer(args):
    from evaluation.run_inference import run_inference
    eval_cfg = {}
    if getattr(args, "config", None):
        cfg = read_yaml(args.config)
        eval_cfg = cfg.get("evaluation", {})

    output_dir = getattr(args, "output_dir", None) or eval_cfg.get("output_dir")
    if not output_dir:
        raise ValueError("Inference requires --output_dir or evaluation.output_dir in --config")

    run_inference(
        model_path=args.model_path,
        output_dir=output_dir,
        dataset_name=getattr(args, "dataset", None) or eval_cfg.get(
            "swe_bench_split",
            "princeton-nlp/SWE-bench_Verified",
        ),
        num_samples=getattr(args, "num_samples", None) or eval_cfg.get("num_repair_samples", 500),
        temperature=getattr(args, "temperature", None) or eval_cfg.get("eval_temperature", 1.0),
        max_new_tokens=getattr(args, "max_new_tokens", None) or eval_cfg.get("max_new_tokens", 2048),
        top_k_chunks=getattr(args, "top_k_chunks", None) or eval_cfg.get("top_k_chunks", 8),
        repo_cache_dir=getattr(args, "repo_cache_dir", None) or eval_cfg.get(
            "repo_cache_dir",
            "data/eval_repos",
        ),
        embed_model_name=getattr(args, "embed_model", None) or eval_cfg.get(
            "embed_model",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
        max_eval_files=getattr(args, "max_eval_files", None) or eval_cfg.get("max_eval_files", 300),
        batch_size=getattr(args, "batch_size", None) or eval_cfg.get("batch_size", 4),
    )


def cmd_eval(args):
    from evaluation.evaluate import generate_submission_file, offline_val_reward
    if not Path(args.raw_output_dir).exists():
        raise FileNotFoundError(f"raw_output_dir does not exist: {args.raw_output_dir}")
    if args.mode == "submission":
        generate_submission_file(
            raw_output_dir=args.raw_output_dir,
            output_file=getattr(args, "output_file", None)
                        or f"{args.raw_output_dir}/all_preds.jsonl",
            model_name=getattr(args, "model_name", "llama3-swerl-8b"),
        )
    elif args.mode == "val_reward":
        offline_val_reward(
            raw_output_dir=args.raw_output_dir,
            val_file=getattr(args, "val_file", "data/processed/val.jsonl"),
        )


def main():
    parser = argparse.ArgumentParser(
        description="SWE-RL Re-implementation Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_data = subparsers.add_parser("data", help="Run data preprocessing pipeline")
    p_data.add_argument("--config", default="configs/data_config.yaml")
    p_data.add_argument(
        "--stage",
        default="all",
        choices=["all", "fetch", "filter", "extract", "index"],
    )

    p_sft_data = subparsers.add_parser("sft_data", help="Generate SFT CoT training data")
    p_sft_data.add_argument("--config", default="configs/train_config.yaml")

    p_sft_train = subparsers.add_parser("sft_train", help="Train SFT baseline")
    p_sft_train.add_argument("--config", default="configs/train_config.yaml")

    p_train = subparsers.add_parser("train", help="Run GRPO (SWE-RL) training")
    p_train.add_argument("--config", default="configs/train_config.yaml")

    p_infer = subparsers.add_parser("infer", help="Run inference on SWE-bench Verified")
    p_infer.add_argument("--model_path", required=True)
    p_infer.add_argument("--config", default=None, help="Optional train config with evaluation section")
    p_infer.add_argument("--output_dir", default=None)
    p_infer.add_argument("--dataset", default=None)
    p_infer.add_argument("--num_samples", type=int, default=None)
    p_infer.add_argument("--temperature", type=float, default=None)
    p_infer.add_argument("--max_new_tokens", type=int, default=None)
    p_infer.add_argument("--top_k_chunks", type=int, default=None)
    p_infer.add_argument("--repo_cache_dir", default=None)
    p_infer.add_argument("--embed_model", default=None)
    p_infer.add_argument("--max_eval_files", type=int, default=None)
    p_infer.add_argument("--batch_size", type=int, default=None)

    p_eval = subparsers.add_parser("eval", help="Post-process and evaluate")
    p_eval.add_argument("--mode", choices=["submission", "val_reward"], default="submission")
    p_eval.add_argument("--raw_output_dir", required=True)
    p_eval.add_argument("--output_file", default=None)
    p_eval.add_argument("--val_file", default="data/processed/val.jsonl")
    p_eval.add_argument("--model_name", default="llama3-swerl-8b")

    args = parser.parse_args()

    dispatch = {
        "data": cmd_data,
        "sft_data": cmd_sft_data,
        "sft_train": cmd_sft_train,
        "train": cmd_train,
        "infer": cmd_infer,
        "eval": cmd_eval,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
