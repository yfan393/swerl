"""
run.py — Top-level orchestrator for SWE-RL Re-implementation
=============================================================
Unified pipeline runner supporting multiple model backends:
- Fine-tuned models (GRPO, SFT)
- OpenAI models (GPT-4, GPT-3.5-turbo)
- Open-source models (Llama 3.1 via vLLM)

Usage:
    # Full data pipeline
    python run.py data --config configs/data_config.yaml

    # Single data stage
    python run.py data --stage fetch|filter|extract|index

    # Generate SFT chain-of-thought data
    python run.py sft_data --config configs/train_config.yaml

    # SFT baseline training
    python run.py sft_train --config configs/train_config.yaml

    # GRPO (SWE-RL) training
    python run.py train --config configs/train_config.yaml

    # Run inference with fine-tuned model
    python run.py infer --model_path outputs/grpo/final --output_dir outputs/eval

    # Run inference with Llama models (via vLLM)
    python run.py llama_infer --port 8000 --output_dir outputs/llama_eval
    python run.py llama_infer --port 8000 --model llama_3b --num_samples 100

    # Compare Llama model sizes
    python run.py llama_compare --port 8000 --num_instances 20

    # Generate submission file
    python run.py eval --mode submission --raw_output_dir outputs/eval

    # Offline validation reward
    python run.py eval --mode val_reward --raw_output_dir outputs/eval

    # Compare first instance across all three models
    python run.py instance_compare \
        --trained_dir outputs/eval_student \
        --llama_1b_dir outputs/llama_1b_eval \
        --llama_3b_dir outputs/llama_3b_eval

Environment Variables:
    port_vllm: vLLM server port (default: 8000)
    OPENAI_API_KEY: OpenAI API key for GPT models
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure imports work even if run from subdirectory
sys.path.insert(0, str(Path(__file__).parent))

from utils.io_utils import read_jsonl, read_yaml


def cmd_data(args):
    """Run data preprocessing pipeline."""
    from data.preprocess_pipeline import run_pipeline

    run_pipeline(config_path=args.config, stage=getattr(args, "stage", "all"))


def cmd_sft_data(args):
    """Generate SFT chain-of-thought training data."""
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
    """Train SFT baseline model."""
    from sft.sft_train import train_sft

    train_sft(config_path=args.config)


def cmd_train(args):
    """Run GRPO (SWE-RL) training."""
    from training.grpo_train import train

    train(config_path=args.config)


def cmd_infer(args):
    """Run inference with fine-tuned model on SWE-bench."""
    from evaluation.run_inference import run_inference

    eval_cfg = {}
    if getattr(args, "config", None):
        cfg = read_yaml(args.config)
        eval_cfg = cfg.get("evaluation", {})

    output_dir = getattr(args, "output_dir", None) or eval_cfg.get("output_dir")
    if not output_dir:
        raise ValueError("Inference requires --output_dir or evaluation.output_dir in config")

    run_inference(
        model_path=args.model_path,
        output_dir=output_dir,
        dataset_name=getattr(args, "dataset", None) or eval_cfg.get("swe_bench_split", "princeton-nlp/SWE-bench_Verified"),
        num_samples=getattr(args, "num_samples", None) or eval_cfg.get("num_repair_samples", 500),
        temperature=getattr(args, "temperature", None) or eval_cfg.get("eval_temperature", 1.0),
        max_new_tokens=getattr(args, "max_new_tokens", None) or eval_cfg.get("max_new_tokens", 2048),
        top_k_chunks=getattr(args, "top_k_chunks", None) or eval_cfg.get("top_k_chunks", 8),
        repo_cache_dir=getattr(args, "repo_cache_dir", None) or eval_cfg.get("repo_cache_dir", "data/eval_repos"),
        embed_model_name=getattr(args, "embed_model", None) or eval_cfg.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2"),
        max_eval_files=getattr(args, "max_eval_files", None) or eval_cfg.get("max_eval_files", 300),
        batch_size=getattr(args, "batch_size", None) or eval_cfg.get("batch_size", 4),
    )


def cmd_llama_infer(args):
    """Run inference with Llama models via vLLM."""
    import logging
    from utils.llama_client import LlamaClusterClient
    from agent.prompts import build_messages
    from utils.io_utils import write_jsonl, ensure_parent_dir
    from datasets import load_dataset
    from tqdm import tqdm

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Get port from args or environment
    port = args.port or int(os.environ.get("port_vllm", 8000))
    logger.info(f"Connecting to vLLM on port {port}")

    # Initialize client
    client = LlamaClusterClient(port=port)

    # Get available models
    available_models = client.get_available_models()
    logger.info(f"Available models: {available_models}")

    # Select model
    if args.model:
        model_id = next((m for m in available_models if args.model in m), available_models[0])
    else:
        model_id = available_models[0]

    logger.info(f"Using model: {model_id}")

    # Load dataset
    logger.info(f"Loading SWE-bench dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split="test")
    num_samples = min(args.num_samples, len(dataset))
    dataset = dataset.select(range(num_samples))

    # Create output directory
    output_dir = Path(args.output_dir)
    ensure_parent_dir(output_dir / "dummy.txt")
    logger.info(f"Results will be saved to: {output_dir}")

    # Run inference
    results = []
    for i, instance in enumerate(tqdm(dataset, desc=f"Inferencing with {model_id.split('/')[-1]}")):
        try:
            # Build context
            from agent.rag_context_builder import build_code_context

            code_context = build_code_context(
                problem_statement=instance.get("problem_statement", ""),
                repo=instance.get("repo", ""),
                file_contents=instance.get("file_contents", {}),
            )

            # Build messages
            messages = build_messages(instance.get("problem_statement", ""), code_context)

            # Call model
            output = client.call(
                messages=messages,
                model=model_id,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )

            results.append({
                "instance_id": instance.get("instance_id"),
                "model": model_id,
                "output": output,
            })

        except Exception as e:
            logger.error(f"Error on instance {i}: {e}")
            results.append({
                "instance_id": instance.get("instance_id"),
                "model": model_id,
                "error": str(e),
            })

    # Save results
    output_file = output_dir / "raw_output.jsonl"
    write_jsonl(results, output_file)
    logger.info(f"Saved {len(results)} results to {output_file}")

    # Print stats
    stats = client.get_stats()
    logger.info(f"\n=== Inference Statistics ===")
    logger.info(f"Total requests: {stats['total_requests']}")
    logger.info(f"Success rate: {stats['success_rate']:.1f}%")
    logger.info(f"Tokens/sec: {stats['tokens_per_second']:.1f}")
    logger.info(f"Avg time: {stats['average_time_per_request_seconds']:.2f}s")


def cmd_llama_compare(args):
    """Compare Llama model sizes on code repair tasks."""
    from evaluation.compare_models import ModelComparison

    print("\n" + "="*70)
    print("LLAMA MODEL COMPARISON")
    print("="*70)

    comparison = ModelComparison(
        config_path=args.config,
        test_file=args.test_file,
        output_dir=args.output_dir,
    )

    results = comparison.run_comparison(
        port=args.port,
        models=args.models or ["llama_1b", "llama_3b", "llama_8b"],
        num_instances=args.num_instances,
    )

    comparison.save_results()
    comparison.print_summary()

    print("="*70)
    print(f"\nResults saved to: {args.output_dir}")


def cmd_instance_compare(args):
    """Compare first instance across trained and Llama models."""
    from evaluation.save_instance_comparison import InstanceComparison

    print("\n" + "="*70)
    print("INSTANCE COMPARISON: Trained Model vs Llama 1B vs Llama 3B")
    print("="*70)

    comparison = InstanceComparison(
        trained_model_dir=args.trained_dir,
        llama_1b_dir=args.llama_1b_dir,
        llama_3b_dir=args.llama_3b_dir,
        output_dir=args.output_dir,
    )

    comparison.compare()
    comparison.save_comparison()
    comparison.print_comparison()

    print(f"\n✓ Comparison saved to: {args.output_dir}")
    print(f"  JSON file: {args.output_dir}/instance_comparison.json")
    print(f"  Markdown: {args.output_dir}/instance_comparison.md (use in report)")


def cmd_eval(args):
    """Post-process and evaluate inference results."""
    from evaluation.evaluate import generate_submission_file, offline_val_reward

    if not Path(args.raw_output_dir).exists():
        raise FileNotFoundError(f"raw_output_dir does not exist: {args.raw_output_dir}")

    if args.mode == "submission":
        generate_submission_file(
            raw_output_dir=args.raw_output_dir,
            output_file=getattr(args, "output_file", None) or f"{args.raw_output_dir}/all_preds.jsonl",
            model_name=getattr(args, "model_name", "swerl-model"),
        )
    elif args.mode == "val_reward":
        offline_val_reward(
            raw_output_dir=args.raw_output_dir,
            val_file=getattr(args, "val_file", "data/processed/val.jsonl"),
        )


def main():
    parser = argparse.ArgumentParser(
        description="SWE-RL Pipeline Runner - Supports GRPO, SFT, and Llama Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Data pipeline
    p_data = subparsers.add_parser("data", help="Run data preprocessing pipeline")
    p_data.add_argument("--config", default="configs/data_config.yaml")
    p_data.add_argument("--stage", default="all", choices=["all", "fetch", "filter", "extract", "index"])

    # SFT data generation
    p_sft_data = subparsers.add_parser("sft_data", help="Generate SFT CoT training data")
    p_sft_data.add_argument("--config", default="configs/train_config.yaml")

    # SFT training
    p_sft_train = subparsers.add_parser("sft_train", help="Train SFT baseline model")
    p_sft_train.add_argument("--config", default="configs/train_config.yaml")

    # GRPO training
    p_train = subparsers.add_parser("train", help="Run GRPO (SWE-RL) training")
    p_train.add_argument("--config", default="configs/train_config.yaml")

    # Fine-tuned model inference
    p_infer = subparsers.add_parser("infer", help="Run inference with fine-tuned model on SWE-bench")
    p_infer.add_argument("--model_path", required=True, help="Path to fine-tuned model")
    p_infer.add_argument("--config", default=None)
    p_infer.add_argument("--output_dir", required=True)
    p_infer.add_argument("--dataset", default=None)
    p_infer.add_argument("--num_samples", type=int, default=None)
    p_infer.add_argument("--temperature", type=float, default=None)
    p_infer.add_argument("--max_new_tokens", type=int, default=None)
    p_infer.add_argument("--top_k_chunks", type=int, default=None)
    p_infer.add_argument("--repo_cache_dir", default=None)
    p_infer.add_argument("--embed_model", default=None)
    p_infer.add_argument("--max_eval_files", type=int, default=None)
    p_infer.add_argument("--batch_size", type=int, default=None)

    # Llama inference
    p_llama_infer = subparsers.add_parser("llama_infer", help="Run inference with Llama models via vLLM")
    p_llama_infer.add_argument("--port", type=int, default=None, help="vLLM server port (uses port_vllm env var if not set)")
    p_llama_infer.add_argument("--model", default=None, help="Model size: llama_1b, llama_3b, or llama_8b")
    p_llama_infer.add_argument("--output_dir", required=True, help="Output directory for results")
    p_llama_infer.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified", help="Dataset to run inference on")
    p_llama_infer.add_argument("--num_samples", type=int, default=500, help="Number of samples to process")
    p_llama_infer.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    p_llama_infer.add_argument("--max_tokens", type=int, default=2048, help="Max tokens to generate")

    # Llama model comparison
    p_llama_compare = subparsers.add_parser("llama_compare", help="Compare Llama model sizes (1B, 3B, 8B)")
    p_llama_compare.add_argument("--port", type=int, default=None, help="vLLM server port")
    p_llama_compare.add_argument("--config", default="configs/llama_models.yaml", help="Llama models config")
    p_llama_compare.add_argument("--test_file", default="data/processed/test.jsonl", help="Test instances")
    p_llama_compare.add_argument("--models", nargs="+", default=None, help="Models to compare")
    p_llama_compare.add_argument("--num_instances", type=int, default=None, help="Number of test instances")
    p_llama_compare.add_argument("--output_dir", default="evaluation/comparison_results")

    # Instance comparison
    p_instance_compare = subparsers.add_parser(
        "instance_compare",
        help="Compare first instance across trained model, Llama 1B, and Llama 3B"
    )
    p_instance_compare.add_argument(
        "--trained_dir",
        default="outputs/eval_student",
        help="Directory with trained model's raw_outputs.jsonl"
    )
    p_instance_compare.add_argument(
        "--llama_1b_dir",
        default="outputs/llama_1b_eval",
        help="Directory with Llama 1B's raw_outputs.jsonl"
    )
    p_instance_compare.add_argument(
        "--llama_3b_dir",
        default="outputs/llama_3b_eval",
        help="Directory with Llama 3B's raw_outputs.jsonl"
    )
    p_instance_compare.add_argument(
        "--output_dir",
        default="evaluation/instance_comparison",
        help="Where to save comparison results"
    )

    # Evaluation
    p_eval = subparsers.add_parser("eval", help="Post-process and evaluate results")
    p_eval.add_argument("--mode", choices=["submission", "val_reward"], default="submission")
    p_eval.add_argument("--raw_output_dir", required=True)
    p_eval.add_argument("--output_file", default=None)
    p_eval.add_argument("--val_file", default="data/processed/val.jsonl")
    p_eval.add_argument("--model_name", default="swerl-model")

    args = parser.parse_args()

    dispatch = {
        "data": cmd_data,
        "sft_data": cmd_sft_data,
        "sft_train": cmd_sft_train,
        "train": cmd_train,
        "infer": cmd_infer,
        "llama_infer": cmd_llama_infer,
        "llama_compare": cmd_llama_compare,
        "instance_compare": cmd_instance_compare,
        "eval": cmd_eval,
    }

    dispatch[args.command](args)


if __name__ == "__main__":
    main()
