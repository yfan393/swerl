"""
compare_models.py
=================
Compare performance of Llama 3.1 models (1B, 3B, 8B) on code repair tasks.

Evaluates:
  - Code repair accuracy (pass rate)
  - Format correctness (valid SEARCH/REPLACE)
  - Inference speed
  - Resource efficiency

Output: JSON results + comparison plots
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from tqdm import tqdm

from agent.rag_context_builder import build_code_context
from reward.reward_fn import (
    extract_thought_solution,
    parse_search_replace,
    check_correctness,
    calculate_combined_reward,
)
from utils.io_utils import read_jsonl, write_jsonl, read_yaml
from utils.llama_client import get_llama_client

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class ModelComparison:
    """Run and track comparison of multiple Llama models."""

    def __init__(
        self,
        config_path: str,
        test_file: str,
        output_dir: str = "evaluation/comparison_results",
    ):
        """
        Initialize comparison study.

        Args:
            config_path: Path to llama_models.yaml config
            test_file: Path to test instances JSONL
            output_dir: Directory for results
        """
        self.config = read_yaml(config_path)
        self.test_instances = read_jsonl(test_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_file": test_file,
            "num_instances": len(self.test_instances),
            "models": {},
        }

    def evaluate_model(
        self,
        model_key: str,
        port: Optional[int] = None,
        num_instances: Optional[int] = None,
    ) -> Dict[str, any]:
        """
        Evaluate a single model on test instances.

        Args:
            model_key: Key in config (e.g., 'llama_1b')
            port: vLLM server port (uses port_vllm env var if None)
            num_instances: Limit number of test instances (for quick eval)

        Returns:
            Dictionary with evaluation metrics
        """
        model_config = self.config["models"][model_key]
        model_id = model_config["model_id"]

        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {model_key} ({model_id})")
        logger.info(f"{'='*60}\n")

        # Initialize client (uses port_vllm env var if port=None)
        client = get_llama_client(port=port)

        # Prepare test set
        test_set = self.test_instances
        if num_instances:
            test_set = test_set[:num_instances]

        logger.info(f"Testing on {len(test_set)} instances")

        # Tracking metrics
        metrics = {
            "model": model_key,
            "model_name": model_name,
            "test_instances": len(test_set),
            "results": [],
            "summary": {
                "total_time_seconds": 0.0,
                "pass_rate": 0.0,
                "format_correctness": 0.0,
                "average_time_per_instance_ms": 0.0,
                "tokens_per_second": 0.0,
                "errors": 0,
            },
        }

        # Run inference on each instance
        start_time = time.time()
        passed = 0
        format_correct = 0

        for i, instance in enumerate(tqdm(test_set, desc=f"Evaluating {model_key}")):
            try:
                # Build prompt with RAG context
                problem_statement = instance.get("problem_statement", "")
                repo = instance.get("repo", "")
                file_contents = instance.get("file_contents", {})

                code_context = build_code_context(
                    problem_statement=problem_statement,
                    repo=repo,
                    file_contents=file_contents,
                )

                # Build messages
                from agent.prompts import build_messages

                messages = build_messages(problem_statement, code_context)

                # Call model
                instance_start = time.time()
                output = client.call(
                    messages=messages,
                    model=model_id,
                    max_tokens=model_config["max_output_tokens"],
                    temperature=model_config["inference"]["temperature"],
                    top_p=model_config["inference"]["top_p"],
                )
                instance_time = time.time() - instance_start

                # Parse output
                try:
                    thought, solution = extract_thought_solution(output)
                    parsed_patches = parse_search_replace(solution)
                    format_ok = len(parsed_patches) > 0
                except Exception:
                    format_ok = False
                    parsed_patches = {}

                # Check correctness (if oracle available)
                correct = False
                if "oracle_patch" in instance:
                    try:
                        correct = check_correctness(
                            instance.get("original_code", {}),
                            parsed_patches,
                            instance.get("oracle_patch", {}),
                        )
                    except Exception:
                        pass

                # Track result
                result = {
                    "instance_id": instance.get("instance_id"),
                    "time_seconds": instance_time,
                    "format_correct": format_ok,
                    "correctness": correct,
                    "output_length": len(output),
                }
                metrics["results"].append(result)

                if correct:
                    passed += 1
                if format_ok:
                    format_correct += 1

            except Exception as e:
                logger.error(f"Error on instance {i}: {e}")
                metrics["summary"]["errors"] += 1
                metrics["results"].append(
                    {
                        "instance_id": instance.get("instance_id"),
                        "error": str(e),
                    }
                )

        # Calculate summary
        total_time = time.time() - start_time
        metrics["summary"]["total_time_seconds"] = total_time
        metrics["summary"]["pass_rate"] = (
            passed / len(test_set) * 100 if test_set else 0
        )
        metrics["summary"]["format_correctness"] = (
            format_correct / len(test_set) * 100 if test_set else 0
        )
        metrics["summary"]["average_time_per_instance_ms"] = (
            (total_time / len(test_set)) * 1000 if test_set else 0
        )

        # Get client stats
        client_stats = client.get_stats()
        metrics["summary"]["tokens_per_second"] = client_stats.get(
            "tokens_per_second", 0
        )
        metrics["client_stats"] = client_stats

        return metrics

    def run_comparison(
        self,
        port: Optional[int] = None,
        models: Optional[List[str]] = None,
        num_instances: Optional[int] = None,
    ) -> Dict:
        """
        Run comparison across multiple models.

        Args:
            port: vLLM server port (uses port_vllm env var if None)
            models: List of model keys to evaluate (default: all in config)
            num_instances: Limit test instances per model

        Returns:
            Results dictionary
        """
        if models is None:
            models = self.config["comparison"]["models_to_compare"]

        logger.info(f"Starting comparison of {len(models)} models")
        if port:
            logger.info(f"vLLM server port: {port}")
        else:
            logger.info("Using port_vllm environment variable")

        for model_key in models:
            model_results = self.evaluate_model(
                model_key=model_key,
                port=port,
                num_instances=num_instances,
            )
            self.results["models"][model_key] = model_results

        return self.results

    def save_results(self, filename: str = "comparison_results.json") -> Path:
        """
        Save results to JSON file.

        Args:
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Results saved to {output_path}")
        return output_path

    def print_summary(self) -> None:
        """Print summary comparison table."""
        logger.info("\n" + "=" * 80)
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info("=" * 80)

        # Header
        print(
            f"{'Model':<20} {'Pass Rate':<12} {'Format OK':<12} {'Avg Time (ms)':<15} {'Tokens/sec':<12}"
        )
        print("-" * 80)

        # Rows
        for model_key, model_results in self.results["models"].items():
            summary = model_results["summary"]
            print(
                f"{model_key:<20} "
                f"{summary['pass_rate']:.1f}%{'':<7} "
                f"{summary['format_correctness']:.1f}%{'':<7} "
                f"{summary['average_time_per_instance_ms']:.1f}{'':<9} "
                f"{summary['tokens_per_second']:.1f}"
            )

        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Compare Llama 3.1 models on code repair",
        epilog="Note: vLLM port can be specified via --port or port_vllm environment variable"
    )
    parser.add_argument(
        "--config",
        default="configs/llama_models.yaml",
        help="Path to model configuration",
    )
    parser.add_argument(
        "--test-file",
        default="data/processed/test.jsonl",
        help="Path to test instances",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="vLLM server port (default: uses port_vllm env var, fallback 8000)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to evaluate (default: all in config)",
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=None,
        help="Limit number of test instances per model",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/comparison_results",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Run comparison
    comparison = ModelComparison(
        config_path=args.config,
        test_file=args.test_file,
        output_dir=args.output_dir,
    )

    results = comparison.run_comparison(
        port=args.port,
        models=args.models,
        num_instances=args.num_instances,
    )

    # Save and print results
    output_file = comparison.save_results()
    comparison.print_summary()

    logger.info(f"\nFull results saved to: {output_file}")


if __name__ == "__main__":
    main()
