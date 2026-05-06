"""
save_instance_comparison.py
===========================
Save and compare outputs from trained model, Llama 1B, and Llama 3B on the first instance.

This utility:
  1. Extracts thinking blocks and patches from model outputs
  2. Computes rewards (similarity, correctness) for each
  3. Saves comprehensive comparison to JSON
  4. Provides formatted markdown for notebook display
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

from reward.reward_fn import (
    extract_thought_solution,
    parse_search_replace,
    calculate_combined_reward,
    FormatError,
)
from utils.io_utils import read_jsonl, ensure_parent_dir

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class InstanceComparison:
    """Load and compare first instance across multiple models."""

    def __init__(
        self,
        trained_model_dir: str,
        llama_1b_dir: str,
        llama_3b_dir: str,
        output_dir: str = "evaluation/instance_comparison",
    ):
        """
        Initialize comparison across three models.

        Args:
            trained_model_dir: Directory with trained model's raw_outputs.jsonl
            llama_1b_dir: Directory with Llama 1B's raw_outputs.jsonl
            llama_3b_dir: Directory with Llama 3B's raw_outputs.jsonl
            output_dir: Where to save comparison results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load outputs from each model
        self.trained_outputs = self._load_first_instance(
            trained_model_dir, "trained_model"
        )
        self.llama_1b_outputs = self._load_first_instance(llama_1b_dir, "llama_1b")
        self.llama_3b_outputs = self._load_first_instance(llama_3b_dir, "llama_3b")

        self.comparison_data = {}

    def _load_first_instance(self, output_dir: str, model_name: str) -> Dict[str, Any]:
        """Load first instance from raw_outputs.jsonl."""
        raw_file = Path(output_dir) / "raw_outputs.jsonl"

        if not raw_file.exists():
            logger.warning(f"No raw_outputs.jsonl found in {output_dir}")
            return {}

        records = list(read_jsonl(raw_file))
        if not records:
            logger.warning(f"No records in {raw_file}")
            return {}

        first = records[0]
        logger.info(
            f"{model_name}: Loaded instance {first.get('instance_id')} "
            f"with {len(first.get('outputs', []))} outputs"
        )

        return first

    def _extract_from_output(self, output: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract thinking and solution from model output."""
        try:
            thought, answer = extract_thought_solution(output)
            return thought, answer
        except FormatError as e:
            logger.debug(f"Failed to extract: {e}")
            return None, None

    def _compute_reward(
        self,
        output: str,
        file_contents: Dict[str, str],
        oracle_new_content: Dict[str, str],
        alpha: float = 0.3,
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute combined reward for an output."""
        try:
            reward, meta = calculate_combined_reward(
                code_context=file_contents,
                oracle_new_content=oracle_new_content,
                output=output,
                alpha=alpha,
            )
            return reward, meta
        except Exception as e:
            logger.debug(f"Reward computation failed: {e}")
            return -1.0, {"error": str(e)}

    def _process_outputs(
        self,
        model_outputs: Dict[str, Any],
        model_name: str,
        file_contents: Dict[str, str],
        oracle_new_content: Dict[str, str],
    ) -> Dict[str, Any]:
        """Process all outputs for a single model."""
        outputs = model_outputs.get("outputs", [])
        if not outputs:
            return {"model": model_name, "error": "No outputs available"}

        # Pick first output (or best if you prefer)
        first_output = outputs[0]

        thought, answer = self._extract_from_output(first_output)
        reward, reward_meta = self._compute_reward(
            first_output, file_contents, oracle_new_content
        )

        # Parse patches
        patches = {}
        if answer:
            try:
                patches = parse_search_replace(answer)
            except Exception:
                patches = {}

        return {
            "model": model_name,
            "instance_id": model_outputs.get("instance_id"),
            "output_full": first_output,  # Full output with tags
            "thinking": thought,
            "solution": answer,
            "patches": patches,
            "reward": {
                "combined": reward,
                "sim_score": reward_meta.get("sim_score"),
                "correctness_score": reward_meta.get("correctness_score"),
                "format_accuracy": reward_meta.get("format_accuracy"),
            },
            "format_error": reward_meta.get("error"),
        }

    def compare(self) -> Dict[str, Any]:
        """Run full comparison."""
        if not self.trained_outputs or not self.llama_1b_outputs or not self.llama_3b_outputs:
            logger.error("Not all model outputs loaded successfully")
            return {}

        # Use trained model's file contents and oracle as reference
        file_contents = self.trained_outputs.get("file_contents", {})
        problem_statement = self.trained_outputs.get("problem_statement", "")
        instance_id = self.trained_outputs.get("instance_id", "unknown")

        # Oracle data (from trained model's record)
        oracle_new_content = {}
        if "oracle_new_content" in self.trained_outputs:
            oracle_new_content = self.trained_outputs["oracle_new_content"]
            if isinstance(oracle_new_content, str):
                oracle_new_content = json.loads(oracle_new_content)

        logger.info(f"\nComparing instance {instance_id}")
        logger.info(f"Problem: {problem_statement[:100]}...")

        # Process each model
        trained_data = self._process_outputs(
            self.trained_outputs,
            "Trained Model",
            file_contents,
            oracle_new_content,
        )
        llama_1b_data = self._process_outputs(
            self.llama_1b_outputs,
            "Llama 1B",
            file_contents,
            oracle_new_content,
        )
        llama_3b_data = self._process_outputs(
            self.llama_3b_outputs,
            "Llama 3B",
            file_contents,
            oracle_new_content,
        )

        self.comparison_data = {
            "instance_id": instance_id,
            "problem_statement": problem_statement,
            "file_contents": file_contents,
            "models": {
                "trained": trained_data,
                "llama_1b": llama_1b_data,
                "llama_3b": llama_3b_data,
            },
        }

        return self.comparison_data

    def save_comparison(self, filename: str = "instance_comparison.json") -> Path:
        """Save comparison to JSON file."""
        output_path = self.output_dir / filename
        ensure_parent_dir(output_path)

        # Remove full outputs from saved JSON (too large)
        save_data = json.loads(json.dumps(self.comparison_data))
        for model_key in save_data.get("models", {}):
            if "output_full" in save_data["models"][model_key]:
                del save_data["models"][model_key]["output_full"]

        with open(output_path, "w") as f:
            json.dump(save_data, f, indent=2)

        logger.info(f"Comparison saved to {output_path}")
        return output_path

    def format_markdown(self) -> str:
        """Generate markdown for notebook display."""
        if not self.comparison_data:
            return "No comparison data available"

        models = self.comparison_data["models"]
        md = []

        md.append("# Instance Comparison: Trained Model vs Llama 1B vs Llama 3B\n")
        md.append(f"**Instance ID:** {self.comparison_data['instance_id']}\n")

        md.append(f"## Problem Statement\n")
        md.append(f"```\n{self.comparison_data['problem_statement']}\n```\n")

        # Reward comparison table
        md.append("## Reward Comparison\n")
        md.append("| Model | Combined Reward | Similarity | Correctness | Format |\n")
        md.append("|-------|-----------------|-----------|------------|--------|\n")

        for model_key, model_data in models.items():
            model_name = model_data["model"]
            reward = model_data["reward"]
            md.append(
                f"| {model_name} | "
                f"{reward.get('combined', 'N/A'):.3f} | "
                f"{reward.get('sim_score', 'N/A'):.3f} | "
                f"{reward.get('correctness_score', 'N/A'):.3f} | "
                f"{'✓' if not model_data.get('format_error') else '✗'} |\n"
            )

        # Thinking blocks
        md.append("## Thinking Blocks\n")
        for model_key, model_data in models.items():
            md.append(f"### {model_data['model']}\n")
            thinking = model_data.get("thinking")
            if thinking:
                md.append(f"```\n{thinking}\n```\n")
            else:
                md.append(f"⚠️ Could not extract thinking block\n")
                if model_data.get("format_error"):
                    md.append(f"Error: {model_data['format_error']}\n")

        # Solutions/Patches
        md.append("## Solutions (SEARCH/REPLACE)\n")
        for model_key, model_data in models.items():
            md.append(f"### {model_data['model']}\n")
            solution = model_data.get("solution")
            if solution:
                md.append(f"```\n{solution}\n```\n")
            else:
                md.append(f"⚠️ Could not extract solution\n")

        return "\n".join(md)

    def print_comparison(self) -> None:
        """Print comparison to console."""
        print(self.format_markdown())


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare first instance across trained model and Llama models"
    )
    parser.add_argument(
        "--trained-dir",
        required=True,
        help="Directory with trained model's raw_outputs.jsonl",
    )
    parser.add_argument(
        "--llama-1b-dir",
        required=True,
        help="Directory with Llama 1B's raw_outputs.jsonl",
    )
    parser.add_argument(
        "--llama-3b-dir",
        required=True,
        help="Directory with Llama 3B's raw_outputs.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/instance_comparison",
        help="Where to save comparison results",
    )

    args = parser.parse_args()

    # Run comparison
    comparison = InstanceComparison(
        trained_model_dir=args.trained_dir,
        llama_1b_dir=args.llama_1b_dir,
        llama_3b_dir=args.llama_3b_dir,
        output_dir=args.output_dir,
    )

    comparison.compare()
    comparison.save_comparison()
    comparison.print_comparison()


if __name__ == "__main__":
    main()
