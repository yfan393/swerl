"""
sft/generate_cot_data.py
========================
Generate synthetic chain-of-thought (CoT) training data for the SFT baseline,
following the Magicoder-style pipeline described in the SWE-RL paper (Appendix C).

For each seed PR in the training set:
  1. Build the code context (same RAG retrieval as RL)
  2. Either call a stronger LLM with the oracle patch as a hint, or build a
     no-cost oracle SEARCH/REPLACE target directly from the training record
  3. Filter: keep only outputs that parse correctly and have reward > threshold

The resulting data is used to fine-tune Llama 3.1 8B with standard cross-entropy loss
(supervised fine-tuning), creating the SFT baseline for comparison with SWE-RL.

Usage:
    python -m sft.generate_cot_data \
        --train_file data/processed/train.jsonl \
        --output_file data/processed/sft_cot_data.jsonl \
        --mode oracle \
        --max_records 50
"""

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Optional

from agent.prompts import SFT_COT_GENERATION
from agent.rag_context_builder import build_code_context
from reward.reward_fn import calculate_combined_reward
from utils.io_utils import append_jsonl, load_jsonl_id_set, read_jsonl

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DEFAULT_MODEL = "local-oracle"
MIN_REWARD_THRESHOLD = 0.5  # Only keep outputs with reward >= this


def validate_inputs(train_file: str, rag_index_path: str, rag_chunk_meta_path: str) -> None:
    if not Path(train_file).exists():
        raise FileNotFoundError(f"Training JSONL not found: {train_file}")
    if not Path(rag_index_path).exists():
        raise FileNotFoundError(f"RAG FAISS index not found: {rag_index_path}")
    if not Path(rag_chunk_meta_path).exists():
        raise FileNotFoundError(f"RAG chunk metadata not found: {rag_chunk_meta_path}")


def _build_request(
    record: dict,
    model: str,
    max_tokens: int,
    top_k_chunks: int,
    rag_index_path: str = "data/rag/faiss.index",
    rag_chunk_meta_path: str = "data/rag/chunks.jsonl",
    rag_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict:
    """Build the API request dict for one record."""
    problem_statement = record["problem_statement"]
    repo = record["repo"]
    file_contents: dict[str, str] = record.get("file_contents", {})
    oracle_patch = record.get("oracle_patch", "")

    code_context_str = build_code_context(
        problem_statement=problem_statement,
        repo=repo,
        file_contents=file_contents,
        max_context_tokens=10_000,
        top_k=top_k_chunks,
        index_path=rag_index_path,
        chunk_meta_path=rag_chunk_meta_path,
        embed_model=rag_embed_model,
    )

    prompt = SFT_COT_GENERATION.format(
        problem_statement=problem_statement,
        code_context=code_context_str,
        oracle_patch=oracle_patch,
    )

    return {
        "_instance_id": record["instance_id"],          # metadata, not sent to API
        "_repo": repo,
        "_code_context_str": code_context_str,
        "_file_contents": file_contents,
        "_oracle_new_content": record.get("oracle_new_content", {}),
        # actual API kwargs
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }


async def generate_all(
    records: list[dict],
    output_file: str,
    model: str = DEFAULT_MODEL,
    max_concurrent: int = 16,
    max_tokens: int = 4096,
    reward_threshold: float = MIN_REWARD_THRESHOLD,
    top_k_chunks: int = 8,
    rag_index_path: str = "data/rag/faiss.index",
    rag_chunk_meta_path: str = "data/rag/chunks.jsonl",
    rag_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    """
    Generate SFT CoT traces for all records using the teacher LLM.

    Uses OpenAIClient (with tenacity retry) + collect_responses_async
    (semaphore-bounded) from utils/api_client.py — same pattern as
    the upstream swe-rl agentless_mini/utils/api.py.
    """
    from utils.api_client import (
        OpenAIClient,
        collect_responses_async,
        extract_text_from_response,
    )

    if not Path(rag_index_path).exists():
        raise FileNotFoundError(f"RAG FAISS index not found: {rag_index_path}")
    if not Path(rag_chunk_meta_path).exists():
        raise FileNotFoundError(f"RAG chunk metadata not found: {rag_chunk_meta_path}")

    client = OpenAIClient()
    semaphore = asyncio.Semaphore(max_concurrent)

    # Resume: skip already processed
    out_path = Path(output_file)
    existing_ids = load_jsonl_id_set(out_path)

    pending = [r for r in records if r["instance_id"] not in existing_ids]
    logger.info(f"Generating CoT for {len(pending)} records (skipping {len(existing_ids)} done)")

    # Build API request dicts (including metadata fields prefixed with _)
    all_requests = [
        _build_request(
            r,
            model=model,
            max_tokens=max_tokens,
            top_k_chunks=top_k_chunks,
            rag_index_path=rag_index_path,
            rag_chunk_meta_path=rag_chunk_meta_path,
            rag_embed_model=rag_embed_model,
        )
        for r in pending
    ]

    # Separate metadata from actual API kwargs before sending
    metadata_list = [
        {k: v for k, v in req.items() if k.startswith("_")}
        for req in all_requests
    ]
    api_requests = [
        {k: v for k, v in req.items() if not k.startswith("_")}
        for req in all_requests
    ]

    idx_and_responses = await collect_responses_async(
        client, semaphore, api_requests, desc="Generating SFT CoT"
    )

    kept = 0
    for idx, response in sorted(idx_and_responses, key=lambda x: x[0]):
        output = extract_text_from_response(response)
        if not output:
            continue

        meta = metadata_list[idx]
        file_contents = meta["_file_contents"]
        oracle_new_content = meta["_oracle_new_content"]

        reward, _ = calculate_combined_reward(
            code_context=file_contents,
            oracle_new_content=oracle_new_content,
            output=output,
            alpha=0.3,
        )
        if reward < reward_threshold:
            logger.debug(f"Filtered {meta['_instance_id']}: reward={reward:.3f}")
            continue

        record_out = {
            "instance_id": meta["_instance_id"],
            "repo": meta["_repo"],
            "problem_statement": pending[idx]["problem_statement"],
            "code_context": meta["_code_context_str"],
            "output": output,
            "reward": reward,
        }
        append_jsonl(out_path, record_out)
        kept += 1

    logger.info(f"Generated {kept}/{len(pending)} valid CoT examples → {output_file}")
    return kept


def _build_oracle_search_replace_output(
    record: dict,
    top_k_chunks: int,
    rag_index_path: str = "data/rag/faiss.index",
    rag_chunk_meta_path: str = "data/rag/chunks.jsonl",
    rag_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Optional[dict]:
    """Create a valid SFT record from oracle file contents without any API call."""
    problem_statement = record["problem_statement"]
    repo = record["repo"]
    file_contents: dict[str, str] = record.get("file_contents", {})
    oracle_new_content: dict[str, str] = record.get("oracle_new_content", {})

    blocks = []
    for path, new_content in oracle_new_content.items():
        old_content = file_contents.get(path)
        if old_content is None or old_content == new_content:
            continue
        blocks.append(
            "```python\n"
            f"### {path}\n"
            "<<<<<<< SEARCH\n"
            f"{old_content.rstrip()}\n"
            "=======\n"
            f"{new_content.rstrip()}\n"
            ">>>>>>> REPLACE\n"
            "```"
        )

    if not blocks:
        return None

    code_context_str = build_code_context(
        problem_statement=problem_statement,
        repo=repo,
        file_contents=file_contents,
        max_context_tokens=10_000,
        top_k=top_k_chunks,
        index_path=rag_index_path,
        chunk_meta_path=rag_chunk_meta_path,
        embed_model=rag_embed_model,
    )
    output = (
        "<think>\n"
        "I inspected the issue and the relevant code context, then applied the "
        "reference fix as SEARCH/REPLACE edits.\n"
        "</think>\n"
        "<solution>\n"
        + "\n\n".join(blocks)
        + "\n</solution>"
    )

    return {
        "instance_id": record["instance_id"],
        "repo": repo,
        "problem_statement": problem_statement,
        "code_context": code_context_str,
        "output": output,
        "reward": 1.0,
        "source": "oracle",
    }


def generate_oracle_all(
    records: list[dict],
    output_file: str,
    top_k_chunks: int = 8,
    rag_index_path: str = "data/rag/faiss.index",
    rag_chunk_meta_path: str = "data/rag/chunks.jsonl",
    rag_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> int:
    """
    Generate SFT data without a paid teacher model.

    This creates exact SEARCH/REPLACE examples from file_contents and
    oracle_new_content. It is less rich than teacher-generated reasoning, but
    it is free, deterministic, and useful for small-scale smoke training.
    """
    if not Path(rag_index_path).exists():
        raise FileNotFoundError(f"RAG FAISS index not found: {rag_index_path}")
    if not Path(rag_chunk_meta_path).exists():
        raise FileNotFoundError(f"RAG chunk metadata not found: {rag_chunk_meta_path}")

    out_path = Path(output_file)
    existing_ids = load_jsonl_id_set(out_path)

    kept = 0
    for record in records:
        if record["instance_id"] in existing_ids:
            continue
        record_out = _build_oracle_search_replace_output(
            record,
            top_k_chunks=top_k_chunks,
            rag_index_path=rag_index_path,
            rag_chunk_meta_path=rag_chunk_meta_path,
            rag_embed_model=rag_embed_model,
        )
        if record_out is None:
            continue
        append_jsonl([record_out], out_path)
        kept += 1

    logger.info(f"Generated {kept} oracle SFT examples -> {output_file}")
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="data/processed/train.jsonl")
    parser.add_argument("--output_file", default="data/processed/sft_cot_data.jsonl")
    parser.add_argument("--mode", choices=["oracle", "api"], default="oracle")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--max_concurrent", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--reward_threshold", type=float, default=MIN_REWARD_THRESHOLD)
    parser.add_argument("--top_k_chunks", type=int, default=8)
    parser.add_argument("--rag_index_path", default="data/rag/faiss.index")
    parser.add_argument("--rag_chunk_meta_path", default="data/rag/chunks.jsonl")
    parser.add_argument(
        "--rag_embed_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    args = parser.parse_args()

    validate_inputs(args.train_file, args.rag_index_path, args.rag_chunk_meta_path)

    records = read_jsonl(args.train_file)

    if args.max_records:
        records = records[: args.max_records]

    if args.mode == "oracle":
        generate_oracle_all(
            records=records,
            output_file=args.output_file,
            top_k_chunks=args.top_k_chunks,
            rag_index_path=args.rag_index_path,
            rag_chunk_meta_path=args.rag_chunk_meta_path,
            rag_embed_model=args.rag_embed_model,
        )
    else:
        asyncio.run(
            generate_all(
                records=records,
                output_file=args.output_file,
                model=args.model,
                max_concurrent=args.max_concurrent,
                max_tokens=args.max_tokens,
                reward_threshold=args.reward_threshold,
                top_k_chunks=args.top_k_chunks,
                rag_index_path=args.rag_index_path,
                rag_chunk_meta_path=args.rag_chunk_meta_path,
                rag_embed_model=args.rag_embed_model,
            )
        )


if __name__ == "__main__":
    main()
