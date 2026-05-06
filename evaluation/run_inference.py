"""
Generate repair outputs for SWE-bench-style instances.

The important implementation detail is that evaluation context is built from the
target repository at base_commit. This avoids relying on a training RAG index
that may not contain benchmark repositories, and it lets post-processing convert
SEARCH/REPLACE answers into real unified diffs.
"""

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent.prompts import build_messages
from agent.rag_context_builder import RAGContextBuilder, build_code_context
from data.build_rag_index import extract_chunks_from_source
from reward.reward_fn import FormatError, extract_thought_solution, parse_search_replace
from utils.api_client import parse_thinking_output
from utils.io_utils import append_jsonl, ensure_parent_dir, load_jsonl_id_set
from utils.repo_utils import ensure_commit_available, list_files_at_commit, read_file_at_commit

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_EMBEDDER_CACHE: dict[str, SentenceTransformer] = {}


def _get_embedder(model_name: str) -> SentenceTransformer:
    """Process-wide embedder cache to avoid reloading per instance."""
    embedder = _EMBEDDER_CACHE.get(model_name)
    if embedder is None:
        logger.info("Loading embedding model: %s", model_name)
        embedder = SentenceTransformer(model_name)
        _EMBEDDER_CACHE[model_name] = embedder
    return embedder


def list_python_files(repo_path: Path, base_commit: str) -> list[str]:
    files = []
    for path in list_files_at_commit(repo_path, base_commit):
        lowered = path.lower()
        if not path.endswith(".py"):
            continue
        if lowered.startswith(("docs/", "tests/", "test/")):
            continue
        files.append(path)
    return files


def _terms(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]+", text) if len(t) > 2}


def select_candidate_files(files: list[str], problem_statement: str, max_files: int) -> list[str]:
    """Path-name prefilter before embedding chunks from repo files."""
    query_terms = _terms(problem_statement)

    def score(path: str) -> tuple[int, int, str]:
        path_terms = _terms(path.replace("/", " "))
        overlap = len(query_terms & path_terms)
        test_penalty = 1 if "test" in path.lower() else 0
        return (-overlap, test_penalty, path)

    return sorted(files, key=score)[:max_files]
def build_eval_code_context(
    instance: dict,
    max_context_tokens: int,
    top_k_chunks: int,
    repo_cache_dir: str,
    embed_model_name: str,
    max_eval_files: int,
) -> tuple[str, dict[str, str], str]:
    repo = instance.get("repo", "")
    base_commit = instance.get("base_commit", "")
    problem_statement = instance.get("problem_statement", "")
    instance_id = instance.get("instance_id", "")

    if not repo or not base_commit:
        logger.warning("Instance %s missing repo/base_commit metadata", instance_id)
        return "", {}, "missing_repo_metadata"

    try:
        repo_path = ensure_commit_available(repo, base_commit, repo_cache_dir)
    except RuntimeError as e:
        logger.warning("%s", e)
        return "", {}, "repo_unavailable"

    python_files = list_python_files(repo_path, base_commit)
    selected_files = select_candidate_files(python_files, problem_statement, max_eval_files)
    logger.info(
        "Eval context %s: %s python files found, reading %s candidates",
        instance_id,
        len(python_files),
        len(selected_files),
    )

    file_contents: dict[str, str] = {}
    chunks: list[dict] = []
    for file_path in selected_files:
        content = read_file_at_commit(repo_path, base_commit, file_path)
        if content is None:
            continue
        file_contents[file_path] = content
        chunks.extend(
            extract_chunks_from_source(
                source=content,
                file_path=file_path,
                instance_id=instance_id,
                repo=repo,
            )
        )

    if not chunks:
        logger.warning("No eval chunks built for %s", instance_id)
        return "", file_contents, "no_chunks"

    logger.info("Embedding %s eval chunks for %s", len(chunks), instance_id)
    embedder = _get_embedder(embed_model_name)
    query_vec = embedder.encode([problem_statement], normalize_embeddings=True).astype("float32")
    texts = [f"{c['file_path']} :: {c.get('name', '')}\n{c['content']}" for c in chunks]
    chunk_vecs = embedder.encode(texts, normalize_embeddings=True).astype("float32")
    scores = np.matmul(chunk_vecs, query_vec[0])
    ranked = np.argsort(-scores)[:top_k_chunks]
    ranked_chunks = [{**chunks[int(i)], "score": float(scores[int(i)])} for i in ranked]

    builder = RAGContextBuilder(
        retriever=None,  # type: ignore[arg-type]
        max_context_tokens=max_context_tokens,
        top_k=top_k_chunks,
    )
    return builder.build_from_chunks(ranked_chunks), file_contents, "repo_embedding"


def load_model(model_path: str, torch_dtype=torch.float16):
    logger.info("Loading model from %s", model_path)
    adapter_config = Path(model_path) / "adapter_config.json"
    if adapter_config.exists():
        from peft import PeftModel

        with open(adapter_config) as f:
            base_model_name = json.load(f).get("base_model_name_or_path", model_path)
        logger.info("Merging LoRA adapter with base model %s", base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()
    return model


def generate_patches_for_instance(
    model,
    tokenizer,
    instance: dict,
    num_samples: int = 500,
    max_new_tokens: int = 2048,
    temperature: float = 1.0,
    top_k_chunks: int = 8,
    max_context_tokens: int = 12_000,
    batch_size: int = 4,
    repo_cache_dir: str = "data/eval_repos",
    embed_model_name: str = DEFAULT_EMBED_MODEL,
    max_eval_files: int = 300,
) -> tuple[list[str], dict[str, str], str]:
    problem_statement = instance.get("problem_statement", "")
    repo = instance.get("repo", "")

    code_context_str, file_contents, context_source = build_eval_code_context(
        instance=instance,
        max_context_tokens=max_context_tokens,
        top_k_chunks=top_k_chunks,
        repo_cache_dir=repo_cache_dir,
        embed_model_name=embed_model_name,
        max_eval_files=max_eval_files,
    )
    if not code_context_str:
        logger.warning(
            "Falling back to training RAG index for %s; eval context source=%s",
            instance.get("instance_id"),
            context_source,
        )
        try:
            code_context_str = build_code_context(
                problem_statement=problem_statement,
                repo=repo,
                file_contents={},
                max_context_tokens=max_context_tokens,
                top_k=top_k_chunks,
            )
        except Exception as e:
            logger.warning("Training RAG fallback failed for %s: %s", instance.get("instance_id"), e)
            code_context_str = ""
    if not code_context_str:
        raise RuntimeError(
            f"Could not build any code context for {instance.get('instance_id')}. "
            "Check network access, repo_cache_dir, or RAG index paths."
        )

    messages = build_messages(problem_statement, code_context_str)
    prompt_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    device = next(model.parameters()).device
    model_max_len = int(getattr(tokenizer, "model_max_length", 8192) or 8192)
    prompt_max_len = max(256, model_max_len - max_new_tokens)
    inputs = tokenizer(
        prompt_str,
        return_tensors="pt",
        truncation=True,
        max_length=prompt_max_len,
    ).to(device)
    prompt_len = inputs["input_ids"].shape[1]

    all_outputs = []
    autocast_enabled = torch.cuda.is_available()
    autocast_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    with torch.no_grad():
        for start in range(0, num_samples, batch_size):
            count = min(batch_size, num_samples - start)
            expanded_input_ids = inputs["input_ids"].expand(count, -1)
            expanded_attention_mask = inputs["attention_mask"].expand(count, -1)
            current_count = count
            generated = 0
            while generated < count:
                run_count = min(current_count, count - generated)
                try:
                    with torch.cuda.amp.autocast(dtype=autocast_dtype, enabled=autocast_enabled):
                        output_ids = model.generate(
                            input_ids=expanded_input_ids[generated:generated + run_count],
                            attention_mask=expanded_attention_mask[generated:generated + run_count],
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=temperature,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    for i in range(run_count):
                        gen_ids = output_ids[i, prompt_len:]
                        all_outputs.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
                    generated += run_count
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    if current_count == 1:
                        raise RuntimeError(
                            "CUDA OOM during inference even at batch_size=1. "
                            "Reduce evaluation.max_new_tokens or use a smaller model."
                        ) from e
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    next_count = max(1, current_count // 2)
                    logger.warning(
                        "CUDA OOM for %s with batch=%s, retrying with batch=%s",
                        instance.get("instance_id"),
                        current_count,
                        next_count,
                    )
                    current_count = next_count

    return all_outputs, file_contents, context_source


def is_valid_format(output: str) -> bool:
    try:
        _, answer = extract_thought_solution(output)
        return bool(parse_search_replace(answer))
    except FormatError:
        pass

    try:
        answer = parse_thinking_output(output)
        if answer and answer != output:
            return bool(parse_search_replace(answer))
    except Exception:
        pass
    return False


def run_inference(
    model_path: str,
    output_dir: str,
    dataset_name: str = "princeton-nlp/SWE-bench_Verified",
    num_samples: int = 500,
    temperature: float = 1.0,
    max_new_tokens: int = 2048,
    top_k_chunks: int = 8,
    resume: bool = True,
    repo_cache_dir: str = "data/eval_repos",
    embed_model_name: str = DEFAULT_EMBED_MODEL,
    max_eval_files: int = 300,
    batch_size: int = 4,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    raw_output_file = output_path / "raw_outputs.jsonl"
    format_report_file = output_path / "format_report.json"

    logger.info(
        "Inference settings: dataset=%s samples=%s max_new_tokens=%s top_k=%s repo_cache=%s",
        dataset_name,
        num_samples,
        max_new_tokens,
        top_k_chunks,
        repo_cache_dir,
    )
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if top_k_chunks <= 0:
        raise ValueError("top_k_chunks must be positive")
    if max_eval_files <= 0:
        raise ValueError("max_eval_files must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    model = load_model(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading dataset %s", dataset_name)
    dataset = load_dataset(dataset_name, split="test")
    instances = list(dataset)
    logger.info("Evaluating %s instances", len(instances))

    done_ids: set[str] = set()
    if resume:
        done_ids = load_jsonl_id_set(raw_output_file)
        logger.info("Resuming: skipping %s already processed instances", len(done_ids))

    format_correct = 0
    total_outputs = 0

    for instance in tqdm(instances, desc="Generating patches"):
        instance_id = instance["instance_id"]
        if instance_id in done_ids:
            continue

        outputs, file_contents, context_source = generate_patches_for_instance(
            model=model,
            tokenizer=tokenizer,
            instance=instance,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k_chunks=top_k_chunks,
            repo_cache_dir=repo_cache_dir,
            embed_model_name=embed_model_name,
            max_eval_files=max_eval_files,
            batch_size=batch_size,
        )

        n_valid = sum(1 for output in outputs if is_valid_format(output))
        format_correct += n_valid
        total_outputs += len(outputs)

        record = {
            "instance_id": instance_id,
            "outputs": outputs,
            "format_accuracy": n_valid / len(outputs) if outputs else 0.0,
            "file_contents": file_contents,
            "context_source": context_source,
        }
        append_jsonl(raw_output_file, record)
        logger.info(
            "%s: valid format %s/%s, context_source=%s, files_saved=%s",
            instance_id,
            n_valid,
            len(outputs),
            context_source,
            len(file_contents),
        )

    fmt_acc = format_correct / total_outputs if total_outputs > 0 else 0.0
    report = {
        "model_path": model_path,
        "dataset": dataset_name,
        "num_instances": len(instances),
        "num_samples_per_instance": num_samples,
        "total_outputs": total_outputs,
        "format_accuracy": fmt_acc,
        "repo_cache_dir": repo_cache_dir,
        "embed_model": embed_model_name,
        "max_eval_files": max_eval_files,
        "batch_size": batch_size,
    }
    ensure_parent_dir(format_report_file)
    with open(format_report_file, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Format accuracy: %.1f%%", fmt_acc * 100)
    logger.info("Raw outputs saved to %s", raw_output_file)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--top_k_chunks", type=int, default=8)
    parser.add_argument("--repo_cache_dir", default="data/eval_repos")
    parser.add_argument("--embed_model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--max_eval_files", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    run_inference(
        model_path=args.model_path,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_k_chunks=args.top_k_chunks,
        repo_cache_dir=args.repo_cache_dir,
        embed_model_name=args.embed_model,
        max_eval_files=args.max_eval_files,
        batch_size=args.batch_size,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
