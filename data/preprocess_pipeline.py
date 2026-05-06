"""
preprocess_pipeline.py
======================
End-to-end orchestration of all data pipeline stages:
    1. fetch_gharchive   — download raw PR events
    2. filter_prs        — enrich + filter via GitHub API
    3. extract_triples   — clone repos, build (issue, code, patch) triples
    4. build_rag_index   — embed code chunks, build FAISS index

Can be run as a single script or stage-by-stage.

Usage:
    # Full pipeline
    python -m data.preprocess_pipeline --config configs/data_config.yaml

    # Single stage
    python -m data.preprocess_pipeline --config configs/data_config.yaml --stage filter
"""

import argparse
import logging
from pathlib import Path

from data.fetch_gharchive import fetch_prs
from data.filter_prs import filter_prs
from data.extract_triples import extract_triples
from data.build_rag_index import build_index
from utils.io_utils import read_yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_config(config_path: str) -> dict:
    return read_yaml(config_path)


def validate_config(cfg: dict, stage: str) -> None:
    """Fail early on missing or contradictory data-pipeline settings."""
    required = ["gharchive", "github_api", "filtering", "dataset", "rag_index"]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Data config missing required sections: {missing}")

    ds = cfg["dataset"]
    if ds.get("num_seeds", 0) <= 0:
        raise ValueError("dataset.num_seeds must be positive")
    train_ratio = ds.get("train_ratio", 0)
    if not 0 < train_ratio < 1:
        raise ValueError("dataset.train_ratio must be between 0 and 1")

    rag = cfg["rag_index"]
    filtering = cfg["filtering"]
    if filtering.get("require_merged", True) is not True:
        raise ValueError("Only filtering.require_merged=true is supported")
    if filtering.get("require_linked_issue", True) is not True:
        raise ValueError("Only filtering.require_linked_issue=true is supported")
    if filtering.get("min_python_files", 1) <= 0:
        raise ValueError("filtering.min_python_files must be positive")
    if filtering.get("min_diff_chars", 50) < 0:
        raise ValueError("filtering.min_diff_chars must be non-negative")
    if filtering.get("max_files_changed", 20) <= 0:
        raise ValueError("filtering.max_files_changed must be positive")
    if rag.get("chunk_level", "function") != "function":
        raise ValueError("Only rag_index.chunk_level='function' is currently supported")
    if rag.get("faiss_index_type", "Flat") != "Flat":
        raise ValueError("Only rag_index.faiss_index_type='Flat' is currently supported")

    if stage in {"extract", "index"}:
        raw_dir = Path(ds["output_dir"]).parent / "raw"
        if stage == "extract" and not (raw_dir / "filtered_prs.jsonl").exists():
            raise FileNotFoundError("Run the filter stage before extract; filtered_prs.jsonl missing")
        if stage == "index" and not Path(ds["train_file"]).exists():
            raise FileNotFoundError("Run the extract stage before index; train.jsonl missing")


def stage_fetch(cfg: dict):
    gh = cfg["gharchive"]
    ds = cfg["dataset"]
    output_file = Path(ds["output_dir"]).parent / "raw" / "raw_prs.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fetch_prs(
        start_date=gh["start_date"],
        end_date=gh["end_date"],
        output_file=str(output_file),
        max_prs=ds["num_seeds"] * 20,   # Over-fetch to compensate for filtering losses
        max_workers=gh.get("max_workers", 4),
    )


def stage_filter(cfg: dict):
    ds = cfg["dataset"]
    raw_dir = Path(ds["output_dir"]).parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    filter_prs(
        input_file=str(raw_dir / "raw_prs.jsonl"),
        output_file=str(raw_dir / "filtered_prs.jsonl"),
        max_records=ds["num_seeds"] * 3,   # Keep 3x for triple extraction losses
        max_workers=cfg["github_api"].get("max_workers", 8),
        max_diff_chars=cfg.get("filtering", {}).get("max_diff_chars", 500_000),
        max_files_changed=cfg.get("filtering", {}).get("max_files_changed", 20),
        min_python_files=cfg.get("filtering", {}).get("min_python_files", 1),
        min_diff_chars=cfg.get("filtering", {}).get("min_diff_chars", 50),
        skip_patterns=cfg.get("filtering", {}).get("skip_patterns", []),
        bot_suffixes=cfg.get("filtering", {}).get("bot_suffixes", []),
    )


def stage_extract(cfg: dict):
    ds = cfg["dataset"]
    raw_dir = Path(ds["output_dir"]).parent / "raw"
    output_dir = Path(ds["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    extract_triples(
        input_file=str(raw_dir / "filtered_prs.jsonl"),
        output_dir=ds["output_dir"],
        repo_cache_dir=ds["repo_cache_dir"],
        num_seeds=ds["num_seeds"],
        train_ratio=ds["train_ratio"],
    )


def stage_index(cfg: dict):
    ds = cfg["dataset"]
    rag = cfg["rag_index"]

    # Create output directories
    Path(rag["index_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(rag["chunk_meta_path"]).parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Building RAG index: train_file=%s index=%s chunks=%s embed_model=%s max_chunk_tokens=%s",
        ds["train_file"],
        rag["index_path"],
        rag["chunk_meta_path"],
        rag["embed_model"],
        rag.get("max_chunk_tokens", 512),
    )
    build_index(
        train_file=ds["train_file"],
        index_path=rag["index_path"],
        chunk_meta_path=rag["chunk_meta_path"],
        embed_model_name=rag["embed_model"],
        max_chunk_tokens=rag.get("max_chunk_tokens", 512),
        chunk_level=rag.get("chunk_level", "function"),
        faiss_index_type=rag.get("faiss_index_type", "Flat"),
    )


STAGES = {
    "fetch": stage_fetch,
    "filter": stage_filter,
    "extract": stage_extract,
    "index": stage_index,
}


def run_pipeline(config_path: str, stage: str = "all"):
    cfg = load_config(config_path)
    validate_config(cfg, stage)
    if stage == "all":
        for name, fn in STAGES.items():
            logger.info(f"=== Running stage: {name} ===")
            fn(cfg)
    elif stage in STAGES:
        logger.info(f"=== Running stage: {stage} ===")
        STAGES[stage](cfg)
    else:
        raise ValueError(f"Unknown stage '{stage}'. Choose from: {list(STAGES)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_config.yaml")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all"] + list(STAGES.keys()),
        help="Which pipeline stage to run (default: all)",
    )
    args = parser.parse_args()
    run_pipeline(config_path=args.config, stage=args.stage)


if __name__ == "__main__":
    main()
