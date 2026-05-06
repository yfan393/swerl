"""
build_rag_index.py
==================
Parse Python source files from each repo in the training set into
function/class-level chunks, embed them with a SentenceTransformer,
and build a FAISS index for fast nearest-neighbour retrieval.

The index maps a GitHub issue description → top-k relevant code chunks
from the same repository.

Chunk schema (stored in chunks.jsonl):
    {
        "chunk_id": int,
        "instance_id": str,   # the PR/seed this chunk belongs to
        "repo": str,
        "file_path": str,
        "chunk_type": "function" | "class" | "file",
        "name": str,
        "start_line": int,
        "end_line": int,
        "content": str,       # raw source text of the chunk
    }

Usage:
    python -m data.build_rag_index \
        --train_file data/processed/train.jsonl \
        --index_path data/rag/faiss.index \
        --chunk_meta_path data/rag/chunks.jsonl \
        --embed_model sentence-transformers/all-MiniLM-L6-v2
"""

import argparse
import ast
import logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from utils.io_utils import read_jsonl, write_jsonl

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DEFAULT_MAX_CHUNK_CHARS = 2000   # truncate very long functions/classes


def extract_chunks_from_source(
    source: str,
    file_path: str,
    instance_id: str,
    repo: str,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> list[dict]:
    """
    Parse a Python source file with AST and extract function- and class-level chunks.
    Falls back to whole-file chunk if parsing fails.
    """
    chunks = []
    lines = source.splitlines()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Fallback: treat entire file as one chunk
        return [
            {
                "instance_id": instance_id,
                "repo": repo,
                "file_path": file_path,
                "chunk_type": "file",
                "name": file_path,
                "start_line": 1,
                "end_line": len(lines),
                "content": source[:max_chunk_chars],
            }
        ]

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

        start = node.lineno - 1  # 0-indexed
        # ast.end_lineno is available in Python 3.8+
        end = getattr(node, "end_lineno", start + 30)
        chunk_lines = lines[start:end]
        content = "\n".join(chunk_lines)
        if not content.strip():
            continue

        chunk_type = "class" if isinstance(node, ast.ClassDef) else "function"
        chunks.append(
            {
                "instance_id": instance_id,
                "repo": repo,
                "file_path": file_path,
                "chunk_type": chunk_type,
                "name": node.name,
                "start_line": start + 1,
                "end_line": end,
                "content": content[:max_chunk_chars],
            }
        )

    # If no functions/classes found, treat file as one chunk
    if not chunks:
        chunks.append(
            {
                "instance_id": instance_id,
                "repo": repo,
                "file_path": file_path,
                "chunk_type": "file",
                "name": file_path,
                "start_line": 1,
                "end_line": len(lines),
                "content": source[:max_chunk_chars],
            }
        )

    return chunks


def build_index(
    train_file: str,
    index_path: str,
    chunk_meta_path: str,
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 256,
    max_chunk_tokens: int = 512,
    chunk_level: str = "function",
    faiss_index_type: str = "Flat",
):
    """
    Full pipeline:
      1. Load training data
      2. Extract code chunks per record
      3. Embed chunks in batches
      4. Build FAISS index
      5. Save index + chunk metadata
    """
    index_path_obj = Path(index_path)
    chunk_meta_path_obj = Path(chunk_meta_path)
    index_path_obj.parent.mkdir(parents=True, exist_ok=True)
    chunk_meta_path_obj.parent.mkdir(parents=True, exist_ok=True)

    if chunk_level != "function":
        raise ValueError(
            f"Unsupported chunk_level={chunk_level!r}. "
            "This implementation currently supports only 'function'."
        )
    if faiss_index_type != "Flat":
        raise ValueError(
            f"Unsupported faiss_index_type={faiss_index_type!r}. "
            "This small-scale implementation currently supports only 'Flat'."
        )
    if not Path(train_file).exists():
        raise FileNotFoundError(f"Training file not found for RAG indexing: {train_file}")

    records = read_jsonl(train_file)

    logger.info(f"Building RAG index from {len(records)} training records")
    max_chunk_chars = max_chunk_tokens * 4
    logger.info(
        "RAG settings: embed_model=%s, batch_size=%s, max_chunk_tokens=%s",
        embed_model_name,
        batch_size,
        max_chunk_tokens,
    )

    all_chunks: list[dict] = []
    for record in tqdm(records, desc="Extracting chunks"):
        file_contents: dict[str, str] = record.get("file_contents", {})
        for file_path, content in file_contents.items():
            chunks = extract_chunks_from_source(
                source=content,
                file_path=file_path,
                instance_id=record["instance_id"],
                repo=record["repo"],
                max_chunk_chars=max_chunk_chars,
            )
            all_chunks.extend(chunks)

    logger.info(f"Total chunks extracted: {len(all_chunks)}")
    if not all_chunks:
        raise ValueError(
            f"No code chunks were extracted from {train_file}; "
            "check that processed records contain file_contents."
        )

    logger.info(f"Loading embedding model: {embed_model_name}")
    model = SentenceTransformer(embed_model_name)
    embedding_dim = model.get_sentence_embedding_dimension()
    logger.info(f"Embedding dimension: {embedding_dim}")

    texts = [
        f"{chunk['file_path']} :: {chunk['name']}\n{chunk['content']}"
        for chunk in all_chunks
    ]

    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks"):
        batch = texts[i: i + batch_size]
        embs = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        all_embeddings.append(embs)

    embeddings = np.vstack(all_embeddings).astype("float32")
    logger.info(f"Embeddings shape: {embeddings.shape}")

    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)
    logger.info(f"FAISS index has {index.ntotal} vectors")

    faiss.write_index(index, str(index_path_obj))
    logger.info(f"Saved FAISS index to {index_path}")

    for i, chunk in enumerate(all_chunks):
        chunk["chunk_id"] = i
    write_jsonl(chunk_meta_path_obj, all_chunks)
    logger.info(f"Saved {len(all_chunks)} chunk records to {chunk_meta_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="data/processed/train.jsonl")
    parser.add_argument("--index_path", default="data/rag/faiss.index")
    parser.add_argument("--chunk_meta_path", default="data/rag/chunks.jsonl")
    parser.add_argument(
        "--embed_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_chunk_tokens", type=int, default=512)
    parser.add_argument("--chunk_level", default="function")
    parser.add_argument("--faiss_index_type", default="Flat")
    args = parser.parse_args()

    build_index(
        train_file=args.train_file,
        index_path=args.index_path,
        chunk_meta_path=args.chunk_meta_path,
        embed_model_name=args.embed_model,
        batch_size=args.batch_size,
        max_chunk_tokens=args.max_chunk_tokens,
        chunk_level=args.chunk_level,
        faiss_index_type=args.faiss_index_type,
    )


if __name__ == "__main__":
    main()
