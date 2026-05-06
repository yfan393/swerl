"""
agent/retriever.py
==================
Embedding-based code retriever. Loads the FAISS index built by
data/build_rag_index.py and retrieves the top-k most relevant code
chunks for a given GitHub issue description.

The retriever is repo-aware: it filters FAISS candidates to only return
chunks belonging to the same repository as the query (so we don't mix
code from different projects during training).

Usage (standalone):
    from agent.retriever import CodeRetriever
    retriever = CodeRetriever(index_path="data/rag/faiss.index",
                              chunk_meta_path="data/rag/chunks.jsonl")
    chunks = retriever.retrieve(
        query="memory leak in cache module",
        repo="owner/repo",
        top_k=8,
    )
"""

import logging
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.io_utils import read_jsonl

logger = logging.getLogger(__name__)


class CodeRetriever:
    """
    Singleton-friendly retriever that wraps a FAISS index of code chunks.

    Attributes:
        index       : FAISS index (inner-product over L2-normalized embeddings)
        chunks      : list of chunk metadata dicts (mirrors chunks.jsonl)
        model       : SentenceTransformer used to embed queries
        repo_index  : dict mapping repo name → list of chunk IDs for fast filtering
    """

    def __init__(
        self,
        index_path: str,
        chunk_meta_path: str,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        logger.info(f"Loading FAISS index from {index_path}")
        self.index = faiss.read_index(str(Path(index_path)))

        logger.info(f"Loading chunk metadata from {chunk_meta_path}")
        self.chunks: list[dict] = read_jsonl(chunk_meta_path)

        assert len(self.chunks) == self.index.ntotal, (
            f"Chunk count mismatch: {len(self.chunks)} chunks vs {self.index.ntotal} vectors"
        )

        logger.info(f"Loading embedding model: {embed_model}")
        self.model = SentenceTransformer(embed_model)

        # Build repo → [chunk_ids] index for O(1) filtering
        self.repo_index: dict[str, list[int]] = {}
        for chunk in self.chunks:
            self.repo_index.setdefault(chunk["repo"], []).append(chunk["chunk_id"])

        logger.info(
            f"Retriever ready: {self.index.ntotal} vectors, "
            f"{len(self.repo_index)} repos"
        )

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a text query, returning a normalized float32 vector."""
        emb = self.model.encode([query], normalize_embeddings=True)
        return emb.astype("float32")  # shape (1, D)

    def retrieve(
        self,
        query: str,
        repo: Optional[str] = None,
        top_k: int = 8,
        search_multiplier: int = 10,
    ) -> list[dict]:
        """
        Retrieve the top-k most relevant code chunks.

        Args:
            query           : The GitHub issue description text.
            repo            : If given, filter results to only chunks from this repo.
                              During training this should always be set to the seed's repo.
                              During SWE-bench evaluation, set to the benchmark repo.
            top_k           : Number of chunks to return.
            search_multiplier : Retrieve top_k * search_multiplier candidates from FAISS
                              before repo-filtering to improve recall.

        Returns:
            List of chunk metadata dicts, ordered by relevance (most relevant first).
        """
        query_vec = self.embed_query(query)

        # If we need repo-level filtering, over-retrieve from FAISS first
        k_search = top_k * search_multiplier if repo else top_k
        k_search = min(k_search, self.index.ntotal)

        scores, indices = self.index.search(query_vec, k_search)
        # scores shape: (1, k_search)  indices shape: (1, k_search)
        scores = scores[0]
        indices = indices[0]

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0:  # FAISS can return -1 for empty slots
                continue
            chunk = self.chunks[idx]
            if repo and chunk["repo"] != repo:
                continue
            results.append({**chunk, "score": float(score)})
            if len(results) >= top_k:
                break

        return results

    def retrieve_for_instance(
        self,
        problem_statement: str,
        repo: str,
        file_contents: dict[str, str],
        top_k: int = 8,
    ) -> list[dict]:
        """
        Convenience method: retrieve top-k chunks for a training instance.
        Falls back to returning all file chunks if retrieval yields nothing.

        Args:
            problem_statement : Issue text used as the query.
            repo              : Repository to filter on.
            file_contents     : {path: content} of files from the training record
                                (used as fallback if retrieval fails).
            top_k             : Desired number of chunks.
        """
        chunks = self.retrieve(query=problem_statement, repo=repo, top_k=top_k)

        if not chunks:
            # Fallback: synthesise fake chunks from file_contents
            logger.warning(
                f"No RAG chunks found for repo={repo}. Using file_contents as fallback."
            )
            for path, content in file_contents.items():
                chunks.append(
                    {
                        "file_path": path,
                        "chunk_type": "file",
                        "name": path,
                        "content": content,
                        "score": 0.0,
                    }
                )
                if len(chunks) >= top_k:
                    break

        return chunks


# ─── Singleton accessor (used by training loop) ────────────────────────────────

_retriever_instances: dict[tuple[str, str, str], CodeRetriever] = {}


def get_retriever(
    index_path: str = "data/rag/faiss.index",
    chunk_meta_path: str = "data/rag/chunks.jsonl",
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> CodeRetriever:
    """Return a process-wide singleton CodeRetriever (lazy init)."""
    key = (index_path, chunk_meta_path, embed_model)
    if key not in _retriever_instances:
        _retriever_instances[key] = CodeRetriever(
            index_path=index_path,
            chunk_meta_path=chunk_meta_path,
            embed_model=embed_model,
        )
    return _retriever_instances[key]
