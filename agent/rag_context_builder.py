"""
agent/rag_context_builder.py
============================
Converts RAG-retrieved code chunks into a formatted code context string
suitable for the policy LLM's input prompt.

Key responsibilities:
  - Deduplicate chunks (same file/function can appear multiple times)
  - Respect a token budget (16k context window minus prompt overhead)
  - Format each chunk as ### path/to/file.py … content …
  - Provide a fallback path for SWE-bench evaluation (where we may use
    the oracle file list instead of retrieval)

This module is the bridge between retriever.py and prompts.py.
"""

import logging
from typing import Optional

from agent.retriever import CodeRetriever, get_retriever
# Use the shared token counter — respects TOKENIZER_MODEL / TOKENIZER_TYPE env vars
# so it automatically uses the HF Llama tokenizer when available, falling back
# to tiktoken for fast approximation during data preprocessing.
from utils.token_counter import count_tokens

logger = logging.getLogger(__name__)

# Estimate of non-code tokens in the prompt (system + user instructions + issue text)
PROMPT_OVERHEAD_TOKENS = 1_500
# Default max total context tokens
DEFAULT_MAX_CONTEXT_TOKENS = 12_000


def format_chunk(chunk: dict) -> str:
    """
    Format a single code chunk as a labeled block.

    Output:
        ### path/to/file.py  (ClassName.method_name, lines 42-87)
        <content>
    """
    path = chunk.get("file_path", "unknown")
    name = chunk.get("name", "")
    start = chunk.get("start_line", "")
    end = chunk.get("end_line", "")
    content = chunk.get("content", "")

    if name and name != path:
        loc_info = f"  # {name}"
        if start and end:
            loc_info += f", lines {start}-{end}"
    else:
        loc_info = f"  # lines {start}-{end}" if start and end else ""

    return f"### {path}{loc_info}\n{content}"


class RAGContextBuilder:
    """
    Builds a code context string for the policy LLM given an issue.

    During RL training: retrieves chunks from the FAISS index and filters
    to the same repository as the training seed.

    During evaluation: can accept pre-specified file contents (oracle files)
    or fall back to retrieval.
    """

    def __init__(
        self,
        retriever: Optional[CodeRetriever] = None,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        top_k: int = 8,
    ):
        self._retriever = retriever
        self.max_context_tokens = max_context_tokens
        self.top_k = top_k

    @property
    def retriever(self) -> CodeRetriever:
        if self._retriever is None:
            self._retriever = get_retriever()
        return self._retriever

    def build_from_chunks(self, chunks: list[dict]) -> str:
        """
        Given a list of retrieved chunks (already ranked by relevance),
        concatenate them into a context string within the token budget.
        """
        seen_keys: set[str] = set()
        parts: list[str] = []
        used_tokens = 0

        for chunk in chunks:
            # Deduplicate: same file + content block
            key = f"{chunk.get('file_path')}::{chunk.get('start_line')}"
            if key in seen_keys:
                continue
            seen_keys.add(key)

            formatted = format_chunk(chunk)
            n_tokens = count_tokens(formatted)

            if used_tokens + n_tokens > self.max_context_tokens:
                # Try to include a truncated version if there's some room left
                remaining = self.max_context_tokens - used_tokens
                if remaining > 200:
                    # Truncate content to fit
                    words = formatted.split("\n")
                    truncated = []
                    t = 0
                    for w in words:
                        t += count_tokens(w)
                        if t > remaining - 20:
                            truncated.append("# ... (truncated)")
                            break
                        truncated.append(w)
                    parts.append("\n".join(truncated))
                break

            parts.append(formatted)
            used_tokens += n_tokens

        return "\n\n".join(parts)

    def build_from_file_contents(self, file_contents: dict[str, str]) -> str:
        """
        Build context from explicit {path: content} dict (used as fallback
        or when oracle files are provided during evaluation).
        """
        # Convert to chunk-like dicts for build_from_chunks
        chunks = [
            {
                "file_path": path,
                "name": path,
                "start_line": 1,
                "end_line": len(content.splitlines()),
                "content": content,
                "score": 1.0,
            }
            for path, content in file_contents.items()
        ]
        return self.build_from_chunks(chunks)

    def build(
        self,
        problem_statement: str,
        repo: str,
        file_contents: Optional[dict[str, str]] = None,
    ) -> str:
        """
        Main entry point. Retrieve top-k chunks for the issue and format them.

        Args:
            problem_statement : The GitHub issue text.
            repo              : Repository slug (owner/name).
            file_contents     : Optional {path: content} dict. If provided and
                                retrieval fails, this is used as fallback.

        Returns:
            Formatted code context string ready for the prompt.
        """
        chunks = self.retriever.retrieve_for_instance(
            problem_statement=problem_statement,
            repo=repo,
            file_contents=file_contents or {},
            top_k=self.top_k,
        )

        if not chunks and file_contents:
            logger.warning(f"Falling back to file_contents for repo={repo}")
            return self.build_from_file_contents(file_contents)

        return self.build_from_chunks(chunks)


# ─── Convenience function used by the training loop ───────────────────────────

def build_code_context(
    problem_statement: str,
    repo: str,
    file_contents: Optional[dict[str, str]] = None,
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    top_k: int = 8,
    index_path: str = "data/rag/faiss.index",
    chunk_meta_path: str = "data/rag/chunks.jsonl",
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> str:
    """
    Module-level convenience function wrapping RAGContextBuilder.
    Uses the process-wide retriever singleton.
    """
    builder = RAGContextBuilder(
        retriever=get_retriever(
            index_path=index_path,
            chunk_meta_path=chunk_meta_path,
            embed_model=embed_model,
        ),
        max_context_tokens=max_context_tokens,
        top_k=top_k,
    )
    return builder.build(
        problem_statement=problem_statement,
        repo=repo,
        file_contents=file_contents,
    )
