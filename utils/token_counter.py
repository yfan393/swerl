"""
token_counter.py
================
Token counting utilities for estimating model costs and memory.

Handles:
  - Approximate token counting
  - Model-specific token limits
  - Memory estimation
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Approximate characters per token (varies by tokenizer)
CHARS_PER_TOKEN = {
    "tiktoken": 4,  # GPT models
    "transformers": 4,  # General transformer models
    "llamatokenizer": 3.5,  # Llama models (slightly denser)
}

# Token limits for common models
MODEL_TOKEN_LIMITS = {
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-turbo": 128000,
    "gpt-3.5-turbo": 4096,
    "claude-3-sonnet": 200000,
    "claude-3-opus": 200000,
    "llama-2-7b": 4096,
    "llama-2-13b": 4096,
    "llama-2-70b": 4096,
    "llama-3-8b": 8192,
    "llama-3-70b": 8192,
    "qwen-7b": 32768,
    "qwen-14b": 32768,
    "mistral-7b": 8192,
    "mistral-large": 32000,
    "codellama-7b": 16384,
    "codellama-13b": 16384,
    "codellama-34b": 16384,
}


def count_tokens_approximate(text: str, method: str = "chars", chars_per_token: int = 4) -> int:
    """
    Approximate token count using character counting.

    Fast but inaccurate method. Good for rough estimation.

    Args:
        text: Text to count
        method: 'chars' for character-based, 'words' for word-based
        chars_per_token: Average characters per token

    Returns:
        Approximate token count
    """
    if method == "chars":
        return max(1, len(text) // chars_per_token)
    elif method == "words":
        words = len(text.split())
        return max(1, words // 1.3)  # Average 1.3 words per token
    else:
        raise ValueError(f"Unknown method: {method}")


def count_tokens_tiktoken(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count tokens using tiktoken (accurate for OpenAI models).

    Args:
        text: Text to count
        model: Model name for encoding

    Returns:
        Exact token count
    """
    try:
        import tiktoken
    except ImportError:
        logger.warning("tiktoken not installed, using approximate counting")
        return count_tokens_approximate(text)

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logger.debug(f"Unknown model {model}, using gpt2 encoding")
        encoding = tiktoken.get_encoding("gpt2")

    tokens = encoding.encode(text)
    return len(tokens)


def count_tokens_transformers(text: str, model_name: str = "meta-llama/Llama-2-7b-hf") -> int:
    """
    Count tokens using transformers library tokenizer.

    Args:
        text: Text to count
        model_name: Model name for tokenizer

    Returns:
        Exact token count
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        logger.warning("transformers not installed, using approximate counting")
        return count_tokens_approximate(text)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokens = tokenizer.encode(text, add_special_tokens=True)
        return len(tokens)
    except Exception as e:
        logger.warning(f"Failed to load tokenizer for {model_name}: {e}")
        return count_tokens_approximate(text)


def count_tokens(
    text: str,
    tokenizer_type: str = "approximate",
    model: Optional[str] = None,
) -> int:
    """
    Count tokens in text.

    Args:
        text: Text to tokenize
        tokenizer_type: 'approximate', 'tiktoken', or 'transformers'
        model: Model name (required for exact counting)

    Returns:
        Token count
    """
    if tokenizer_type == "approximate":
        return count_tokens_approximate(text)
    elif tokenizer_type == "tiktoken":
        return count_tokens_tiktoken(text, model or "gpt-3.5-turbo")
    elif tokenizer_type == "transformers":
        return count_tokens_transformers(text, model or "meta-llama/Llama-2-7b-hf")
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


def estimate_tokens_messages(
    messages: list[dict],
    tokenizer_type: str = "approximate",
    model: Optional[str] = None,
) -> int:
    """
    Estimate tokens in a chat message list.

    Args:
        messages: List of {"role": "...", "content": "..."}
        tokenizer_type: Type of tokenizer
        model: Model name

    Returns:
        Estimated token count (including overhead)
    """
    total = 0
    for msg in messages:
        # Add overhead for role and formatting
        total += 4  # Role and formatting overhead
        total += count_tokens(msg.get("content", ""), tokenizer_type, model)

    # Add conversation overhead
    total += 2
    return total


def get_model_token_limit(model_name: str) -> int:
    """
    Get context window size for a model.

    Args:
        model_name: Model name or identifier

    Returns:
        Maximum tokens in context window

    Raises:
        ValueError: If model not found
    """
    # Try exact match
    if model_name in MODEL_TOKEN_LIMITS:
        return MODEL_TOKEN_LIMITS[model_name]

    # Try case-insensitive substring match
    model_lower = model_name.lower()
    for known_model, limit in MODEL_TOKEN_LIMITS.items():
        if known_model in model_lower or model_lower in known_model:
            logger.debug(f"Matched {model_name} to {known_model}")
            return limit

    # Default to 4k if unknown
    logger.warning(f"Unknown model {model_name}, assuming 4k token limit")
    return 4096


def estimate_memory_mb(
    num_tokens: int,
    model_size_b: float,
    bits_per_param: int = 16,
) -> float:
    """
    Estimate GPU memory needed for inference.

    Args:
        num_tokens: Number of tokens in context
        model_size_b: Model size in billions of parameters
        bits_per_param: Bits per parameter (16 for float16, 8 for int8, etc.)

    Returns:
        Estimated memory in MB
    """
    # Model weights
    model_params = model_size_b * 1e9
    weights_mb = (model_params * bits_per_param) / (8 * 1024 * 1024)

    # KV cache (attention memory)
    # Rough estimate: 2 * (num_layers * num_heads * head_dim * seq_len * batch_size)
    # Simplified: ~4 bytes per token per 1B parameters
    kv_cache_mb = (model_size_b * num_tokens * 4) / 1024

    # Safety margin
    total = (weights_mb + kv_cache_mb) * 1.2

    return total


def fits_in_memory(
    num_tokens: int,
    model_size_b: float,
    gpu_memory_gb: float,
    bits_per_param: int = 16,
) -> bool:
    """
    Check if inference fits in available GPU memory.

    Args:
        num_tokens: Number of tokens
        model_size_b: Model size in billions
        gpu_memory_gb: Available GPU memory in GB
        bits_per_param: Bits per parameter

    Returns:
        True if it fits
    """
    needed_mb = estimate_memory_mb(num_tokens, model_size_b, bits_per_param)
    available_mb = gpu_memory_gb * 1024
    return needed_mb <= available_mb
