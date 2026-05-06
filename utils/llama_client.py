"""
llama_client.py
===============
OpenAI-compatible client for Llama 3.1 models running on vLLM cluster.

Features:
  - Uses official OpenAI Python package
  - Works with vLLM's OpenAI-compatible API
  - Environment variable support (port_vllm)
  - Retry logic and error handling
  - Usage statistics tracking
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI, AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class LlamaClusterClient:
    """
    OpenAI-compatible client for Llama models on vLLM.

    Uses the official OpenAI Python package to communicate with vLLM's
    OpenAI-compatible API endpoint.

    Example:
        client = LlamaClusterClient(port=8000)
        response = client.call(messages, model="meta-llama/Llama-3.1-8B-Instruct")
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        api_key: str = "EMPTY",
        timeout: int = 120,
        max_retries: int = 3,
    ):
        """
        Initialize vLLM client.

        Args:
            host: Server hostname (default: 127.0.0.1)
            port: Server port. If None, reads from port_vllm env var
            api_key: API key (default: "EMPTY" for vLLM)
            timeout: Request timeout in seconds
            max_retries: Max retries on failure
        """
        # Get port from environment variable or use provided value
        if port is None:
            port = int(os.environ.get("port_vllm", 8000))

        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}/v1"
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=timeout,
        )

        # Async client
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=timeout,
        )

        logger.info(f"Initialized vLLM client: {self.base_url}")

        # Stats tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        self.total_errors = 0
        self.total_time_seconds = 0.0

    def get_available_models(self) -> List[str]:
        """
        Get list of available models from vLLM server.

        Returns:
            List of model IDs
        """
        try:
            models = self.client.models.list()
            model_ids = [model.id for model in models.data]
            logger.info(f"Available models: {model_ids}")
            return model_ids
        except Exception as e:
            logger.error(f"Failed to fetch models: {e}")
            return []

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def call(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """
        Make a chat completion request to vLLM.

        Args:
            messages: Chat messages in OpenAI format
            model: Model ID (if None, uses first available model)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            **kwargs: Additional parameters (frequency_penalty, presence_penalty, etc.)

        Returns:
            Generated text response

        Raises:
            Exception: If API call fails after retries
        """
        start_time = time.time()

        try:
            # Use first available model if not specified
            if model is None:
                available = self.get_available_models()
                if not available:
                    raise ValueError("No models available on vLLM server")
                model = available[0]
                logger.debug(f"Using default model: {model}")

            # Make API call
            response = self.client.chat.completions.create(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                **kwargs,
            )

            # Extract response
            text = response.choices[0].message.content

            # Track usage
            if response.usage:
                self.total_input_tokens += response.usage.prompt_tokens
                self.total_output_tokens += response.usage.completion_tokens

            self.total_requests += 1
            elapsed = time.time() - start_time
            self.total_time_seconds += elapsed

            logger.debug(
                f"Request completed in {elapsed:.2f}s "
                f"({response.usage.completion_tokens if response.usage else 0} output tokens)"
            )

            return text

        except Exception as e:
            logger.error(f"vLLM API call failed: {e}")
            self.total_errors += 1
            raise

    async def call_async(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 1.0,
        **kwargs,
    ) -> str:
        """
        Async version of chat completion.

        Args:
            messages: Chat messages
            model: Model ID
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            top_p: Nucleus sampling
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        if model is None:
            available = self.get_available_models()
            if not available:
                raise ValueError("No models available")
            model = available[0]

        start_time = time.time()

        try:
            response = await self.async_client.chat.completions.create(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            )

            text = response.choices[0].message.content

            if response.usage:
                self.total_input_tokens += response.usage.prompt_tokens
                self.total_output_tokens += response.usage.completion_tokens

            self.total_requests += 1
            elapsed = time.time() - start_time
            self.total_time_seconds += elapsed

            return text

        except Exception as e:
            logger.error(f"vLLM async call failed: {e}")
            self.total_errors += 1
            raise

    def batch_call(
        self,
        message_batches: List[List[Dict[str, str]]],
        model: Optional[str] = None,
        max_tokens: int = 2048,
        delay_between_calls: float = 0.1,
        **kwargs,
    ) -> List[str]:
        """
        Make multiple API calls with rate limiting.

        Args:
            message_batches: List of message lists
            model: Model ID
            max_tokens: Max tokens per call
            delay_between_calls: Delay between calls (seconds)
            **kwargs: Additional parameters

        Returns:
            List of responses
        """
        results = []
        for i, messages in enumerate(message_batches):
            if i > 0:
                time.sleep(delay_between_calls)

            try:
                result = self.call(
                    messages=messages,
                    model=model,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                results.append(result)
                logger.debug(f"Batch {i+1}/{len(message_batches)} completed")
            except Exception as e:
                logger.error(f"Batch {i} failed: {e}")
                results.append("")  # Empty result on failure

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get API usage statistics."""
        avg_time = (
            self.total_time_seconds / self.total_requests
            if self.total_requests > 0
            else 0
        )
        tokens_per_second = (
            (self.total_input_tokens + self.total_output_tokens)
            / self.total_time_seconds
            if self.total_time_seconds > 0
            else 0
        )
        success_rate = (
            (self.total_requests - self.total_errors) / self.total_requests * 100
            if self.total_requests > 0
            else 0
        )

        return {
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "success_rate": success_rate,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "average_time_per_request_seconds": avg_time,
            "tokens_per_second": tokens_per_second,
            "total_time_seconds": self.total_time_seconds,
        }

    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        self.total_errors = 0
        self.total_time_seconds = 0.0


# Global client cache
_llama_clients: Dict[tuple, LlamaClusterClient] = {}


def get_llama_client(
    host: str = "127.0.0.1",
    port: Optional[int] = None,
    api_key: str = "EMPTY",
) -> LlamaClusterClient:
    """
    Get or create a cached vLLM client.

    Args:
        host: Server hostname
        port: Server port (uses port_vllm env var if None)
        api_key: API key

    Returns:
        LlamaClusterClient instance
    """
    # Use environment variable if port not provided
    if port is None:
        port = int(os.environ.get("port_vllm", 8000))

    cache_key = (host, port)

    if cache_key not in _llama_clients:
        logger.info(f"Creating new vLLM client: {host}:{port}")
        _llama_clients[cache_key] = LlamaClusterClient(
            host=host,
            port=port,
            api_key=api_key,
        )

    return _llama_clients[cache_key]


def clear_llama_cache() -> None:
    """Clear all cached vLLM clients."""
    _llama_clients.clear()
    logger.info("Cleared vLLM client cache")
