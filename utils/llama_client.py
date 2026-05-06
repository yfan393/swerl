"""
llama_client.py
===============
Client for Llama 3.1 models running on a cluster via vLLM, TGI, or similar inference servers.

Supports:
  - Llama 1B, 3B, 8B models
  - Cluster-based inference via HTTP API
  - Compatible with OpenAI API format
  - Retry logic and error handling
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Union

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class LlamaClusterClient:
    """
    Client for Llama models running on a cluster inference server.

    Works with:
    - vLLM (text-generation-inference compatible)
    - Together AI
    - Fireworks
    - Any OpenAI-compatible endpoint
    """

    def __init__(
        self,
        endpoint_url: str,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        timeout: int = 120,
        max_retries: int = 3,
        api_key: Optional[str] = None,
    ):
        """
        Initialize Llama cluster client.

        Args:
            endpoint_url: Base URL of the cluster inference server
                         (e.g., "http://cluster.example.com:8000/v1")
            model_name: Model identifier on the cluster
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries on failure
            api_key: Optional API key if required by the cluster
        """
        self.endpoint_url = endpoint_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_key = api_key

        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

        # Stats tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        self.total_errors = 0
        self.total_time_seconds = 0.0

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def call(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """
        Make a single API call to the Llama cluster.

        Args:
            messages: Chat messages in OpenAI format
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop_sequences: Sequences to stop generation
            **kwargs: Additional API parameters

        Returns:
            Generated text response

        Raises:
            requests.RequestException: If API call fails
        """
        start_time = time.time()

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        if stop_sequences:
            payload["stop"] = stop_sequences

        payload.update(kwargs)

        try:
            response = self.session.post(
                f"{self.endpoint_url}/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()

            # Track usage
            usage = result.get("usage", {})
            self.total_input_tokens += usage.get("prompt_tokens", 0)
            self.total_output_tokens += usage.get("completion_tokens", 0)
            self.total_requests += 1
            elapsed = time.time() - start_time
            self.total_time_seconds += elapsed

            logger.debug(
                f"Request completed in {elapsed:.2f}s "
                f"({usage.get('completion_tokens', 0)} output tokens)"
            )

            # Extract text
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                raise ValueError("No choices in API response")

        except requests.RequestException as e:
            logger.error(f"Cluster API call failed: {e}")
            self.total_errors += 1
            raise

    async def call_async(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 1.0,
        **kwargs,
    ) -> str:
        """
        Async version of API call.

        Args:
            messages: Chat messages
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            top_p: Nucleus sampling
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.call(messages, max_tokens, temperature, top_p, **kwargs),
        )

    def batch_call(
        self,
        message_batches: List[List[Dict[str, str]]],
        max_tokens: int = 2048,
        delay_between_calls: float = 0.5,
        **kwargs,
    ) -> List[str]:
        """
        Make multiple API calls with rate limiting.

        Args:
            message_batches: List of message lists
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
                result = self.call(messages, max_tokens=max_tokens, **kwargs)
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
            (self.total_input_tokens + self.total_output_tokens) / self.total_time_seconds
            if self.total_time_seconds > 0
            else 0
        )

        return {
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "success_rate": (
                (self.total_requests - self.total_errors) / self.total_requests * 100
                if self.total_requests > 0
                else 0
            ),
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
_llama_clients: Dict[str, LlamaClusterClient] = {}


def get_llama_client(
    endpoint_url: str,
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    api_key: Optional[str] = None,
) -> LlamaClusterClient:
    """
    Get or create a cached Llama cluster client.

    Args:
        endpoint_url: Cluster inference server URL
        model_name: Model identifier
        api_key: Optional API key

    Returns:
        LlamaClusterClient instance
    """
    cache_key = f"{endpoint_url}:{model_name}"

    if cache_key not in _llama_clients:
        logger.info(f"Creating new Llama client for {model_name}")
        _llama_clients[cache_key] = LlamaClusterClient(
            endpoint_url=endpoint_url,
            model_name=model_name,
            api_key=api_key,
        )

    return _llama_clients[cache_key]


def clear_llama_cache() -> None:
    """Clear all cached Llama clients."""
    _llama_clients.clear()
    logger.info("Cleared Llama client cache")
