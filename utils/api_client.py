"""
api_client.py
=============
OpenAI-compatible API client for code generation and LLM calls.

Handles:
  - API calls with retry logic
  - Batch requests
  - Error handling
  - Cost tracking
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class OpenAIClient:
    """OpenAI-compatible API client (works with OpenAI, local endpoints, etc.)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        """
        Initialize API client.

        Args:
            api_key: API key (defaults to OPENAI_API_KEY env var)
            base_url: API base URL (defaults to OpenAI URL)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "")).rstrip("/")
        if not self.base_url:
            self.base_url = "https://api.openai.com/v1"

        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

        # Tracking
        self.total_tokens_used = 0
        self.total_cost = 0.0

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def call(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 1.0,
        **kwargs,
    ) -> str:
        """
        Make a single API call.

        Args:
            messages: Chat messages
            model: Model name
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional API parameters

        Returns:
            Generated text response

        Raises:
            requests.RequestException: If API call fails
        """
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            **kwargs,
        }

        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()

            # Track tokens
            if "usage" in result:
                self.total_tokens_used += result["usage"].get("total_tokens", 0)
                # Simple cost estimate for GPT-3.5: $0.0005 per 1k tokens
                self.total_cost += result["usage"].get("total_tokens", 0) * 0.0005 / 1000

            # Extract text
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                raise ValueError("No choices in API response")

        except requests.RequestException as e:
            logger.error(f"API call failed: {e}")
            raise

    async def call_async(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """
        Async version of API call.

        Args:
            messages: Chat messages
            model: Model name
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.call(messages, model, max_tokens, temperature, **kwargs),
        )

    def batch_call(
        self,
        message_batches: List[List[Dict[str, str]]],
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 2048,
        max_concurrent: int = 5,
        delay_between_calls: float = 0.1,
        **kwargs,
    ) -> List[str]:
        """
        Make multiple API calls with rate limiting.

        Args:
            message_batches: List of message lists
            model: Model name
            max_tokens: Max tokens per call
            max_concurrent: Max concurrent requests
            delay_between_calls: Delay between calls (seconds)
            **kwargs: Additional parameters

        Returns:
            List of responses (one per batch)
        """
        results = []
        for i, messages in enumerate(message_batches):
            if i > 0:
                time.sleep(delay_between_calls)

            try:
                result = self.call(
                    messages,
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

    async def batch_call_async(
        self,
        message_batches: List[List[Dict[str, str]]],
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 2048,
        max_concurrent: int = 5,
        **kwargs,
    ) -> List[str]:
        """
        Async batch calls with concurrency control.

        Args:
            message_batches: List of message lists
            model: Model name
            max_tokens: Max tokens per call
            max_concurrent: Max concurrent requests
            **kwargs: Additional parameters

        Returns:
            List of responses
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def call_with_semaphore(messages):
            async with semaphore:
                return await self.call_async(messages, model, max_tokens, **kwargs)

        tasks = [call_with_semaphore(messages) for messages in message_batches]
        return await asyncio.gather(*tasks)

    def set_api_key(self, api_key: str) -> None:
        """Update API key."""
        self.api_key = api_key
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def set_base_url(self, base_url: str) -> None:
        """Update API base URL."""
        self.base_url = base_url.rstrip("/")

    def get_stats(self) -> Dict[str, Any]:
        """Get API usage statistics."""
        return {
            "total_tokens": self.total_tokens_used,
            "estimated_cost": f"${self.total_cost:.4f}",
            "average_tokens_per_call": (
                self.total_tokens_used // 1 if self.total_tokens_used > 0 else 0
            ),
        }

    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self.total_tokens_used = 0
        self.total_cost = 0.0


# Helper functions for easy access
_default_client = None


def get_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> OpenAIClient:
    """Get or create default API client."""
    global _default_client
    if _default_client is None:
        _default_client = OpenAIClient(api_key=api_key, base_url=base_url)
    return _default_client


def call_api(
    messages: List[Dict[str, str]],
    model: str = "gpt-3.5-turbo",
    max_tokens: int = 2048,
    **kwargs,
) -> str:
    """Quick API call using default client."""
    client = get_client()
    return client.call(messages, model, max_tokens, **kwargs)


def parse_thinking_output(output: str) -> tuple:
    """
    Parse <think> and <solution> blocks from model output.

    Args:
        output: Model output text

    Returns:
        Tuple of (thinking_text, solution_text)
    """
    try:
        think_start = output.find("<think>")
        think_end = output.find("</think>")
        solution_start = output.find("<solution>")
        solution_end = output.find("</solution>")

        if think_start < 0 or think_end < 0 or solution_start < 0 or solution_end < 0:
            raise ValueError("Missing required tags")

        thinking = output[think_start + 7:think_end].strip()
        solution = output[solution_start + 10:solution_end].strip()

        return thinking, solution
    except Exception as e:
        logger.error(f"Failed to parse output: {e}")
        return "", output


def parse_thinking_output(output: str) -> tuple:
    """
    Parse <think> and <solution> blocks from model output.

    Args:
        output: Model output text

    Returns:
        Tuple of (thinking_text, solution_text)
    """
    try:
        think_start = output.find("<think>")
        think_end = output.find("</think>")
        solution_start = output.find("<solution>")
        solution_end = output.find("</solution>")

        if think_start < 0 or think_end < 0 or solution_start < 0 or solution_end < 0:
            raise ValueError("Missing required tags")

        thinking = output[think_start + 7:think_end].strip()
        solution = output[solution_start + 10:solution_end].strip()

        return thinking, solution
    except Exception as e:
        logger.error(f"Failed to parse output: {e}")
        return "", output
