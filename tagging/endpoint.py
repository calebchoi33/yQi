import os
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

_CLIENT: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """Return a singleton OpenAI client using OPENAI_API_KEY."""
    global _CLIENT
    if _CLIENT is None:
        api_key = os.environ["OPENAI_API_KEY"]
        _CLIENT = OpenAI(api_key=api_key)
    return _CLIENT


def call_chat(
    messages: List[Dict[str, Any]],
    *,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = "auto",
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> Any:
    """
    Call the Chat Completions API with optional tools support.

    Args:
        messages: List of role/content dicts per Chat Completions spec.
        tools: Optional list of tool schemas (function tools).
        tool_choice: Tool choice policy (e.g., "auto" or specific function name).
        model: Model name; defaults to env OPENAI_MODEL or 'gpt-4o-mini'.
        temperature: Sampling temperature.

    Returns:
        The API response object as returned by the SDK.
    """

    client = _get_client()
    model = model or os.environ["OPENAI_MODEL"]

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if tools:
        kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice

    return client.chat.completions.create(**kwargs)


def chat_with_retry(
    messages: List[Dict[str, Any]],
    *,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = "auto",
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_retries: int = 3,
):
    """Call chat API with exponential backoff retry.

    Retries up to max_retries on exceptions (e.g., rate limits), waiting 1, 2, 4 ... seconds.
    """
    attempt = 0
    while True:
        try:
            return call_chat(
                messages,
                tools=tools,
                tool_choice=tool_choice,
                model=model,
                temperature=temperature,
            )
        except Exception as e:
            if attempt >= max_retries:
                raise e

            delay = 2**attempt

            print(f"--------------------------------")
            print(f"Error: {e}")
            print(f"Attempt {attempt + 1}/{max_retries}")
            print(f"Retrying in {delay} seconds...")
            print(f"--------------------------------")

            time.sleep(delay)
            attempt += 1
