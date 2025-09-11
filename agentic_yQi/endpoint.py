import os
from typing import Any, Dict, List, Optional

try:
    # OpenAI Python SDK v1.x
    from openai import OpenAI
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "The `openai` package is required. Install with: pip install openai"
    ) from e


_CLIENT: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """Return a singleton OpenAI client using OPENAI_KEY or OPENAI_API_KEY."""
    global _CLIENT
    if _CLIENT is None:
        api_key = os.environ.get("OPENAI_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "Missing API key. Set OPENAI_KEY or OPENAI_API_KEY in your environment."
            )
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
    model = model or os.environ.get("OPENAI_MODEL", "gpt-4.1")

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }

    if tools:
        kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

    return client.chat.completions.create(**kwargs)


def first_message_text(response: Any) -> str:
    """Extract the first assistant message text content, if available."""
    try:
        msg = response.choices[0].message
        return (msg.content or "").strip() if msg else ""
    except Exception:
        return ""

