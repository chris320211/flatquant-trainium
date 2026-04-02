"""
Anthropic Claude clients for the FlatQuant porting graph.

- Planning / analysis (ref_reader, validation summary): Claude Opus 4.6
- Code generation (codegen, registration): Claude Sonnet

Override models with ANTHROPIC_MODEL_PLANNING and ANTHROPIC_MODEL_CODEGEN.
"""

import os
from typing import Any

from langchain_anthropic import ChatAnthropic

# Defaults match Anthropic API model IDs; override via environment if names change.
_DEFAULT_PLANNING = "claude-opus-4-6"
_DEFAULT_CODEGEN = "claude-sonnet-4-20250514"


def get_planning_llm() -> ChatAnthropic:
    """Opus — pattern extraction from FlatQuant refs, validation narrative."""
    return ChatAnthropic(
        model=os.environ.get("ANTHROPIC_MODEL_PLANNING", _DEFAULT_PLANNING),
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        max_tokens=16_384,
        temperature=1,
    )


def get_codegen_llm() -> ChatAnthropic:
    """Sonnet — generating Python wrappers, calibration, patching."""
    return ChatAnthropic(
        model=os.environ.get("ANTHROPIC_MODEL_CODEGEN", _DEFAULT_CODEGEN),
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        max_tokens=32_768,
        temperature=0.3,
    )


def anthropic_text(response: Any) -> str:
    """Normalize AIMessage.content to a string (handles str or content-block lists)."""
    c = getattr(response, "content", response)
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for blk in c:
            if isinstance(blk, dict) and blk.get("type") == "text":
                parts.append(str(blk.get("text", "")))
            elif isinstance(blk, str):
                parts.append(blk)
            elif hasattr(blk, "text"):
                parts.append(str(getattr(blk, "text", "")))
        return "".join(parts)
    return str(c)
