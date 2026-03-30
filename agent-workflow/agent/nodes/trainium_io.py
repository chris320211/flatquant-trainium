"""Shared helpers for Trainium skill-phase nodes."""

import json
import re
from typing import Any


def parse_llm_json_object(raw_text: str) -> dict[str, Any]:
    """Parse a JSON object from LLM output; strip optional markdown fences."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text.strip())
    try:
        out = json.loads(text)
        if isinstance(out, dict):
            return out
    except json.JSONDecodeError:
        pass
    return {"parse_error": True, "raw_excerpt": raw_text[:12_000]}
