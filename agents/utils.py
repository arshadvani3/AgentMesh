"""Shared utilities for AgentMesh demo agents."""

from __future__ import annotations

import json
import re


def extract_json(text: str) -> dict | list:
    """Extract a JSON object or array from LLM output.

    Tries three strategies in order:
    1. Direct parse (LLM returned clean JSON)
    2. Strip markdown code fences then parse
    3. Regex extraction of first {...} or [...] block

    Raises ValueError if all strategies fail.
    """
    text = text.strip()

    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown fences (```json ... ``` or ``` ... ```)
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 3: grab first { ... } or [ ... ] block
    for pattern in (r"\{[\s\S]*\}", r"\[[\s\S]*\]"):
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

    raise ValueError(f"Could not extract JSON from LLM output: {text[:200]!r}")
