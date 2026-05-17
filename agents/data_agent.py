"""Data Agent -- analyzes CSV/tabular data with real statistical computation.

Registers on the mesh with the 'analyze_csv' capability.

When a 'csv_path' or 'csv_url' key is present in input_data the agent:
  1. Loads the CSV (local path or HTTP fetch)
  2. Computes real statistics via pandas (if available) or numpy+csv fallback
  3. Passes the *computed numbers* to Groq for narrative synthesis

This ensures the final report contains verifiably real figures, not LLM
hallucinations about what the data "might" say.

When no CSV source is provided the agent falls back to LLM-only analysis
of whatever inline text is given, preserving backward compatibility.

Run standalone:
    python -m agents.data_agent
"""

from __future__ import annotations

import asyncio
import csv
import io
import ipaddress
import json
import logging
import os
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from agents.utils import extract_json
from sdk.agent import MeshAgent, capability

logger = logging.getLogger("agentmesh.agents.data")

# ---------------------------------------------------------------------------
# Real computation helpers
# ---------------------------------------------------------------------------

def _compute_stats_pandas(content: str) -> dict[str, Any]:
    """Compute statistics using pandas (preferred path)."""
    import pandas as pd  # noqa: PLC0415

    df = pd.read_csv(io.StringIO(content))
    shape = df.shape

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    describe = df[numeric_cols].describe().to_dict() if numeric_cols else {}

    # Round all floats to 2 dp for readability
    describe_clean: dict[str, Any] = {}
    for col, stats in describe.items():
        describe_clean[col] = {k: round(v, 2) if isinstance(v, float) else v for k, v in stats.items()}

    missing = df.isnull().sum().to_dict()

    # Top correlations (numeric only, skip trivial self-correlations)
    top_corr: list[dict[str, Any]] = []
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True)
        seen: set[frozenset] = set()
        for col_a in corr.columns:
            for col_b in corr.columns:
                if col_a == col_b:
                    continue
                pair = frozenset({col_a, col_b})
                if pair in seen:
                    continue
                seen.add(pair)
                val = corr.loc[col_a, col_b]
                if abs(val) > 0.3:
                    top_corr.append({"cols": [col_a, col_b], "r": round(val, 3)})
        top_corr.sort(key=lambda x: abs(x["r"]), reverse=True)
        top_corr = top_corr[:5]

    # Categorical summaries (value counts for object columns)
    cat_summaries: dict[str, Any] = {}
    for col in df.select_dtypes(include="object").columns:
        vc = df[col].value_counts().head(5).to_dict()
        cat_summaries[col] = vc

    return {
        "rows": shape[0],
        "cols": shape[1],
        "columns": df.columns.tolist(),
        "numeric_stats": describe_clean,
        "missing_values": missing,
        "top_correlations": top_corr,
        "categorical_summaries": cat_summaries,
        "engine": "pandas",
    }


def _compute_stats_numpy(content: str) -> dict[str, Any]:
    """Compute statistics using numpy + stdlib csv (fallback path)."""
    import numpy as np  # noqa: PLC0415

    reader = csv.DictReader(io.StringIO(content))
    rows = list(reader)
    if not rows:
        return {"rows": 0, "cols": 0, "columns": [], "engine": "numpy"}

    columns = list(rows[0].keys())

    # Identify numeric columns
    numeric_cols: list[str] = []
    for col in columns:
        try:
            float(rows[0][col].replace(",", ""))
            numeric_cols.append(col)
        except (ValueError, AttributeError):
            pass

    numeric_stats: dict[str, Any] = {}
    for col in numeric_cols:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[col].replace(",", "")))
            except (ValueError, AttributeError):
                pass
        if vals:
            arr = np.array(vals)
            numeric_stats[col] = {
                "count": len(arr),
                "mean": round(float(np.mean(arr)), 2),
                "std": round(float(np.std(arr)), 2),
                "min": round(float(np.min(arr)), 2),
                "25%": round(float(np.percentile(arr, 25)), 2),
                "50%": round(float(np.median(arr)), 2),
                "75%": round(float(np.percentile(arr, 75)), 2),
                "max": round(float(np.max(arr)), 2),
            }

    # Categorical value counts
    cat_summaries: dict[str, Any] = {}
    cat_cols = [c for c in columns if c not in numeric_cols]
    for col in cat_cols:
        counts: dict[str, int] = {}
        for r in rows:
            v = r.get(col, "")
            counts[v] = counts.get(v, 0) + 1
        # Top 5 by count
        top5 = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5])
        cat_summaries[col] = top5

    # Missing values
    missing: dict[str, int] = {}
    for col in columns:
        missing[col] = sum(1 for r in rows if not r.get(col, "").strip())

    return {
        "rows": len(rows),
        "cols": len(columns),
        "columns": columns,
        "numeric_stats": numeric_stats,
        "missing_values": missing,
        "top_correlations": [],
        "categorical_summaries": cat_summaries,
        "engine": "numpy",
    }


def compute_csv_stats(content: str) -> dict[str, Any]:
    """Compute real statistics from CSV text. Tries pandas first, falls back to numpy."""
    try:
        return _compute_stats_pandas(content)
    except Exception as e:
        logger.debug(f"[DataAgent] pandas unavailable ({e}), using numpy fallback")
        return _compute_stats_numpy(content)


_CSV_SAFE_DIR = Path(os.environ.get("CSV_SAFE_DIR", "/data/uploads")).resolve()

_PRIVATE_RANGES = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
]


def _is_ssrf_url(url: str) -> bool:
    """Return True if the URL should be blocked to prevent SSRF."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return True
    host = parsed.hostname or ""
    try:
        addr = ipaddress.ip_address(socket.gethostbyname(host))
        return any(addr in net for net in _PRIVATE_RANGES)
    except Exception:
        return True  # block if we can't resolve


async def _load_csv(input_data: dict) -> tuple[str | None, str | None]:
    """Load CSV content from csv_path (local file) or csv_url (HTTP).

    csv_path is restricted to CSV_SAFE_DIR (default /data/uploads).
    csv_url blocks private/internal IP ranges to prevent SSRF.

    Returns (content, source_label) or (None, None) if neither key present.
    """
    if "csv_path" in input_data:
        path = Path(input_data["csv_path"]).resolve()
        if not str(path).startswith(str(_CSV_SAFE_DIR)):
            logger.warning(f"[DataAgent] csv_path outside safe dir: {path}")
            return None, None
        try:
            with open(path) as f:
                return f.read(), str(path)
        except OSError as e:
            logger.warning(f"[DataAgent] Could not open csv_path={path}: {e}")
            return None, None

    if "csv_url" in input_data:
        url = input_data["csv_url"]
        if _is_ssrf_url(url):
            logger.warning(f"[DataAgent] Blocked SSRF attempt: {url}")
            return None, None
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=False) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.text, url
        except Exception as e:
            logger.warning(f"[DataAgent] Could not fetch csv_url={url}: {e}")
            return None, None

    return None, None


# ---------------------------------------------------------------------------
# DataAgent
# ---------------------------------------------------------------------------

class DataAgent(MeshAgent):
    """Performs real statistical computation + LLM narrative on tabular data."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._llm = ChatGroq(
            model=os.environ.get("DATA_AGENT_MODEL", "llama-3.3-70b-versatile"),
            api_key=os.environ["GROQ_API_KEY"],
            temperature=0.1,
        )

    @capability(
        name="analyze_csv",
        description=(
            "Real statistical analysis of CSV data. "
            "Accepts a local csv_path or remote csv_url to load actual data, "
            "computes genuine statistics (mean, std, distributions, correlations), "
            "then synthesizes an LLM narrative grounded in those real numbers. "
            "Also accepts inline data or a query for LLM-only analysis as fallback."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to analyze or find in the data"},
                "csv_path": {"type": "string", "description": "Local file path to a CSV"},
                "csv_url": {"type": "string", "description": "HTTP URL of a CSV file"},
                "data": {"type": "string", "description": "Inline CSV text or dataset description"},
                "analysis_type": {
                    "type": "string",
                    "enum": ["summary", "trends", "comparison", "anomalies"],
                },
            },
            "required": ["query"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "analysis": {"type": "string"},
                "key_findings": {"type": "array", "items": {"type": "string"}},
                "data_summary": {"type": "string"},
                "computed_stats": {"type": "object"},
            },
        },
        avg_latency_ms=5000,
        cost_per_call_usd=0.002,
    )
    async def analyze_csv(self, input_data: dict) -> dict:
        """Run real statistical computation + LLM narrative on the provided data.

        Args:
            input_data: Dict with query (required), plus csv_path, csv_url,
                        data (optional), analysis_type (optional).

        Returns:
            Dict with analysis narrative, key findings, data summary, and
            computed_stats (real numbers, present when CSV was loaded).
        """
        query = input_data.get("query", "")
        analysis_type = input_data.get("analysis_type", "summary")

        # --- Attempt real computation ---
        csv_content, source = await _load_csv(input_data)
        computed_stats: dict[str, Any] = {}
        stats_context = ""

        if csv_content:
            try:
                computed_stats = compute_csv_stats(csv_content)
                engine = computed_stats.get("engine", "unknown")
                logger.info(
                    f"[DataAgent] Loaded CSV ({source}): "
                    f"{computed_stats['rows']} rows × {computed_stats['cols']} cols "
                    f"via {engine}"
                )
                # Build a concise stats summary to inject into the LLM prompt
                stats_context = (
                    f"REAL COMPUTED STATISTICS (use these exact numbers in your analysis):\n"
                    f"{json.dumps(computed_stats, indent=2)}\n\n"
                )
            except Exception as e:
                logger.warning(f"[DataAgent] Stats computation failed: {e}")

        # Fall back to inline data string if no CSV loaded
        if not stats_context:
            data = input_data.get("data", "No data provided.")
            stats_context = f"Dataset description / inline data:\n{data}\n\n"

        system_prompt = (
            "You are a data analyst. Given real computed statistics and a query, "
            "write a precise, insightful analysis that cites the actual numbers. "
            "Never invent statistics — only use the numbers provided. "
            "IMPORTANT: Respond with raw JSON only. No markdown fences, no explanation outside the JSON."
        )

        user_prompt = (
            f"Analysis type: {analysis_type}\n"
            f"Query: {query}\n\n"
            f"{stats_context}"
            "Return a JSON object with exactly these keys:\n"
            '- "analysis": comprehensive narrative citing real numbers (string)\n'
            '- "key_findings": list of 3-5 specific findings with numbers (array of strings)\n'
            '- "data_summary": one-sentence dataset description (string)\n\n'
            "Respond with raw JSON only."
        )

        logger.info(f"[DataAgent] Synthesizing narrative for: {query[:80]}...")
        response = await self._llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        raw = response.content.strip()

        try:
            parsed = extract_json(raw)
            if isinstance(parsed, dict):
                return {
                    "analysis": parsed.get("analysis", raw),
                    "key_findings": parsed.get("key_findings", []),
                    "data_summary": parsed.get("data_summary", ""),
                    "computed_stats": computed_stats,
                }
        except ValueError:
            logger.warning("[DataAgent] Could not parse JSON from LLM response, returning raw text")

        return {
            "analysis": raw,
            "key_findings": [],
            "data_summary": f"Analysis of: {query}",
            "computed_stats": computed_stats,
        }


# ---------------------------------------------------------------------------
# Run standalone
# ---------------------------------------------------------------------------

async def main():
    agent = DataAgent(
        name="Data Agent",
        registry_url=os.environ.get("REGISTRY_URL", "http://localhost:8000"),
        ws_port=9002,
        tags=["data", "analysis", "csv", "statistics"],
    )
    await agent.start()


if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    asyncio.run(main())
