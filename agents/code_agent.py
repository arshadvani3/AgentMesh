"""Code Agent -- generates and optionally executes code examples via LLM + sandbox.

Registers on the mesh with the 'fetch_code' capability.

When input_data contains "execute": true, each generated snippet is run inside
an isolated subprocess with a 5-second hard timeout. Real stdout/stderr is
captured and returned alongside the code — so callers see whether the code
actually works, not just what it looks like.

Execution is opt-in (default False) so the agent is safe to use in contexts
where running arbitrary code is undesirable.

Run standalone:
    python -m agents.code_agent
"""

from __future__ import annotations

import ast
import asyncio
import logging
import os
import subprocess
import sys

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from agents.utils import extract_json
from sdk.agent import MeshAgent, capability

logger = logging.getLogger("agentmesh.agents.code")

_EXEC_TIMEOUT_SECONDS = 5


# ---------------------------------------------------------------------------
# Sandbox execution helper
# ---------------------------------------------------------------------------

def _run_in_sandbox(code: str) -> dict:
    """Execute a Python snippet in a subprocess.

    Uses subprocess (not exec/eval) so:
    - No shared state with the agent process
    - Real stdout/stderr captured cleanly
    - Hard timeout via subprocess.run timeout arg

    SECURITY NOTE: This provides process isolation only. The subprocess runs as
    the same OS user with full filesystem and network access. For production use,
    wrap with OS-level isolation (Docker --network none, bubblewrap, or nsjail)
    before enabling execute=true on untrusted input.

    Args:
        code: Python source to execute.

    Returns:
        Dict with stdout, stderr, exit_code, executed (bool), error (str | None).
    """
    # Syntax check first — fast fail, no subprocess overhead
    try:
        ast.parse(code)
    except SyntaxError as e:
        return {
            "executed": False,
            "stdout": "",
            "stderr": f"SyntaxError: {e}",
            "exit_code": 1,
            "error": f"Syntax error prevented execution: {e}",
        }

    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=_EXEC_TIMEOUT_SECONDS,
        )
        return {
            "executed": True,
            "stdout": result.stdout[:2000],   # cap at 2k chars — prevent huge outputs
            "stderr": result.stderr[:1000],
            "exit_code": result.returncode,
            "error": None,
        }
    except subprocess.TimeoutExpired:
        return {
            "executed": False,
            "stdout": "",
            "stderr": "",
            "exit_code": -1,
            "error": f"Execution timed out after {_EXEC_TIMEOUT_SECONDS}s",
        }
    except Exception as e:
        return {
            "executed": False,
            "stdout": "",
            "stderr": "",
            "exit_code": -1,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# CodeAgent
# ---------------------------------------------------------------------------

class CodeAgent(MeshAgent):
    """Generates annotated code examples, optionally running them in a sandbox."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.environ["GROQ_API_KEY"],
            temperature=0.2,
        )

    @capability(
        name="fetch_code",
        description=(
            "Generates relevant code examples for a given query and optionally executes "
            "them in a sandboxed subprocess, returning real stdout/stderr. "
            "Set execute=true in input to enable sandbox execution. "
            "Handles AI frameworks, Python patterns, library usage, and best practices."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What code patterns or examples to generate"},
                "language": {"type": "string", "description": "Programming language (default: python)"},
                "framework": {"type": "string", "description": "Specific framework or library to target"},
                "num_examples": {"type": "integer", "description": "Number of examples to return (default: 2)"},
                "execute": {
                    "type": "boolean",
                    "description": "Run each example in a sandboxed subprocess and return real output",
                    "default": False,
                },
            },
            "required": ["query"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "examples": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "code": {"type": "string"},
                            "explanation": {"type": "string"},
                            "language": {"type": "string"},
                            "executed": {"type": "boolean"},
                            "execution_result": {"type": "object"},
                        },
                    },
                },
                "summary": {"type": "string"},
            },
        },
        avg_latency_ms=10000,
        cost_per_call_usd=0.003,
    )
    async def fetch_code(self, input_data: dict) -> dict:
        """Generate code examples and optionally execute them.

        Args:
            input_data: Dict with query (required), language, framework,
                        num_examples, execute (all optional).

        Returns:
            Dict with examples (title, code, explanation, language,
            executed, execution_result) and summary.
        """
        query = input_data.get("query", "")
        language = input_data.get("language", "python")
        framework = input_data.get("framework", "")
        num_examples = input_data.get("num_examples", 2)
        should_execute = input_data.get("execute", False) and language.lower() == "python"

        framework_clause = f" using {framework}" if framework else ""

        system_prompt = (
            "You are an expert software engineer. "
            "Produce clean, well-commented, production-quality code examples with realistic imports. "
            "Each example must be self-contained and runnable. "
            "IMPORTANT: Respond with raw JSON only. No markdown fences, no explanation outside the JSON."
        )

        user_prompt = (
            f"Generate {num_examples} code example(s) for:\n"
            f"Query: {query}\n"
            f"Language: {language}{framework_clause}\n\n"
            "Return a JSON object with exactly these keys:\n"
            '- "examples": array where each item has "title", "code", "explanation", "language"\n'
            '- "summary": one paragraph describing the overall pattern\n\n'
            "Respond with raw JSON only."
        )

        logger.info(f"[CodeAgent] Generating code for: {query[:80]}...")
        response = await self._llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        raw = response.content.strip()

        examples = []
        summary = ""

        try:
            parsed = extract_json(raw)
            if isinstance(parsed, dict):
                examples = parsed.get("examples", [])
                summary = parsed.get("summary", "")
        except ValueError:
            logger.warning("[CodeAgent] Could not parse JSON from LLM response, using fallback")
            examples = [{"title": f"Example: {query[:60]}", "code": raw,
                         "explanation": "Generated by CodeAgent", "language": language}]
            summary = f"Code examples for: {query}"

        # Optionally execute each example in a sandbox
        if should_execute:
            loop = asyncio.get_running_loop()
            for ex in examples:
                code = ex.get("code", "")
                if not code:
                    ex["executed"] = False
                    ex["execution_result"] = {}
                    continue
                # Run in executor so we don't block the async event loop
                exec_result = await loop.run_in_executor(None, _run_in_sandbox, code)
                ex["executed"] = exec_result.pop("executed")
                ex["execution_result"] = exec_result
                if ex["executed"]:
                    logger.info(
                        f"[CodeAgent] Executed '{ex.get('title','')}': "
                        f"exit={exec_result.get('exit_code')}"
                    )
                else:
                    logger.info(
                        f"[CodeAgent] Skipped execution for '{ex.get('title','')}': "
                        f"{exec_result.get('error')}"
                    )
        else:
            for ex in examples:
                ex.setdefault("executed", False)
                ex.setdefault("execution_result", {})

        return {"examples": examples, "summary": summary}


# ---------------------------------------------------------------------------
# Run standalone
# ---------------------------------------------------------------------------

async def main():
    agent = CodeAgent(
        name="Code Agent",
        registry_url=os.environ.get("REGISTRY_URL", "http://localhost:8000"),
        ws_port=9003,
        tags=["code", "github", "programming", "execution"],
    )
    await agent.start()


if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    asyncio.run(main())
