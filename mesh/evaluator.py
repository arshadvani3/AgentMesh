"""OutputEvaluator — scores agent outputs to drive trust updates.

Instead of hardcoding quality=0.8 for every completed task, this evaluator
produces a real 0.0–1.0 score per capability type:

  - analyze_csv   : deterministic — did reported numbers match computed stats?
  - fetch_code    : deterministic — did the code execute without error?
  - write_report  : LLM-as-judge via Groq
  - web_search    : did results come back non-empty?
  - default       : LLM-as-judge with generic rubric

Scores feed directly into the ELO trust update in sdk/agent.py.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("agentmesh.evaluator")


class EvalResult:
    """Result of evaluating a single agent output."""

    def __init__(
        self,
        score: float,
        reasoning: str,
        checks: dict[str, Any] | None = None,
    ):
        self.score = max(0.0, min(1.0, score))
        self.reasoning = reasoning
        self.checks: dict[str, Any] = checks or {}

    def __repr__(self) -> str:
        return f"EvalResult(score={self.score:.2f}, reasoning={self.reasoning!r})"


class OutputEvaluator:
    """Scores agent outputs 0.0–1.0 using capability-specific heuristics.

    Deterministic checks are used wherever possible. LLM-as-judge is only
    used for open-ended text outputs (write_report, unknown capabilities).
    """

    def __init__(self, groq_api_key: str | None = None):
        self._api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        self._llm: Any = None  # langchain_groq.ChatGroq, lazily initialised

    def _get_llm(self) -> Any:
        if self._llm is not None:
            return self._llm
        if not self._api_key:
            return None
        try:
            from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415
            from langchain_groq import ChatGroq  # noqa: PLC0415
            from pydantic import SecretStr  # noqa: PLC0415
            self._llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=SecretStr(self._api_key),
                temperature=0.0,
            )
            self._llm._hm = HumanMessage
            self._llm._sm = SystemMessage
        except Exception as e:
            logger.warning(f"OutputEvaluator: could not init LLM ({e})")
        return self._llm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        task_capability: str,
        input_data: dict,
        output: dict,
        context: dict | None = None,
    ) -> EvalResult:
        """Score an agent output for a given capability.

        Args:
            task_capability: The capability name (e.g. 'analyze_csv').
            input_data: The original task input dict.
            output: The agent's output dict.
            context: Optional extra context (e.g. computed_stats for grounding).

        Returns:
            EvalResult with score in [0.0, 1.0].
        """
        try:
            if task_capability == "analyze_csv":
                return self._eval_csv_analysis(output, context)
            if task_capability == "fetch_code":
                return self._eval_code(output)
            if task_capability == "web_search":
                return self._eval_web_search(output)
            if task_capability == "write_report":
                return await self._eval_report_llm(input_data, output)
            return await self._eval_generic_llm(task_capability, input_data, output)
        except Exception as e:
            logger.warning(f"OutputEvaluator: evaluation failed for '{task_capability}': {e}")
            return EvalResult(score=0.5, reasoning=f"Evaluation error: {e}")

    # ------------------------------------------------------------------
    # Deterministic evaluators
    # ------------------------------------------------------------------

    def _eval_csv_analysis(
        self, output: dict, context: dict | None
    ) -> EvalResult:
        """Score CSV analysis by checking: non-empty analysis + optional number grounding."""
        checks: dict[str, Any] = {}
        scores: list[float] = []

        # Check 1: analysis text exists and is substantive
        analysis = output.get("analysis", "")
        has_content = isinstance(analysis, str) and len(analysis) > 100
        checks["has_content"] = has_content
        scores.append(1.0 if has_content else 0.0)

        # Check 2: key findings present
        findings = output.get("key_findings", [])
        has_findings = isinstance(findings, list) and len(findings) > 0
        checks["has_findings"] = has_findings
        scores.append(1.0 if has_findings else 0.2)

        # Check 3: grounding — did reported mean match computed stats?
        grounding = 0.7  # default: neutral if no context to check against
        if context and "computed_stats" in context:
            computed = context["computed_stats"]
            numeric = computed.get("numeric_stats", {})
            # Check if analysis mentions any numeric value that appears in computed stats
            mentioned = 0
            checkable = 0
            for _col, stats in numeric.items():
                mean_val = stats.get("mean")
                if mean_val is not None:
                    checkable += 1
                    # Fuzzy check: does the analysis mention a number close to the mean?
                    if analysis and _number_mentioned(analysis, mean_val, tolerance=0.1):
                        mentioned += 1
            if checkable > 0:
                grounding = mentioned / checkable
        checks["grounding"] = grounding
        scores.append(grounding)

        score = sum(scores) / len(scores)
        reasoning = (
            f"CSV analysis: content={'ok' if has_content else 'missing'}, "
            f"findings={'ok' if has_findings else 'missing'}, "
            f"grounding={grounding:.2f}"
        )
        return EvalResult(score=score, reasoning=reasoning, checks=checks)

    def _eval_code(self, output: dict) -> EvalResult:
        """Score code output based on execution results."""
        checks: dict[str, Any] = {}
        examples = output.get("examples", [])

        if not examples:
            return EvalResult(
                score=0.3,
                reasoning="No code examples returned",
                checks={"examples_count": 0},
            )

        executed_count = 0
        success_count = 0
        syntax_error_count = 0

        for ex in examples:
            exec_result = ex.get("execution_result")
            if exec_result is None:
                # Code not executed (execute=False) — partial credit
                continue
            executed_count += 1
            exit_code = exec_result.get("exit_code", -1)
            stderr = exec_result.get("stderr", "")
            if exit_code == 0:
                success_count += 1
            elif "SyntaxError" in stderr:
                syntax_error_count += 1

        checks["examples_count"] = len(examples)
        checks["executed_count"] = executed_count
        checks["success_count"] = success_count
        checks["syntax_errors"] = syntax_error_count

        if executed_count == 0:
            # Not executed — judge on presence of code text alone
            has_code = all(bool(ex.get("code", "").strip()) for ex in examples)
            score = 0.65 if has_code else 0.2
            return EvalResult(
                score=score,
                reasoning=f"Code generated but not executed ({len(examples)} examples)",
                checks=checks,
            )

        # Full execution scoring
        syntax_penalty = syntax_error_count / executed_count * 0.4
        success_rate = success_count / executed_count
        score = max(0.1, success_rate - syntax_penalty)
        reasoning = (
            f"Code execution: {success_count}/{executed_count} succeeded, "
            f"{syntax_error_count} syntax errors"
        )
        return EvalResult(score=score, reasoning=reasoning, checks=checks)

    def _eval_web_search(self, output: dict) -> EvalResult:
        """Score web search by checking results are non-empty and structured."""
        results = output.get("results", "")
        checks: dict[str, Any] = {}

        if not results:
            return EvalResult(score=0.1, reasoning="Empty search results", checks={"empty": True})

        content_len = len(str(results))
        checks["content_length"] = content_len
        checks["has_source"] = bool(output.get("source"))

        # Longer, richer results score higher, up to a cap
        score = min(1.0, 0.4 + (content_len / 2000) * 0.5)
        if output.get("source"):
            score = min(1.0, score + 0.1)

        return EvalResult(
            score=score,
            reasoning=f"Web search: {content_len} chars returned",
            checks=checks,
        )

    # ------------------------------------------------------------------
    # LLM-as-judge evaluators
    # ------------------------------------------------------------------

    async def _eval_report_llm(self, input_data: dict, output: dict) -> EvalResult:
        """Score a written report using LLM-as-judge."""
        query = input_data.get("query", input_data.get("topic", ""))
        report = output.get("report", str(output))

        if not report or len(report) < 50:
            return EvalResult(score=0.1, reasoning="Report too short or empty")

        # Deterministic floor: if report is long and mentions the query topic, it's at least ok
        base_score = min(0.7, 0.3 + len(report) / 3000)

        llm = self._get_llm()
        if llm is None:
            return EvalResult(
                score=base_score,
                reasoning=f"No LLM available; length-based score: {base_score:.2f}",
            )

        prompt = (
            f"Query: {query}\n\n"
            f"Report (first 800 chars):\n{report[:800]}\n\n"
            "Rate this report 0.0–1.0 on: relevance to query, completeness, "
            "and use of specific data (not vague generalities). "
            "Return ONLY valid JSON: {\"score\": <float>, \"reasoning\": \"<one sentence>\"}"
        )
        try:
            from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415
            resp = await llm.ainvoke([
                SystemMessage(content="You are a strict evaluator. Return only the JSON requested."),
                HumanMessage(content=prompt),
            ])
            from agents.utils import extract_json  # noqa: PLC0415
            parsed = extract_json(resp.content)
            if not isinstance(parsed, dict):
                raise ValueError("extract_json returned non-dict")
            score = float(parsed.get("score", base_score))
            reasoning = str(parsed.get("reasoning", "LLM judge"))
            return EvalResult(score=score, reasoning=reasoning, checks={"method": "llm_judge"})
        except Exception as e:
            logger.debug(f"LLM judge failed: {e}")
            return EvalResult(score=base_score, reasoning=f"LLM judge failed; length score: {base_score:.2f}")

    async def _eval_generic_llm(
        self, capability: str, input_data: dict, output: dict
    ) -> EvalResult:
        """Generic LLM-as-judge for unknown capability types."""
        query = str(input_data)[:300]
        result = str(output)[:600]

        base_score = 0.6 if output else 0.1

        llm = self._get_llm()
        if llm is None:
            return EvalResult(score=base_score, reasoning="No LLM available; default score")

        prompt = (
            f"Capability: {capability}\n"
            f"Input: {query}\n"
            f"Output: {result}\n\n"
            "Score this output 0.0–1.0 on quality and relevance. "
            "Return ONLY valid JSON: {\"score\": <float>, \"reasoning\": \"<one sentence>\"}"
        )
        try:
            from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415
            resp = await llm.ainvoke([
                SystemMessage(content="You are a strict evaluator. Return only the JSON requested."),
                HumanMessage(content=prompt),
            ])
            from agents.utils import extract_json  # noqa: PLC0415
            parsed = extract_json(resp.content)
            if not isinstance(parsed, dict):
                raise ValueError("extract_json returned non-dict")
            score = float(parsed.get("score", base_score))
            reasoning = str(parsed.get("reasoning", "LLM judge"))
            return EvalResult(score=score, reasoning=reasoning, checks={"method": "llm_judge"})
        except Exception as e:
            logger.debug(f"Generic LLM judge failed: {e}")
            return EvalResult(score=base_score, reasoning="Generic fallback score")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _number_mentioned(text: str, value: float, tolerance: float = 0.1) -> bool:
    """Return True if a number within `tolerance` fraction of `value` appears in text."""
    import re  # noqa: PLC0415
    # Match comma-grouped numbers (1,000,000) OR plain integers/floats (675000000, 3.14)
    numbers = re.findall(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?", text)
    for num_str in numbers:
        try:
            num = float(num_str.replace(",", ""))
            if value != 0 and abs(num - value) / abs(value) <= tolerance:
                return True
            if value == 0 and abs(num) < 1e-6:
                return True
        except ValueError:
            continue
    return False
