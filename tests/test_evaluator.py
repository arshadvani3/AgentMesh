"""Tests for OutputEvaluator — capability-specific output scoring."""

from __future__ import annotations

import pytest

from mesh.evaluator import EvalResult, OutputEvaluator, _number_mentioned


@pytest.fixture
def evaluator() -> OutputEvaluator:
    return OutputEvaluator(groq_api_key=None)  # no LLM — deterministic paths only


class TestEvalResult:
    def test_score_clamped_low(self):
        r = EvalResult(score=-0.5, reasoning="test")
        assert r.score == 0.0

    def test_score_clamped_high(self):
        r = EvalResult(score=1.5, reasoning="test")
        assert r.score == 1.0

    def test_score_valid(self):
        r = EvalResult(score=0.75, reasoning="ok")
        assert r.score == 0.75

    def test_checks_default_empty(self):
        r = EvalResult(score=0.5, reasoning="x")
        assert r.checks == {}


class TestNumberMentioned:
    def test_exact_match(self):
        assert _number_mentioned("mean is 675000000", 675000000, tolerance=0.01)

    def test_within_tolerance(self):
        assert _number_mentioned("about 100", 99, tolerance=0.05)

    def test_outside_tolerance(self):
        assert not _number_mentioned("value is 200", 100, tolerance=0.05)

    def test_comma_formatted(self):
        assert _number_mentioned("total of 1,000,000 dollars", 1000000, tolerance=0.01)

    def test_no_numbers(self):
        assert not _number_mentioned("no numbers here", 42, tolerance=0.1)


class TestEvalCsvAnalysis:
    @pytest.mark.asyncio
    async def test_good_analysis_scores_high(self, evaluator: OutputEvaluator):
        output = {
            "analysis": "A" * 200,  # long enough
            "key_findings": ["AI led with $675M", "Fintech second"],
        }
        result = await evaluator.evaluate("analyze_csv", {}, output)
        assert result.score > 0.5

    @pytest.mark.asyncio
    async def test_empty_analysis_scores_low(self, evaluator: OutputEvaluator):
        result = await evaluator.evaluate("analyze_csv", {}, {"analysis": "", "key_findings": []})
        assert result.score < 0.5

    @pytest.mark.asyncio
    async def test_grounding_check_with_context(self, evaluator: OutputEvaluator):
        output = {
            "analysis": "The mean funding was 675000000 across companies. " * 10,
            "key_findings": ["AI dominated"],
        }
        context = {"computed_stats": {"numeric_stats": {"amount_usd": {"mean": 675000000}}}}
        result = await evaluator.evaluate("analyze_csv", {}, output, context=context)
        assert result.checks["grounding"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_grounding_fails_when_no_numbers(self, evaluator: OutputEvaluator):
        output = {
            "analysis": "The data shows interesting patterns. " * 10,
            "key_findings": ["some finding"],
        }
        context = {"computed_stats": {"numeric_stats": {"amount_usd": {"mean": 675000000}}}}
        result = await evaluator.evaluate("analyze_csv", {}, output, context=context)
        assert result.checks["grounding"] == pytest.approx(0.0)


class TestEvalCode:
    @pytest.mark.asyncio
    async def test_successful_execution_scores_high(self, evaluator: OutputEvaluator):
        output = {
            "examples": [
                {"code": "print(1)", "execution_result": {"exit_code": 0, "stdout": "1", "stderr": ""}},
            ]
        }
        result = await evaluator.evaluate("fetch_code", {}, output)
        assert result.score >= 0.9

    @pytest.mark.asyncio
    async def test_syntax_error_scores_low(self, evaluator: OutputEvaluator):
        output = {
            "examples": [
                {"code": "def broken(", "execution_result": {"exit_code": 1, "stdout": "", "stderr": "SyntaxError: invalid syntax"}},
            ]
        }
        result = await evaluator.evaluate("fetch_code", {}, output)
        assert result.score < 0.5

    @pytest.mark.asyncio
    async def test_not_executed_partial_credit(self, evaluator: OutputEvaluator):
        output = {
            "examples": [
                {"code": "import matplotlib\nprint('chart')", "executed": False},
            ]
        }
        result = await evaluator.evaluate("fetch_code", {}, output)
        assert 0.4 < result.score < 0.9

    @pytest.mark.asyncio
    async def test_no_examples_scores_low(self, evaluator: OutputEvaluator):
        result = await evaluator.evaluate("fetch_code", {}, {"examples": []})
        assert result.score < 0.4

    @pytest.mark.asyncio
    async def test_mixed_results_partial_score(self, evaluator: OutputEvaluator):
        output = {
            "examples": [
                {"code": "print(1)", "execution_result": {"exit_code": 0, "stdout": "1", "stderr": ""}},
                {"code": "def bad(", "execution_result": {"exit_code": 1, "stdout": "", "stderr": "SyntaxError"}},
            ]
        }
        result = await evaluator.evaluate("fetch_code", {}, output)
        assert 0.1 < result.score < 0.9


class TestEvalWebSearch:
    @pytest.mark.asyncio
    async def test_rich_results_score_high(self, evaluator: OutputEvaluator):
        output = {"results": "x" * 1500, "source": "brave-search-mcp"}
        result = await evaluator.evaluate("web_search", {}, output)
        assert result.score > 0.7

    @pytest.mark.asyncio
    async def test_empty_results_score_low(self, evaluator: OutputEvaluator):
        result = await evaluator.evaluate("web_search", {}, {"results": ""})
        assert result.score < 0.3

    @pytest.mark.asyncio
    async def test_source_boosts_score(self, evaluator: OutputEvaluator):
        base = await evaluator.evaluate("web_search", {}, {"results": "some data"})
        with_source = await evaluator.evaluate("web_search", {}, {"results": "some data", "source": "mcp"})
        assert with_source.score >= base.score


class TestEvalReportFallback:
    @pytest.mark.asyncio
    async def test_long_report_gets_decent_score_no_llm(self, evaluator: OutputEvaluator):
        output = {"report": "This is a comprehensive analysis. " * 60}
        result = await evaluator.evaluate("write_report", {"query": "analyze data"}, output)
        assert result.score > 0.4

    @pytest.mark.asyncio
    async def test_empty_report_scores_low(self, evaluator: OutputEvaluator):
        result = await evaluator.evaluate("write_report", {"query": "test"}, {"report": ""})
        assert result.score < 0.3


class TestEvalGenericFallback:
    @pytest.mark.asyncio
    async def test_unknown_capability_with_output(self, evaluator: OutputEvaluator):
        result = await evaluator.evaluate("custom_task", {"query": "do something"}, {"result": "done"})
        assert result.score > 0.0

    @pytest.mark.asyncio
    async def test_unknown_capability_empty_output(self, evaluator: OutputEvaluator):
        result = await evaluator.evaluate("custom_task", {}, {})
        assert result.score == pytest.approx(0.1)

    @pytest.mark.asyncio
    async def test_eval_exception_returns_neutral(self, evaluator: OutputEvaluator):
        # Passing an output that causes internal issues — should not raise
        result = await evaluator.evaluate("analyze_csv", {}, None)  # type: ignore[arg-type]
        assert 0.0 <= result.score <= 1.0
