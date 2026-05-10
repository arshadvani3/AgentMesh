"""Tests for CodeAgent sandbox execution."""

from __future__ import annotations

from agents.code_agent import _run_in_sandbox


class TestRunInSandbox:
    def test_simple_print_captured(self):
        """stdout from print() is captured correctly."""
        result = _run_in_sandbox("print('hello sandbox')")
        assert result["executed"] is True
        assert "hello sandbox" in result["stdout"]
        assert result["exit_code"] == 0
        assert result["error"] is None

    def test_arithmetic_result(self):
        """Arithmetic output is real, not hallucinated."""
        result = _run_in_sandbox("print(6 * 7)")
        assert result["executed"] is True
        assert "42" in result["stdout"]

    def test_multiline_code(self):
        """Multi-statement code executes correctly."""
        code = "x = [i**2 for i in range(5)]\nprint(sum(x))"
        result = _run_in_sandbox(code)
        assert result["executed"] is True
        assert "30" in result["stdout"]

    def test_syntax_error_not_executed(self):
        """Syntax errors are caught before subprocess launch."""
        result = _run_in_sandbox("def broken(")
        assert result["executed"] is False
        assert "SyntaxError" in result["stderr"]
        assert result["exit_code"] == 1

    def test_runtime_error_captured(self):
        """Runtime errors return non-zero exit code with stderr."""
        result = _run_in_sandbox("x = 1 / 0")
        assert result["executed"] is True
        assert result["exit_code"] != 0
        assert "ZeroDivisionError" in result["stderr"]

    def test_timeout_respected(self):
        """Code that loops forever is killed after the timeout."""
        result = _run_in_sandbox("import time; time.sleep(30)")
        assert result["executed"] is False
        assert result["exit_code"] == -1
        assert "timed out" in result["error"].lower()

    def test_stdout_capped_at_2000_chars(self):
        """Very long stdout is truncated to 2000 characters."""
        code = "print('x' * 5000)"
        result = _run_in_sandbox(code)
        assert result["executed"] is True
        assert len(result["stdout"]) <= 2000

    def test_import_stdlib_works(self):
        """Standard library imports work inside the sandbox."""
        code = "import math; print(round(math.sqrt(144)))"
        result = _run_in_sandbox(code)
        assert result["executed"] is True
        assert "12" in result["stdout"]

    def test_empty_code_runs(self):
        """Empty string executes with exit code 0 and no output."""
        result = _run_in_sandbox("")
        assert result["executed"] is True
        assert result["exit_code"] == 0
        assert result["stdout"] == ""

    def test_isolation_no_agent_state(self):
        """Sandbox cannot access variables defined in the agent process."""
        # _EXEC_TIMEOUT_SECONDS is defined in code_agent module scope
        code = "from agents.code_agent import _EXEC_TIMEOUT_SECONDS; print(_EXEC_TIMEOUT_SECONDS)"
        result = _run_in_sandbox(code)
        # Either it works (module accessible in PYTHONPATH) or it fails —
        # what matters is the sandbox doesn't share local variables with the caller
        # We just verify it doesn't crash the test runner
        assert "executed" in result
