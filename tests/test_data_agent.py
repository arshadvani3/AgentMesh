"""Tests for DataAgent — real CSV computation via pandas/numpy."""

from __future__ import annotations

import textwrap

import pytest

from agents.data_agent import _compute_stats_numpy, compute_csv_stats

# ---------------------------------------------------------------------------
# Sample CSV data
# ---------------------------------------------------------------------------

SAMPLE_CSV = textwrap.dedent("""\
    company,sector,amount_usd,year
    Alpha,AI,1000000,2021
    Beta,Fintech,5000000,2022
    Gamma,AI,3000000,2021
    Delta,Fintech,2000000,2023
    Epsilon,AI,4000000,2022
    Zeta,Data,500000,2023
""")

STARTUPS_CSV_PATH = "data/startups.csv"


class TestComputeStatsNumpy:
    """Test the numpy fallback path directly (always available)."""

    def test_shape(self):
        stats = _compute_stats_numpy(SAMPLE_CSV)
        assert stats["rows"] == 6
        assert stats["cols"] == 4

    def test_column_names(self):
        stats = _compute_stats_numpy(SAMPLE_CSV)
        assert "company" in stats["columns"]
        assert "amount_usd" in stats["columns"]

    def test_numeric_stats_present(self):
        stats = _compute_stats_numpy(SAMPLE_CSV)
        assert "amount_usd" in stats["numeric_stats"]
        ns = stats["numeric_stats"]["amount_usd"]
        assert ns["count"] == 6
        assert ns["mean"] == pytest.approx(2583333.33, rel=1e-3)
        assert ns["min"] == 500000
        assert ns["max"] == 5000000

    def test_categorical_summaries(self):
        stats = _compute_stats_numpy(SAMPLE_CSV)
        sector_counts = stats["categorical_summaries"].get("sector", {})
        assert sector_counts.get("AI") == 3
        assert sector_counts.get("Fintech") == 2

    def test_engine_label(self):
        stats = _compute_stats_numpy(SAMPLE_CSV)
        assert stats["engine"] == "numpy"

    def test_empty_csv_returns_zero_rows(self):
        stats = _compute_stats_numpy("col1,col2\n")
        assert stats["rows"] == 0

    def test_missing_values_counted(self):
        csv_with_blanks = "a,b\n1,\n2,3\n"
        stats = _compute_stats_numpy(csv_with_blanks)
        assert stats["missing_values"].get("b", 0) == 1


class TestComputeStats:
    """Test the unified compute_csv_stats dispatcher (uses pandas if available)."""

    def test_returns_expected_keys(self):
        stats = compute_csv_stats(SAMPLE_CSV)
        for key in ("rows", "cols", "columns", "numeric_stats",
                    "missing_values", "categorical_summaries", "engine"):
            assert key in stats, f"Missing key: {key}"

    def test_shape_matches(self):
        stats = compute_csv_stats(SAMPLE_CSV)
        assert stats["rows"] == 6
        assert stats["cols"] == 4

    def test_numeric_mean_correct(self):
        stats = compute_csv_stats(SAMPLE_CSV)
        mean = stats["numeric_stats"]["amount_usd"]["mean"]
        assert mean == pytest.approx(2583333.33, rel=1e-2)

    def test_engine_is_pandas_or_numpy(self):
        stats = compute_csv_stats(SAMPLE_CSV)
        assert stats["engine"] in ("pandas", "numpy")

    def test_real_startups_csv(self):
        """Verify the canonical demo dataset loads and computes correctly."""
        try:
            with open(STARTUPS_CSV_PATH) as f:
                content = f.read()
        except FileNotFoundError:
            pytest.skip("data/startups.csv not found")

        stats = compute_csv_stats(content)
        assert stats["rows"] == 100
        assert stats["cols"] == 6
        # AI should be the largest sector
        sector_counts = stats["categorical_summaries"].get("sector", {})
        assert sector_counts.get("AI", 0) == 20
        # amount_usd mean should be in the hundreds of millions range
        mean_amount = stats["numeric_stats"]["amount_usd"]["mean"]
        assert mean_amount > 100_000_000


class TestLoadCsvPath:
    """Test the _load_csv helper (local file path)."""

    @pytest.mark.asyncio
    async def test_load_local_path(self, tmp_path):
        """_load_csv loads a local file when csv_path is given."""
        from agents.data_agent import _load_csv

        csv_file = tmp_path / "test.csv"
        csv_file.write_text(SAMPLE_CSV)

        content, source = await _load_csv({"csv_path": str(csv_file)})
        assert content is not None
        assert "Alpha" in content
        assert source == str(csv_file)

    @pytest.mark.asyncio
    async def test_load_missing_path_returns_none(self):
        """_load_csv returns (None, None) when file does not exist."""
        from agents.data_agent import _load_csv

        content, source = await _load_csv({"csv_path": "/nonexistent/path.csv"})
        assert content is None
        assert source is None

    @pytest.mark.asyncio
    async def test_no_csv_key_returns_none(self):
        """_load_csv returns (None, None) when neither csv_path nor csv_url is set."""
        from agents.data_agent import _load_csv

        content, source = await _load_csv({"query": "some query"})
        assert content is None
        assert source is None
