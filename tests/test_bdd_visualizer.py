"""Tests for BDD summary and formatting."""
from smvis.bdd_visualizer import get_bdd_summary, format_encoding_table, format_iteration_log


class TestBddSummary:
    def test_summary_keys(self, counter_bdd):
        summary = get_bdd_summary(counter_bdd)
        assert "encoding" in summary
        assert "total_bdd_vars" in summary
        assert "total_reachable" in summary
        assert "iterations" in summary

    def test_encoding_entries(self, counter_bdd):
        summary = get_bdd_summary(counter_bdd)
        for enc in summary["encoding"]:
            assert "variable" in enc
            assert "domain_size" in enc
            assert "bits" in enc
            assert "bdd_vars" in enc

    def test_reachable_count(self, counter_bdd):
        summary = get_bdd_summary(counter_bdd)
        assert summary["total_reachable"] == 24


class TestFormatting:
    def test_encoding_table(self, counter_bdd):
        table = format_encoding_table(counter_bdd)
        assert "Variable" in table
        assert "mode" in table

    def test_iteration_log(self, counter_bdd):
        log = format_iteration_log(counter_bdd)
        assert "Iter" in log
        assert "Total Reachable" in log
