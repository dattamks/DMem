"""Regression tests for the token-savings measurement (real mechanism sizes)."""

from dmem.eval.tokens import (measure_token_savings, format_savings_report,
                              count_tokens, ScenarioSaving)


def test_counter_monotonic():
    assert count_tokens("a longer piece of text here") > count_tokens("short")
    assert count_tokens("") >= 1


def test_scenarios_present_and_positive_savings():
    report = measure_token_savings()
    names = {s.name for s in report.scenarios}
    assert names == {"documents", "long_conversation", "model_switch"}
    for s in report.scenarios:
        assert s.baseline_tokens > s.dmem_tokens  # DMem sends less
        assert s.saving_pct > 0


def test_savings_within_expected_bands():
    report = measure_token_savings()
    by = {s.name: s.saving_pct for s in report.scenarios}
    # measured mechanism savings should be in these (generous) bands
    assert by["documents"] > 40           # selective retrieval vs full doc
    assert by["long_conversation"] > 30   # compaction vs full transcript
    assert by["model_switch"] > 70        # budgeted handoff vs full history
    assert report.blended_pct() > 40


def test_saving_pct_math():
    s = ScenarioSaving("x", baseline_tokens=1000, dmem_tokens=250)
    assert abs(s.saving_pct - 75.0) < 1e-6
    assert ScenarioSaving("y", 0, 0).saving_pct == 0.0


def test_report_renders():
    text = format_savings_report(measure_token_savings())
    assert "token-savings" in text
    assert "blended" in text
    assert "%" in text
