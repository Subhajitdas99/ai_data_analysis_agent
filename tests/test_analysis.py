from pathlib import Path

from ai_data_analysis_agent.analysis import analyze_dataset


def test_analyze_dataset_generates_summary_and_charts(tmp_path: Path) -> None:
    dataset = tmp_path / "sample.csv"
    dataset.write_text(
        "month,revenue,orders\n2026-01,10,2\n2026-02,1000,3\n2026-03,12,2\n",
        encoding="utf-8",
    )

    result = analyze_dataset(
        dataset,
        tmp_path / "charts",
        question="Find anomalies in revenue and explain trends.",
    )

    assert result.row_count == 3
    assert result.column_count == 3
    assert "Analysis Report" in result.summary_markdown
    assert "Analyst Prompt" in result.summary_markdown
    assert "revenue" in result.summary_markdown.lower()
    assert result.chart_paths


def test_dimension_columns_are_not_treated_as_primary_metrics(tmp_path: Path) -> None:
    dataset = tmp_path / "walmart_like.csv"
    dataset.write_text(
        (
            "Store,Date,Weekly_Sales,Holiday_Flag,Temperature\n"
            "1,05-02-2010,1643690.9,0,42.31\n"
            "1,12-02-2010,1641957.44,1,38.51\n"
            "2,05-02-2010,2136989.46,0,38.49\n"
            "2,12-02-2010,2137809.5,1,38.49\n"
        ),
        encoding="utf-8",
    )

    result = analyze_dataset(
        dataset,
        tmp_path / "charts",
        question="Compare store performance and highlight the biggest sales drivers.",
    )

    assert "Primary metric: Weekly_Sales" in result.summary_markdown
    assert "Business dimensions: Store, Holiday_Flag" in result.summary_markdown
