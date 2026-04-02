from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re

from ai_data_analysis_agent.analysis import analyze_dataset

try:
    from agno.agent import Agent
except ImportError:  # pragma: no cover - optional runtime integration
    Agent = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze structured data, generate summary statistics, and write a markdown report."
    )
    parser.add_argument("--input", required=True, help="Path to the input CSV or Excel dataset.")
    parser.add_argument(
        "--output",
        default="reports/analysis-report.md",
        help="Path to the markdown report to generate.",
    )
    parser.add_argument(
        "--question",
        default="Analyze this dataset and highlight trends, anomalies, and important business insights.",
        help="Natural-language question to guide the analysis report.",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Start an interactive analysis loop and generate a report for each question.",
    )
    return parser


def build_agent() -> object | None:
    """Return a minimal Agno agent when the package is installed."""
    if Agent is None:
        return None

    return Agent(
        name="AI Data Analysis Agent",
        description=(
            "Processes structured data, performs statistical analysis, "
            "and generates structured analytical reports."
        ),
        instructions=[
            "Inspect the schema before making claims.",
            "Summarize descriptive statistics clearly.",
            "Call out missing data and anomaly signals.",
            "Recommend visualizations and next analytical steps.",
        ],
    )


def _write_outputs(output_path: Path, report_slug: str, result: object) -> None:
    output_path.write_text(result.summary_markdown, encoding="utf-8")
    forecast_csv_path = output_path.parent / f"{report_slug}-store-forecast.csv"
    rising_csv_path = output_path.parent / f"{report_slug}-top-risers.csv"
    softening_csv_path = output_path.parent / f"{report_slug}-top-softening.csv"
    if result.store_forecast_table is not None:
        result.store_forecast_table.to_csv(forecast_csv_path, index=False)
    if result.rising_store_forecast_table is not None:
        result.rising_store_forecast_table.to_csv(rising_csv_path, index=False)
    if result.softening_store_forecast_table is not None:
        result.softening_store_forecast_table.to_csv(softening_csv_path, index=False)

    print(f"Report written to: {output_path}")
    print(f"Charts written to: {output_path.parent / 'charts' / report_slug}")
    if result.store_forecast_table is not None:
        print(f"Store forecast CSV written to: {forecast_csv_path}")
    if result.rising_store_forecast_table is not None:
        print(f"Top risers CSV written to: {rising_csv_path}")
    if result.softening_store_forecast_table is not None:
        print(f"Top softening CSV written to: {softening_csv_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    agent = build_agent()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.chat:
        chat_reports_dir = output_path.parent / "chat_reports"
        chat_reports_dir.mkdir(parents=True, exist_ok=True)
        print("Interactive chat mode started. Type a question, or type 'exit' to stop.")
        while True:
            question = input("analysis> ").strip()
            if question.lower() in {"exit", "quit"}:
                print("Chat mode ended.")
                break
            if not question:
                continue

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            chat_output = chat_reports_dir / f"chat-{timestamp}.md"
            report_slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", chat_output.stem).strip("-") or "analysis-report"
            charts_dir = chat_output.parent / "charts" / report_slug
            result = analyze_dataset(args.input, charts_dir, question=question)

            print("AI Data Analysis Agent complete.")
            print(f"Agno agent available: {'yes' if agent is not None else 'no'}")
            print(f"Rows analyzed: {result.row_count}")
            print(f"Columns analyzed: {result.column_count}")
            print(f"Question: {question}")
            _write_outputs(chat_output, report_slug, result)
        return

    report_slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", output_path.stem).strip("-") or "analysis-report"
    charts_dir = output_path.parent / "charts" / report_slug
    result = analyze_dataset(args.input, charts_dir, question=args.question)
    print("AI Data Analysis Agent complete.")
    print(f"Agno agent available: {'yes' if agent is not None else 'no'}")
    print(f"Rows analyzed: {result.row_count}")
    print(f"Columns analyzed: {result.column_count}")
    print(f"Question: {args.question}")
    _write_outputs(output_path, report_slug, result)


if __name__ == "__main__":
    main()
