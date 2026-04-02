# AI Data Analysis Agent

An AI-powered analytical agent capable of processing structured data, performing statistical analysis, and generating reports.

## Profile

- Framework: Agno
- Difficulty: Medium
- Skills: data-analysis, visualization
- Example use cases:
  1. Analyze a sales dataset and highlight trends.
  2. Find anomalies in financial data.
- API keys expected:
  - `OPENROUTER_API_KEY`
  - `MEM0_API_KEY`

## What This Agent Can Do

- Clean structured datasets
- Compute descriptive statistics
- Identify trends and potential anomalies
- Generate chart-ready assets
- Produce a structured markdown report

## Project Layout

```text
ai_data_analysis_agent/
  __init__.py
  analysis.py
  main.py
agent_config.json
sample_data/
  sales.csv
tests/
  test_analysis.py
```

## Quickstart

```powershell
uv sync
uv run ai-data-analysis-agent --input .\sample_data\sales.csv --output .\reports\sales-report.md
```

Prompt-driven usage:

```powershell
uv run ai-data-analysis-agent `
  --input .\sample_data\sales.csv `
  --output .\reports\sales-report.md `
  --question "Find anomalies in revenue and explain the biggest business trends."
```

You can also start from Python directly:

```powershell
uv run python -m ai_data_analysis_agent.main --input .\sample_data\sales.csv
```

## Environment Variables

This scaffold is usable without live model calls for deterministic analysis. If you later wire in hosted reasoning or memory, set:

```powershell
$env:OPENROUTER_API_KEY="your-openrouter-key"
$env:MEM0_API_KEY="your-mem0-key"
```

## Output

The CLI writes:

- A markdown report under `reports/`
- PNG charts under `reports/charts/<report-name>/`
- A full store forecast CSV plus separate top-risers and top-softening CSV exports when store forecasting is available

## Example Questions

- `Analyze weekly sales and highlight the strongest trends.`
- `Find anomalies in revenue and orders.`
- `Explain holiday impact on sales.`
- `Compare performance by store or department.`

## Interactive Chat

You can also keep the dataset loaded and ask follow-up questions in a loop:

```powershell
uv run ai-data-analysis-agent --input .\sample_data\sales.csv --chat
```

Each chat question writes a timestamped markdown report under `reports/chat_reports/`.

## Notes

This starter keeps the analysis path local and deterministic so you can develop and test without depending on external APIs. It is ready to be extended with Agno agents, tools, and memory integrations.

## Agno Integration

The scaffold includes a lightweight Agno-compatible agent constructor in `ai_data_analysis_agent/main.py`. If `agno` is installed, you can import and extend `build_agent()` to add hosted LLMs, tools, or memory backends while keeping the deterministic CSV analysis pipeline as the execution layer.


