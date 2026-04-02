from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class AnalysisResult:
    row_count: int
    column_count: int
    numeric_columns: list[str]
    missing_values: dict[str, int]
    summary_markdown: str
    chart_paths: list[Path]
    store_forecast_table: pd.DataFrame | None = None
    rising_store_forecast_table: pd.DataFrame | None = None
    softening_store_forecast_table: pd.DataFrame | None = None


@dataclass
class ForecastResult:
    metric: str
    periods: int
    forecast_frame: pd.DataFrame
    summary_lines: list[str]


def load_dataset(dataset_path: str | Path) -> pd.DataFrame:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _tokenize_question(question: str | None) -> set[str]:
    if not question:
        return set()
    return set(_normalize_text(question).split())


def _parse_dates(dataframe: pd.DataFrame) -> pd.DataFrame:
    cleaned = dataframe.copy()
    for column in cleaned.columns:
        if "date" in column.lower():
            parsed = pd.to_datetime(cleaned[column], errors="coerce", dayfirst=True)
            if parsed.notna().any():
                cleaned[column] = parsed
    return cleaned


def _coerce_numeric_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    cleaned = dataframe.copy()
    for column in cleaned.columns:
        if pd.api.types.is_datetime64_any_dtype(cleaned[column]):
            continue
        if cleaned[column].dtype == object:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="ignore")
    return cleaned


def _find_date_column(dataframe: pd.DataFrame) -> str | None:
    for column in dataframe.columns:
        if pd.api.types.is_datetime64_any_dtype(dataframe[column]):
            return column
    return None


def _is_dimension_column(dataframe: pd.DataFrame, column: str) -> bool:
    lower = column.lower()
    if any(token in lower for token in ("store", "dept", "department", "flag", "category", "type")):
        return True
    series = dataframe[column]
    if pd.api.types.is_datetime64_any_dtype(series):
        return False
    if not pd.api.types.is_numeric_dtype(series):
        return series.nunique(dropna=True) <= min(50, max(10, len(series) // 20))
    unique_count = series.nunique(dropna=True)
    return unique_count <= min(50, max(10, len(series) // 20))


def _metric_candidates(dataframe: pd.DataFrame) -> list[str]:
    return [
        column
        for column in dataframe.select_dtypes(include="number").columns.tolist()
        if not _is_dimension_column(dataframe, column)
    ]


def _dimension_candidates(dataframe: pd.DataFrame) -> list[str]:
    return [column for column in dataframe.columns if _is_dimension_column(dataframe, column)]


def _pick_primary_metric(dataframe: pd.DataFrame, question: str | None) -> str | None:
    metrics = _metric_candidates(dataframe)
    if not metrics:
        return None

    preferred_names = ("weekly_sales", "sales", "revenue", "profit", "orders", "amount")
    for preferred in preferred_names:
        for column in metrics:
            if preferred in column.lower():
                return column

    tokens = _tokenize_question(question)
    scored: list[tuple[int, str]] = []
    for column in metrics:
        score = len(tokens.intersection(set(_normalize_text(column).split())))
        if score > 0:
            scored.append((score, column))
    if scored:
        scored.sort(reverse=True)
        return scored[0][1]
    return metrics[0]


def _find_focus_metrics(dataframe: pd.DataFrame, question: str | None) -> list[str]:
    metrics = _metric_candidates(dataframe)
    if not metrics:
        return []

    primary_metric = _pick_primary_metric(dataframe, question)
    remaining = [column for column in metrics if column != primary_metric]
    focus = [primary_metric] if primary_metric else []

    tokens = _tokenize_question(question)
    scored: list[tuple[int, str]] = []
    for column in remaining:
        score = len(tokens.intersection(set(_normalize_text(column).split())))
        if _normalize_text(column) in _normalize_text(question or ""):
            score += 2
        if score > 0:
            scored.append((score, column))

    scored.sort(reverse=True)
    for _, column in scored[:2]:
        focus.append(column)

    for column in remaining:
        if len(focus) >= 3:
            break
        if column not in focus:
            focus.append(column)
    return focus


def _find_relevant_dimensions(dataframe: pd.DataFrame, question: str | None) -> list[str]:
    dimensions = _dimension_candidates(dataframe)
    if not dimensions:
        return []

    preferred = ["store", "dept", "department", "holiday"]
    ranked: list[str] = []
    lower_map = {column.lower(): column for column in dimensions}
    for token in preferred:
        for key, column in lower_map.items():
            if token in key and column not in ranked:
                ranked.append(column)

    question_tokens = _tokenize_question(question)
    for column in dimensions:
        if column in ranked:
            continue
        if question_tokens.intersection(set(_normalize_text(column).split())):
            ranked.append(column)

    for column in dimensions:
        if column not in ranked:
            ranked.append(column)
    return ranked[:3]


def _format_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No data available."
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return f"```text\n{frame.to_string(index=False)}\n```"


def _format_value(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{value:,.2f}"
    return str(value)


def _find_dimension_column(dataframe: pd.DataFrame, *keywords: str) -> str | None:
    for column in dataframe.columns:
        lower = column.lower()
        if any(keyword in lower for keyword in keywords):
            return column
    return None


def _build_forecast(dataframe: pd.DataFrame, metric: str | None, periods: int = 8) -> ForecastResult | None:
    date_column = _find_date_column(dataframe)
    if metric is None or date_column is None:
        return None

    series = (
        dataframe[[date_column, metric]]
        .dropna()
        .groupby(date_column)[metric]
        .mean()
        .sort_index()
    )
    if len(series) < 8:
        return None

    history = series.reset_index()
    history.columns = ["Date", "Value"]
    history["Step"] = range(len(history))

    recent_window = min(12, len(history))
    recent_history = history.tail(recent_window).copy()
    slope, intercept = __import__("numpy").polyfit(recent_history["Step"], recent_history["Value"], 1)

    seasonal_window = min(4, len(recent_history))
    seasonal_baseline = recent_history["Value"].tail(seasonal_window).mean()

    future_dates = pd.date_range(
        start=history["Date"].iloc[-1] + pd.Timedelta(days=7),
        periods=periods,
        freq="W-FRI",
    )
    future_steps = range(len(history), len(history) + periods)

    forecast_values: list[float] = []
    last_actual = float(history["Value"].iloc[-1])
    for offset, step in enumerate(future_steps):
        trend_value = (slope * step) + intercept
        seasonal_value = float(recent_history["Value"].iloc[offset % seasonal_window])
        blended = (0.65 * trend_value) + (0.35 * seasonal_value)
        forecast_values.append(max(0.0, blended))

    forecast_frame = pd.DataFrame(
        {
            "Date": future_dates,
            metric: forecast_values,
        }
    )
    avg_forecast = float(forecast_frame[metric].mean())
    last_forecast = float(forecast_frame[metric].iloc[-1])
    change_pct = ((avg_forecast - last_actual) / last_actual * 100) if last_actual else 0.0

    summary_lines = [
        f"- Forecast horizon: next {periods} weeks.",
        f"- Latest observed `{metric}`: {_format_value(last_actual)}.",
        f"- Average forecast `{metric}`: {_format_value(avg_forecast)}.",
        f"- End-of-horizon forecast `{metric}`: {_format_value(last_forecast)}.",
        f"- Expected change versus latest actual: {change_pct:.2f}%.",
    ]
    return ForecastResult(metric=metric, periods=periods, forecast_frame=forecast_frame, summary_lines=summary_lines)


def _build_store_forecast_table(dataframe: pd.DataFrame, metric: str | None, periods: int = 4) -> pd.DataFrame:
    store_column = _find_dimension_column(dataframe, "store")
    date_column = _find_date_column(dataframe)
    if metric is None or store_column is None or date_column is None:
        return pd.DataFrame()

    store_summary = (
        dataframe[[store_column, metric]]
        .dropna()
        .groupby(store_column)[metric]
        .mean()
        .sort_values(ascending=False)
    )
    if store_summary.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []

    for store in store_summary.index.tolist():
        store_frame = dataframe.loc[dataframe[store_column] == store].copy()
        forecast = _build_forecast(store_frame, metric, periods=periods)
        if forecast is None:
            continue
        last_actual_series = (
            store_frame[[date_column, metric]]
            .dropna()
            .groupby(date_column)[metric]
            .mean()
            .sort_index()
        )
        last_actual = float(last_actual_series.iloc[-1])
        avg_forecast = float(forecast.forecast_frame[metric].mean())
        end_forecast = float(forecast.forecast_frame[metric].iloc[-1])
        change_pct = ((avg_forecast - last_actual) / last_actual * 100) if last_actual else 0.0
        outlook = "rising" if change_pct > 1 else "softening" if change_pct < -1 else "stable"
        rows.append(
            {
                "Store": store,
                "latest_actual": round(last_actual, 2),
                f"avg_next_{periods}_weeks": round(avg_forecast, 2),
                f"end_of_{periods}_weeks": round(end_forecast, 2),
                "change_pct": round(change_pct, 2),
                "outlook": outlook,
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("change_pct", ascending=False).reset_index(drop=True)


def create_charts(dataframe: pd.DataFrame, output_dir: str | Path, question: str | None = None) -> list[Path]:
    charts_dir = Path(output_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    chart_paths: list[Path] = []
    primary_metric = _pick_primary_metric(dataframe, question)
    if primary_metric is None:
        return chart_paths

    date_column = _find_date_column(dataframe)
    dimensions = _find_relevant_dimensions(dataframe, question)

    if date_column:
        figure_path = charts_dir / f"{primary_metric}_trend.png"
        trend = (
            dataframe[[date_column, primary_metric]]
            .dropna()
            .groupby(date_column)[primary_metric]
            .mean()
            .sort_index()
        )
        if not trend.empty:
            plt.figure(figsize=(10, 4))
            trend.plot(kind="line", title=f"{primary_metric} over time")
            plt.xlabel(date_column)
            plt.ylabel(primary_metric)
            plt.tight_layout()
            plt.savefig(figure_path)
            plt.close()
            chart_paths.append(figure_path)

        forecast = _build_forecast(dataframe, primary_metric)
        if forecast is not None:
            figure_path = charts_dir / f"{primary_metric}_forecast.png"
            plt.figure(figsize=(10, 4))
            trend.tail(26).plot(label="Historical", color="#1f77b4")
            forecast_series = forecast.forecast_frame.set_index("Date")[primary_metric]
            forecast_series.plot(label="Forecast", color="#d62728")
            plt.title(f"{primary_metric} forecast")
            plt.xlabel(date_column)
            plt.ylabel(primary_metric)
            plt.legend()
            plt.tight_layout()
            plt.savefig(figure_path)
            plt.close()
            chart_paths.append(figure_path)

        monthly = (
            dataframe.assign(_month=dataframe[date_column].dt.to_period("M").astype(str))
            .groupby("_month")[primary_metric]
            .mean()
            .tail(12)
        )
        if not monthly.empty:
            figure_path = charts_dir / f"{primary_metric}_monthly_seasonality.png"
            plt.figure(figsize=(10, 4))
            monthly.plot(kind="bar", title=f"{primary_metric} monthly seasonality")
            plt.xlabel("Month")
            plt.ylabel(primary_metric)
            plt.tight_layout()
            plt.savefig(figure_path)
            plt.close()
            chart_paths.append(figure_path)

    figure_path = charts_dir / f"{primary_metric}_distribution.png"
    plt.figure(figsize=(8, 4))
    dataframe[primary_metric].dropna().plot(kind="hist", bins=25, title=f"{primary_metric} distribution")
    plt.xlabel(primary_metric)
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()
    chart_paths.append(figure_path)

    if dimensions:
        top_dimension = dimensions[0]
        grouped = (
            dataframe[[top_dimension, primary_metric]]
            .dropna()
            .groupby(top_dimension)[primary_metric]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )
        if not grouped.empty:
            figure_path = charts_dir / f"{top_dimension}_{primary_metric}_top10.png"
            plt.figure(figsize=(10, 5))
            grouped.sort_values().plot(kind="barh", title=f"Top {top_dimension} by average {primary_metric}")
            plt.xlabel(primary_metric)
            plt.ylabel(top_dimension)
            plt.tight_layout()
            plt.savefig(figure_path)
            plt.close()
            chart_paths.append(figure_path)

            bottom_grouped = (
                dataframe[[top_dimension, primary_metric]]
                .dropna()
                .groupby(top_dimension)[primary_metric]
                .mean()
                .sort_values(ascending=True)
                .head(10)
            )
            figure_path = charts_dir / f"{top_dimension}_{primary_metric}_bottom10.png"
            plt.figure(figsize=(10, 5))
            bottom_grouped.plot(kind="barh", title=f"Bottom {top_dimension} by average {primary_metric}")
            plt.xlabel(primary_metric)
            plt.ylabel(top_dimension)
            plt.tight_layout()
            plt.savefig(figure_path)
            plt.close()
            chart_paths.append(figure_path)

    holiday_dimension = next((column for column in dimensions if "holiday" in column.lower()), None)
    if holiday_dimension:
        holiday_view = (
            dataframe[[holiday_dimension, primary_metric]]
            .dropna()
            .groupby(holiday_dimension)[primary_metric]
            .mean()
            .sort_index()
        )
        if not holiday_view.empty:
            figure_path = charts_dir / f"{holiday_dimension}_{primary_metric}.png"
            plt.figure(figsize=(7, 4))
            holiday_view.plot(kind="bar", title=f"{primary_metric} by {holiday_dimension}")
            plt.xlabel(holiday_dimension)
            plt.ylabel(primary_metric)
            plt.tight_layout()
            plt.savefig(figure_path)
            plt.close()
            chart_paths.append(figure_path)

    return chart_paths


def _build_anomaly_notes(dataframe: pd.DataFrame, metric: str | None) -> list[str]:
    if metric is None:
        return []

    series = dataframe[metric].dropna()
    if series.empty:
        return []

    mean = series.mean()
    std = series.std()
    if std == 0 or pd.isna(std):
        return []

    outlier_mask = (dataframe[metric] - mean).abs() > (2 * std)
    outlier_count = int(outlier_mask.sum())
    if not outlier_count:
        return []

    notes = [f"- `{metric}` shows {outlier_count} potential outliers using a 2-sigma rule."]
    date_column = _find_date_column(dataframe)
    context_columns = [column for column in ["Store", "Dept", "Department"] if column in dataframe.columns]
    preview_columns = context_columns + ([date_column] if date_column else []) + [metric]
    preview = dataframe.loc[outlier_mask, preview_columns].head(5)
    for _, row in preview.iterrows():
        parts = []
        for column in preview_columns:
            value = row[column]
            if pd.isna(value):
                continue
            if hasattr(value, "strftime"):
                value = value.strftime("%Y-%m-%d")
            parts.append(f"{column}={value}")
        notes.append(f"- Outlier sample: {', '.join(parts)}")
    return notes


def _build_dimension_table(dataframe: pd.DataFrame, dimension: str, metric: str) -> str:
    grouped = (
        dataframe[[dimension, metric]]
        .dropna()
        .groupby(dimension)[metric]
        .agg(["mean", "sum", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
        .head(10)
    )
    grouped.columns = [dimension, f"avg_{metric}", f"total_{metric}", "records"]
    return _format_table(grouped.round(2))


def _build_department_section(dataframe: pd.DataFrame, metric: str | None) -> str:
    department_column = _find_dimension_column(dataframe, "dept", "department")
    if metric is None or department_column is None:
        return "Department-level analysis is not available for this dataset."

    grouped = (
        dataframe[[department_column, metric]]
        .dropna()
        .groupby(department_column)[metric]
        .agg(["mean", "sum", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    if grouped.empty:
        return "Department-level analysis is not available for this dataset."

    grouped.columns = [department_column, f"avg_{metric}", f"total_{metric}", "records"]
    top_table = _format_table(grouped.head(10).round(2))
    bottom_table = _format_table(grouped.sort_values(f"avg_{metric}", ascending=True).head(10).round(2))
    return f"### Top Departments\n\n{top_table}\n\n### Bottom Departments\n\n{bottom_table}"


def _build_driver_summary(dataframe: pd.DataFrame, metric: str | None) -> str:
    if metric is None:
        return "- No continuous business metric was available for driver analysis."

    candidate_drivers = [
        column
        for column in dataframe.select_dtypes(include="number").columns
        if column != metric and not _is_dimension_column(dataframe, column)
    ]
    if not candidate_drivers:
        return "- No additional continuous drivers were available for correlation analysis."

    correlations = dataframe[[metric] + candidate_drivers].corr(numeric_only=True)[metric].drop(metric)
    correlations = correlations.reindex(correlations.abs().sort_values(ascending=False).index)
    top = correlations.head(4)
    if top.empty:
        return "- Driver correlations could not be computed."

    lines = []
    for column, value in top.items():
        direction = "positive" if value > 0 else "negative"
        lines.append(f"- `{column}` has a {direction} correlation of {value:.3f} with `{metric}`.")
    return "\n".join(lines)


def _build_store_rankings(dataframe: pd.DataFrame, metric: str | None) -> str:
    store_column = _find_dimension_column(dataframe, "store")
    if metric is None or store_column is None:
        return "Store-level ranking is not available."

    grouped = (
        dataframe[[store_column, metric]]
        .dropna()
        .groupby(store_column)[metric]
        .agg(["mean", "sum", "count"])
        .reset_index()
    )
    if grouped.empty:
        return "Store-level ranking is not available."

    grouped.columns = [store_column, f"avg_{metric}", f"total_{metric}", "records"]
    top_table = _format_table(grouped.sort_values(f"avg_{metric}", ascending=False).head(5).round(2))
    bottom_table = _format_table(grouped.sort_values(f"avg_{metric}", ascending=True).head(5).round(2))
    return f"### Top 5 Stores\n\n{top_table}\n\n### Bottom 5 Stores\n\n{bottom_table}"


def _build_seasonality_section(dataframe: pd.DataFrame, metric: str | None) -> str:
    date_column = _find_date_column(dataframe)
    if metric is None or date_column is None:
        return "Seasonality analysis is not available."

    seasonality = dataframe[[date_column, metric]].dropna().copy()
    seasonality["Month"] = seasonality[date_column].dt.month_name().str.slice(0, 3)
    seasonality["Quarter"] = "Q" + seasonality[date_column].dt.quarter.astype(str)

    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly = seasonality.groupby("Month")[metric].mean().reindex(month_order).dropna().reset_index()
    monthly.columns = ["Month", f"avg_{metric}"]
    quarterly = seasonality.groupby("Quarter")[metric].mean().reset_index()
    quarterly.columns = ["Quarter", f"avg_{metric}"]

    summary_lines: list[str] = []
    if not monthly.empty:
        best_month = monthly.sort_values(f"avg_{metric}", ascending=False).iloc[0]
        weakest_month = monthly.sort_values(f"avg_{metric}", ascending=True).iloc[0]
        summary_lines.append(
            f"- Strongest month: `{best_month['Month']}` at {_format_value(best_month[f'avg_{metric}'])} average {metric}."
        )
        summary_lines.append(
            f"- Weakest month: `{weakest_month['Month']}` at {_format_value(weakest_month[f'avg_{metric}'])} average {metric}."
        )

    sections = []
    if summary_lines:
        sections.append("\n".join(summary_lines))
    if not monthly.empty:
        sections.append("### Monthly Pattern\n\n" + _format_table(monthly.round(2)))
    if not quarterly.empty:
        sections.append("### Quarterly Pattern\n\n" + _format_table(quarterly.round(2)))
    return "\n\n".join(sections) if sections else "Seasonality analysis is not available."


def _build_holiday_section(dataframe: pd.DataFrame, metric: str | None) -> str:
    holiday_column = _find_dimension_column(dataframe, "holiday")
    date_column = _find_date_column(dataframe)
    if metric is None or holiday_column is None:
        return "Holiday impact analysis is not available."

    holiday_rows = dataframe.loc[dataframe[holiday_column] == 1].copy()
    if holiday_rows.empty:
        return "No holiday rows were found in the dataset."

    summary = (
        dataframe[[holiday_column, metric]]
        .dropna()
        .groupby(holiday_column)[metric]
        .mean()
        .sort_index()
    )
    baseline = float(summary.get(0, 0))
    holiday_avg = float(summary.get(1, 0))
    uplift = ((holiday_avg - baseline) / baseline * 100) if baseline else 0

    sections = [
        f"- Holiday periods average {_format_value(holiday_avg)} in `{metric}` versus {_format_value(baseline)} for non-holiday periods.",
        f"- Estimated holiday uplift: {uplift:.2f}%.",
    ]

    preview_columns = [column for column in ["Store"] if column in holiday_rows.columns]
    if date_column:
        preview_columns.append(date_column)
    preview_columns.append(metric)
    strongest = holiday_rows[preview_columns].sort_values(metric, ascending=False).head(5).copy()
    if date_column and date_column in strongest.columns:
        strongest[date_column] = strongest[date_column].dt.strftime("%Y-%m-%d")
    sections.append("### Strongest Holiday Weeks\n\n" + _format_table(strongest.round(2)))
    return "\n\n".join(sections)


def _build_recommendations(dataframe: pd.DataFrame, metric: str | None) -> str:
    if metric is None:
        return "- No recommendations available because a primary business metric was not identified."

    notes: list[str] = []
    store_column = _find_dimension_column(dataframe, "store")
    if store_column:
        grouped = (
            dataframe[[store_column, metric]]
            .dropna()
            .groupby(store_column)[metric]
            .mean()
            .sort_values(ascending=False)
        )
        if len(grouped) >= 5:
            top_store = grouped.index[0]
            low_store = grouped.index[-1]
            notes.append(f"- Study operating patterns in store `{top_store}` and adapt the strongest practices to store `{low_store}`.")

    holiday_column = _find_dimension_column(dataframe, "holiday")
    if holiday_column:
        holiday_view = (
            dataframe[[holiday_column, metric]]
            .dropna()
            .groupby(holiday_column)[metric]
            .mean()
            .sort_index()
        )
        if 0 in holiday_view.index and 1 in holiday_view.index and holiday_view[1] > holiday_view[0]:
            notes.append("- Increase staffing, inventory, and promotions around holiday weeks because they outperform baseline periods.")

    driver_summary = _build_driver_summary(dataframe, metric)
    if "Unemployment" in driver_summary:
        notes.append("- Track macroeconomic conditions such as unemployment when planning sales targets because weak demand environments coincide with softer sales.")

    if not notes:
        notes.append("- Continue monitoring top-performing segments and repeat this analysis on the latest data to validate current strategies.")
    return "\n".join(notes)


def _build_forecast_section(dataframe: pd.DataFrame, metric: str | None) -> str:
    forecast = _build_forecast(dataframe, metric)
    if forecast is None:
        return "Forecasting is not available because the dataset does not contain enough dated history."

    forecast_table = forecast.forecast_frame.copy()
    forecast_table["Date"] = forecast_table["Date"].dt.strftime("%Y-%m-%d")
    forecast_table[metric] = forecast_table[metric].round(2)
    sections = ["\n".join(forecast.summary_lines), "### Forecast Table\n\n" + _format_table(forecast_table)]

    store_forecasts = _build_store_forecast_table(dataframe, metric, periods=4)
    if not store_forecasts.empty:
        sections.append("### Top Rising Stores\n\n" + _format_table(store_forecasts.head(10)))
        sections.append("### Top Softening Stores\n\n" + _format_table(store_forecasts.sort_values("change_pct", ascending=True).head(10)))
        sections.append("### All Store Forecasts\n\n" + _format_table(store_forecasts))

    return "\n\n".join(sections)


def _build_department_forecast_section(dataframe: pd.DataFrame, metric: str | None, periods: int = 4) -> str:
    department_column = _find_dimension_column(dataframe, "dept", "department")
    date_column = _find_date_column(dataframe)
    if metric is None or department_column is None or date_column is None:
        return "Department-level forecasting is not available for this dataset."

    department_summary = (
        dataframe[[department_column, metric]]
        .dropna()
        .groupby(department_column)[metric]
        .mean()
        .sort_values(ascending=False)
    )
    if department_summary.empty:
        return "Department-level forecasting is not available for this dataset."

    rows: list[dict[str, object]] = []
    selected_departments = list(department_summary.head(5).index) + list(department_summary.tail(5).index)
    seen: set[object] = set()
    ordered_departments = [dept for dept in selected_departments if not (dept in seen or seen.add(dept))]
    for department in ordered_departments:
        department_frame = dataframe.loc[dataframe[department_column] == department].copy()
        forecast = _build_forecast(department_frame, metric, periods=periods)
        if forecast is None:
            continue
        last_actual_series = (
            department_frame[[date_column, metric]]
            .dropna()
            .groupby(date_column)[metric]
            .mean()
            .sort_index()
        )
        last_actual = float(last_actual_series.iloc[-1])
        avg_forecast = float(forecast.forecast_frame[metric].mean())
        change_pct = ((avg_forecast - last_actual) / last_actual * 100) if last_actual else 0.0
        rows.append(
            {
                "Department": department,
                "latest_actual": round(last_actual, 2),
                f"avg_next_{periods}_weeks": round(avg_forecast, 2),
                "change_pct": round(change_pct, 2),
            }
        )

    if not rows:
        return "Department-level forecasting is not available for this dataset."
    return _format_table(pd.DataFrame(rows).sort_values("change_pct", ascending=False))


def _build_question_insights(dataframe: pd.DataFrame, question: str | None) -> str:
    primary_metric = _pick_primary_metric(dataframe, question)
    dimensions = _find_relevant_dimensions(dataframe, question)
    date_column = _find_date_column(dataframe)

    if not question:
        return "- No custom question provided. The report uses the default exploratory analysis flow."

    insights: list[str] = [f"- Question: {question}"]
    if primary_metric:
        series = dataframe[primary_metric].dropna()
        insights.append(
            f"- Primary metric: `{primary_metric}` with average {series.mean():,.2f}, median {series.median():,.2f}, and max {series.max():,.2f}."
        )

    if date_column and primary_metric:
        trend = (
            dataframe[[date_column, primary_metric]]
            .dropna()
            .groupby(date_column)[primary_metric]
            .mean()
            .sort_index()
        )
        if len(trend) >= 2:
            delta = trend.iloc[-1] - trend.iloc[0]
            direction = "up" if delta > 0 else "down" if delta < 0 else "flat"
            insights.append(
                f"- Over the observed period, `{primary_metric}` ends {direction} by {abs(delta):,.2f}."
            )

    for dimension in dimensions[:2]:
        if primary_metric is None:
            break
        grouped = (
            dataframe[[dimension, primary_metric]]
            .dropna()
            .groupby(dimension)[primary_metric]
            .mean()
            .sort_values(ascending=False)
        )
        if not grouped.empty:
            best_group = grouped.index[0]
            best_value = grouped.iloc[0]
            insights.append(
                f"- Highest average `{primary_metric}` is in `{dimension}={best_group}` at {best_value:,.2f}."
            )

    return "\n".join(insights)


def _build_executive_narrative(dataframe: pd.DataFrame, question: str | None) -> str:
    metric = _pick_primary_metric(dataframe, question)
    if metric is None:
        return "- A primary business metric was not identified for a narrative summary."

    lines: list[str] = []
    store_column = _find_dimension_column(dataframe, "store")
    if store_column:
        grouped = dataframe.groupby(store_column)[metric].mean().sort_values(ascending=False)
        if not grouped.empty:
            lines.append(
                f"- Sales concentration is led by store `{grouped.index[0]}`, which posts the highest average `{metric}` at {_format_value(grouped.iloc[0])}."
            )

    holiday_column = _find_dimension_column(dataframe, "holiday")
    if holiday_column:
        holiday_view = dataframe.groupby(holiday_column)[metric].mean().sort_index()
        if 0 in holiday_view.index and 1 in holiday_view.index:
            uplift = ((holiday_view[1] - holiday_view[0]) / holiday_view[0] * 100) if holiday_view[0] else 0
            lines.append(
                f"- Holiday periods lift average `{metric}` by {uplift:.2f}% compared with non-holiday weeks."
            )

    date_column = _find_date_column(dataframe)
    if date_column:
        monthly = (
            dataframe.assign(_month=dataframe[date_column].dt.month_name().str.slice(0, 3))
            .groupby("_month")[metric]
            .mean()
        )
        order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly = monthly.reindex(order).dropna()
        if not monthly.empty:
            lines.append(
                f"- Peak monthly demand appears in `{monthly.idxmax()}`, while `{monthly.idxmin()}` is the softest month in the observed period."
            )

    return "\n".join(lines) if lines else "- No executive narrative could be generated from the current dataset."


def build_report(
    dataframe: pd.DataFrame, chart_paths: list[Path], question: str | None = None
) -> str:
    summary_source = dataframe.copy()
    date_column = _find_date_column(summary_source)
    if date_column:
        summary_source[date_column] = summary_source[date_column].dt.strftime("%Y-%m-%d")
    summary_frame = summary_source.describe(include="all").fillna("")
    try:
        numeric_summary = summary_frame.to_markdown()
    except ImportError:
        numeric_summary = f"```text\n{summary_frame.to_string()}\n```"

    missing_values = dataframe.isna().sum().sort_values(ascending=False)
    top_missing = missing_values[missing_values > 0]
    primary_metric = _pick_primary_metric(dataframe, question)
    focus_metrics = _find_focus_metrics(dataframe, question)
    dimensions = _find_relevant_dimensions(dataframe, question)
    date_column = _find_date_column(dataframe)
    anomaly_notes = _build_anomaly_notes(dataframe, primary_metric)

    missing_section = (
        "\n".join(f"- `{column}`: {count}" for column, count in top_missing.items())
        if not top_missing.empty
        else "- No missing values detected."
    )
    chart_section = "\n".join(f"- `{path.as_posix()}`" for path in chart_paths) or "- No charts generated."
    anomaly_section = "\n".join(anomaly_notes) or "- No strong anomaly signals detected."
    question_section = _build_question_insights(dataframe, question)
    driver_section = _build_driver_summary(dataframe, primary_metric)
    narrative_section = _build_executive_narrative(dataframe, question)
    store_ranking_section = _build_store_rankings(dataframe, primary_metric)
    seasonality_section = _build_seasonality_section(dataframe, primary_metric)
    holiday_section = _build_holiday_section(dataframe, primary_metric)
    department_section = _build_department_section(dataframe, primary_metric)
    recommendation_section = _build_recommendations(dataframe, primary_metric)
    forecast_section = _build_forecast_section(dataframe, primary_metric)
    department_forecast_section = _build_department_forecast_section(dataframe, primary_metric)

    performance_sections: list[str] = []
    for dimension in dimensions:
        if primary_metric is None:
            break
        performance_sections.append(f"### {dimension} Breakdown\n\n{_build_dimension_table(dataframe, dimension, primary_metric)}")
    performance_section = "\n\n".join(performance_sections) or "No business dimensions were detected."

    return f"""# Analysis Report

## Dataset Overview

- Rows: {len(dataframe)}
- Columns: {len(dataframe.columns)}
- Date column: {date_column or "None"}
- Primary metric: {primary_metric or "None"}
- Focus metrics: {", ".join(focus_metrics) or "None"}
- Business dimensions: {", ".join(dimensions) or "None"}

## Analyst Prompt

{question_section}

## Missing Data

{missing_section}

## Executive Summary

{narrative_section}

## Key Drivers

{driver_section}

## Business Breakdown

{performance_section}

## Store Rankings

{store_ranking_section}

## Seasonality

{seasonality_section}

## Holiday Impact

{holiday_section}

## Department Analysis

{department_section}

## Recommendations

{recommendation_section}

## Forecast

{forecast_section}

## Department Forecast

{department_forecast_section}

## Statistical Summary

{numeric_summary}

## Trends And Anomalies

{anomaly_section}

## Generated Charts

{chart_section}
"""


def analyze_dataset(
    dataset_path: str | Path, charts_dir: str | Path, question: str | None = None
) -> AnalysisResult:
    dataframe = load_dataset(dataset_path)
    dataframe = _parse_dates(dataframe)
    dataframe = _coerce_numeric_columns(dataframe)
    chart_paths = create_charts(dataframe, charts_dir, question=question)
    summary_markdown = build_report(dataframe, chart_paths, question=question)
    store_forecast_table = _build_store_forecast_table(dataframe, _pick_primary_metric(dataframe, question), periods=4)
    rising_store_forecast_table = None
    softening_store_forecast_table = None
    if not store_forecast_table.empty:
        rising_store_forecast_table = store_forecast_table.head(10).copy()
        softening_store_forecast_table = store_forecast_table.sort_values("change_pct", ascending=True).head(10).copy()

    return AnalysisResult(
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        numeric_columns=dataframe.select_dtypes(include="number").columns.tolist(),
        missing_values={column: int(count) for column, count in dataframe.isna().sum().items()},
        summary_markdown=summary_markdown,
        chart_paths=chart_paths,
        store_forecast_table=store_forecast_table if not store_forecast_table.empty else None,
        rising_store_forecast_table=rising_store_forecast_table,
        softening_store_forecast_table=softening_store_forecast_table,
    )
