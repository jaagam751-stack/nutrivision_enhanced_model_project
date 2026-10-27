"""Aggregation and Plotly chart builders for the dashboard and meal history (Phase 5).

Everything here is **pure**: it takes rows that the repository already fetched and
returns dataframes or Plotly figures. No database access, no Streamlit calls, so
the whole module is unit-testable without a server.

All colours come from :mod:`config` so the charts match the app's dark theme.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable, Mapping, Optional, Sequence

import pandas as pd
import plotly.graph_objects as go

from config import (
    CHART_HEIGHT,
    CHART_HEIGHT_SMALL,
    COLOR_ACCENT,
    COLOR_ACCENT_2,
    COLOR_BAD,
    COLOR_GRID,
    COLOR_GOOD,
    COLOR_MUTED,
    COLOR_TEXT,
    COLOR_WARN,
    MACRO_COLORS,
)

logger = logging.getLogger(__name__)

#: Column order used by both pages when rendering a meals table.
MEAL_TABLE_COLUMNS = [
    "When",
    "Food",
    "Portion (g)",
    "Energy (kcal)",
    "Protein (g)",
    "Fat (g)",
    "Carbs (g)",
    "Fibre (g)",
    "Health score",
]


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MealSummary:
    """Totals for a set of meals."""

    meals: int = 0
    calories: float = 0.0
    protein: float = 0.0
    fat: float = 0.0
    carbs: float = 0.0
    fiber: float = 0.0
    avg_health_score: float = 0.0
    avg_calories_per_meal: float = 0.0

    @property
    def is_empty(self) -> bool:
        return self.meals == 0

    def as_rows(self) -> list[tuple[str, str]]:
        """Label/value rows for a card."""
        return [
            ("Meals logged", f"{self.meals}"),
            ("Energy", f"{self.calories:,.0f} kcal"),
            ("Protein", f"{self.protein:,.1f} g"),
            ("Fat", f"{self.fat:,.1f} g"),
            ("Carbohydrates", f"{self.carbs:,.1f} g"),
            ("Fibre", f"{self.fiber:,.1f} g"),
            ("Average health score", f"{self.avg_health_score:,.1f} / 100"),
        ]


def summarise_meals(meals: Iterable[Any]) -> MealSummary:
    """Total and average the nutrition of any iterable of meal rows."""
    rows = list(meals)
    if not rows:
        return MealSummary()

    calories = sum(float(getattr(row, "calories", 0) or 0) for row in rows)
    protein = sum(float(getattr(row, "protein", 0) or 0) for row in rows)
    fat = sum(float(getattr(row, "fat", 0) or 0) for row in rows)
    carbs = sum(float(getattr(row, "carbohydrates", 0) or 0) for row in rows)
    fiber = sum(float(getattr(row, "fiber", 0) or 0) for row in rows)
    scores = [float(getattr(row, "health_score", 0) or 0) for row in rows]

    return MealSummary(
        meals=len(rows),
        calories=round(calories, 1),
        protein=round(protein, 1),
        fat=round(fat, 1),
        carbs=round(carbs, 1),
        fiber=round(fiber, 1),
        avg_health_score=round(sum(scores) / len(scores), 1),
        avg_calories_per_meal=round(calories / len(rows), 1),
    )


def meals_frame(meals: Sequence[Any]) -> pd.DataFrame:
    """Turn meal rows into a display dataframe (newest first)."""
    if not meals:
        return pd.DataFrame(columns=MEAL_TABLE_COLUMNS)

    records = []
    for row in meals:
        stamp = getattr(row, "timestamp", None)
        records.append(
            {
                "When": _local(stamp).strftime("%d %b %Y · %H:%M") if stamp else "—",
                "Food": str(getattr(row, "food_name", "—")),
                "Portion (g)": round(float(getattr(row, "portion_grams", 0) or 0), 0),
                "Energy (kcal)": round(float(getattr(row, "calories", 0) or 0), 0),
                "Protein (g)": round(float(getattr(row, "protein", 0) or 0), 1),
                "Fat (g)": round(float(getattr(row, "fat", 0) or 0), 1),
                "Carbs (g)": round(float(getattr(row, "carbohydrates", 0) or 0), 1),
                "Fibre (g)": round(float(getattr(row, "fiber", 0) or 0), 1),
                "Health score": round(float(getattr(row, "health_score", 0) or 0), 1),
                "_timestamp": _local(stamp) if stamp else pd.NaT,
            }
        )

    frame = pd.DataFrame.from_records(records)
    return frame.sort_values("_timestamp", ascending=False).drop(columns=["_timestamp"]).reset_index(drop=True)


def daily_frame(daily_totals: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Convert ``repository.get_daily_totals`` output into a typed dataframe."""
    columns = ["date", "meals", "calories", "protein", "fat", "carbohydrates", "fiber", "avg_health_score"]
    if not daily_totals:
        return pd.DataFrame(columns=columns)

    frame = pd.DataFrame.from_records(daily_totals)
    frame["date"] = pd.to_datetime(frame["date"])
    # Rows may legitimately omit columns (a bare query, a future schema change),
    # so reindex to the full set instead of assuming every key is present.
    frame = frame.reindex(columns=columns)
    for column in columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return frame.sort_values("date").reset_index(drop=True)


def fill_missing_days(frame: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Insert zero rows for days with no meals, so charts show real gaps.

    Without this a two-day gap would be drawn as a straight line between the
    surrounding days, which would misrepresent the data.
    """
    columns = ["date", "meals", "calories", "protein", "fat", "carbohydrates", "fiber", "avg_health_score"]
    calendar = pd.DataFrame({"date": pd.date_range(start=start, end=end, freq="D")})
    if frame.empty:
        return calendar.assign(**{column: 0.0 for column in columns[1:]})

    merged = calendar.merge(frame, on="date", how="left")
    for column in columns[1:]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
    return merged


def weekly_frame(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily rows into ISO weeks (Monday start) with daily averages."""
    if daily.empty:
        return pd.DataFrame(columns=["week_start", "label", "avg_calories", "avg_protein", "avg_fiber", "days"])

    frame = daily.copy()
    frame["week_start"] = frame["date"].dt.to_period("W-SUN").apply(lambda period: period.start_time)
    grouped = (
        frame.groupby("week_start")
        .agg(
            days=("date", "count"),
            total_calories=("calories", "sum"),
            meals=("meals", "sum"),
            protein=("protein", "sum"),
            fiber=("fiber", "sum"),
        )
        .reset_index()
    )
    # Average over days *with meals*, not over the week length, so a partially
    # logged week is not artificially deflated.
    logged_days = frame[frame["meals"] > 0].groupby("week_start").size().rename("logged_days")
    grouped = grouped.merge(logged_days, on="week_start", how="left")
    # A week where nothing was logged is not a "0 kcal week" — it is missing data.
    # Showing it as zero would misrepresent the history, so those weeks are dropped
    # and the average is taken over days that actually have meals.
    grouped = grouped[grouped["logged_days"].notna() & (grouped["logged_days"] > 0)].copy()
    if grouped.empty:
        return pd.DataFrame(columns=["week_start", "label", "avg_calories", "avg_protein", "avg_fiber", "days"])
    grouped["avg_calories"] = (grouped["total_calories"] / grouped["logged_days"]).round(0)
    grouped["avg_protein"] = (grouped["protein"] / grouped["logged_days"]).round(1)
    grouped["avg_fiber"] = (grouped["fiber"] / grouped["logged_days"]).round(1)
    grouped["label"] = grouped["week_start"].dt.strftime("%d %b")
    return grouped.sort_values("week_start").reset_index(drop=True)


def _local(stamp: Any) -> Optional[pd.Timestamp]:
    """Best-effort conversion of a (possibly timezone-aware) timestamp."""
    try:
        parsed = pd.to_datetime(stamp)
        return parsed.tz_convert(None) if getattr(parsed, "tzinfo", None) else parsed
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Chart theme
# --------------------------------------------------------------------------- #
def _base_layout(title: str, height: int, y_title: str = "", x_title: str = "") -> dict[str, Any]:
    """Shared dark-theme layout so every chart looks like part of the app."""
    return {
        "title": {"text": title, "font": {"size": 15, "color": COLOR_TEXT}},
        "height": height,
        "margin": {"l": 10, "r": 10, "t": 46, "b": 10},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": COLOR_MUTED, "size": 12},
        "xaxis": {
            "title": {"text": x_title, "font": {"size": 12}},
            "gridcolor": COLOR_GRID,
            "zeroline": False,
        },
        "yaxis": {
            "title": {"text": y_title, "font": {"size": 12}},
            "gridcolor": COLOR_GRID,
            "zeroline": False,
        },
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0, "bgcolor": "rgba(0,0,0,0)"},
        "hoverlabel": {"bgcolor": "#111726", "font": {"color": COLOR_TEXT}},
    }


def _empty_figure(title: str, message: str, height: int = CHART_HEIGHT) -> go.Figure:
    """A chart-shaped placeholder that explains why there is nothing to draw."""
    figure = go.Figure()
    figure.update_layout(**_base_layout(title, height))
    figure.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 13, "color": COLOR_MUTED},
    )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return figure


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def calorie_trend_figure(daily: pd.DataFrame, target: float | None = None, title: str = "Daily energy") -> go.Figure:
    """Bar chart of daily calories with an optional target reference line."""
    if daily.empty or float(daily["calories"].sum()) == 0:
        return _empty_figure(title, "No meals logged in this period yet.")

    figure = go.Figure()
    figure.add_bar(
        x=daily["date"],
        y=daily["calories"],
        name="Energy",
        marker={"color": COLOR_ACCENT, "line": {"width": 0}},
        hovertemplate="%{x|%d %b}<br>%{y:.0f} kcal<extra></extra>",
    )
    if target:
        figure.add_hline(
            y=float(target),
            line={"color": COLOR_ACCENT_2, "width": 2, "dash": "dash"},
            annotation_text=f"target {target:,.0f} kcal",
            annotation_position="top left",
            annotation_font={"color": COLOR_ACCENT_2, "size": 11},
        )
    figure.update_layout(**_base_layout(title, CHART_HEIGHT, y_title="kcal"))
    figure.update_xaxes(tickformat="%d %b")
    return figure


def macro_donut_figure(summary: MealSummary, title: str = "Where the energy comes from") -> go.Figure:
    """Donut of energy contributed by protein / carbohydrate / fat."""
    protein_kcal = summary.protein * 4
    carb_kcal = summary.carbs * 4
    fat_kcal = summary.fat * 9
    total = protein_kcal + carb_kcal + fat_kcal
    if total <= 0:
        return _empty_figure(title, "Log a meal to see its macronutrient split.", CHART_HEIGHT_SMALL)

    figure = go.Figure(
        go.Pie(
            labels=["Protein", "Carbohydrate", "Fat"],
            values=[protein_kcal, carb_kcal, fat_kcal],
            hole=0.58,
            marker={"colors": [MACRO_COLORS["protein"], MACRO_COLORS["carbs"], MACRO_COLORS["fat"]]},
            textinfo="label+percent",
            textfont={"size": 12, "color": "#0b0f19"},
            hovertemplate="%{label}: %{value:.0f} kcal (%{percent})<extra></extra>",
        )
    )
    figure.update_layout(**_base_layout(title, CHART_HEIGHT_SMALL))
    figure.update_layout(showlegend=False)
    return figure


def macro_stack_figure(daily: pd.DataFrame, title: str = "Macronutrients per day") -> go.Figure:
    """Stacked bars of daily protein / carbohydrate / fat in grams."""
    if daily.empty or float(daily[["protein", "carbohydrates", "fat"]].sum().sum()) == 0:
        return _empty_figure(title, "No meals logged in this period yet.")

    figure = go.Figure()
    for column, label, colour in (
        ("protein", "Protein", MACRO_COLORS["protein"]),
        ("carbohydrates", "Carbohydrate", MACRO_COLORS["carbs"]),
        ("fat", "Fat", MACRO_COLORS["fat"]),
    ):
        figure.add_bar(
            x=daily["date"],
            y=daily[column],
            name=label,
            marker={"color": colour, "line": {"width": 0}},
            hovertemplate="%{x|%d %b}<br>" + label + ": %{y:.0f} g<extra></extra>",
        )
    figure.update_layout(**_base_layout(title, CHART_HEIGHT, y_title="grams"))
    figure.update_layout(barmode="stack")
    figure.update_xaxes(tickformat="%d %b")
    return figure


def health_score_trend_figure(daily: pd.DataFrame, title: str = "Average health score per day") -> go.Figure:
    """Line chart of the average rule-based health score, on a fixed 0–100 scale."""
    scored = daily[daily["meals"] > 0] if not daily.empty else daily
    if scored.empty:
        return _empty_figure(title, "No scored meals in this period yet.")

    figure = go.Figure()
    figure.add_scatter(
        x=scored["date"],
        y=scored["avg_health_score"],
        mode="lines+markers",
        name="Health score",
        line={"color": COLOR_GOOD, "width": 2.5},
        marker={"size": 7, "color": COLOR_GOOD},
        fill="tozeroy",
        fillcolor="rgba(52, 211, 153, 0.12)",
        hovertemplate="%{x|%d %b}<br>score %{y:.1f}/100<extra></extra>",
    )
    layout = _base_layout(title, CHART_HEIGHT, y_title="score / 100")
    layout["yaxis"]["range"] = [0, 100]
    figure.update_layout(**layout)
    figure.update_xaxes(tickformat="%d %b")
    return figure


def weekly_average_figure(weekly: pd.DataFrame, target: float | None = None, title: str = "Weekly average energy") -> go.Figure:
    """Bars of average daily calories per week (days with meals only)."""
    if weekly.empty or float(weekly["avg_calories"].sum()) == 0:
        return _empty_figure(title, "Not enough history for a weekly view yet.")

    colours = [COLOR_ACCENT if not target else _deviation_colour(row, target) for row in weekly["avg_calories"]]
    figure = go.Figure(
        go.Bar(
            x=weekly["label"],
            y=weekly["avg_calories"],
            marker={"color": colours, "line": {"width": 0}},
            customdata=weekly[["days", "avg_protein", "avg_fiber"]].to_numpy(),
            hovertemplate=(
                "week of %{x}<br>avg %{y:.0f} kcal/day<br>"
                "%{customdata[0]} days logged · %{customdata[1]:.0f} g protein · "
                "%{customdata[2]:.0f} g fibre<extra></extra>"
            ),
        )
    )
    if target:
        figure.add_hline(
            y=float(target),
            line={"color": COLOR_ACCENT_2, "width": 2, "dash": "dash"},
            annotation_text=f"target {target:,.0f} kcal/day",
            annotation_position="top left",
            annotation_font={"color": COLOR_ACCENT_2, "size": 11},
        )
    figure.update_layout(**_base_layout(title, CHART_HEIGHT, y_title="kcal/day"))
    return figure


def _deviation_colour(value: float, target: float) -> str:
    """Colour a bar by how far it sits from the profile's daily target."""
    if target <= 0:
        return COLOR_ACCENT
    deviation = (float(value) - float(target)) / float(target)
    if abs(deviation) <= 0.10:
        return COLOR_GOOD
    if abs(deviation) <= 0.25:
        return COLOR_WARN
    return COLOR_BAD


def prediction_forecast_figure(
    daily: pd.DataFrame,
    value: float | None = None,
    low: float | None = None,
    high: float | None = None,
    forecast_start: date | None = None,
    forecast_end: date | None = None,
    target: float | None = None,
    title: str = "Energy history and next-week forecast",
) -> go.Figure:
    """Daily energy so far plus the PNIE forecast window (Phase 6).

    The history line uses **one point per logged day**, so gaps stay gaps instead
    of being drawn as 0 kcal. The dashed average line is a rolling mean over
    logged days only. When a forecast is supplied it is drawn as a dashed flat
    line across its window, with the hold-out error as a shaded band and a dotted
    divider marking where the recorded history ends.
    """
    if daily is None or daily.empty:
        return _empty_figure(title, "Save a few meals and the forecast chart appears here.")

    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date")
    logged = frame[pd.to_numeric(frame.get("meals"), errors="coerce").fillna(0) > 0]

    if logged.empty:
        return _empty_figure(title, "No logged days yet — charts need saved meals, not empty days.")

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=logged["date"],
            y=logged["calories"],
            mode="markers+lines",
            name="Daily energy",
            line={"color": COLOR_ACCENT, "width": 2},
            marker={"size": 5, "color": COLOR_ACCENT},
            hovertemplate="%{x|%d %b}<br>%{y:.0f} kcal<extra></extra>",
        )
    )

    if len(logged) >= 3:
        rolling = pd.to_numeric(logged["calories"], errors="coerce").rolling(7, min_periods=3).mean()
        figure.add_trace(
            go.Scatter(
                x=logged["date"],
                y=rolling,
                mode="lines",
                name="7-logged-day average",
                line={"color": COLOR_ACCENT_2, "width": 2, "dash": "dot"},
                hovertemplate="%{x|%d %b}<br>average %{y:.0f} kcal<extra></extra>",
            )
        )

    if value is not None and forecast_start is not None and forecast_end is not None:
        start = pd.Timestamp(forecast_start)
        end = pd.Timestamp(forecast_end)

        if low is not None and high is not None and high >= low:
            figure.add_trace(
                go.Scatter(
                    x=[start, end], y=[high, high], mode="lines",
                    line={"width": 0}, hoverinfo="skip", showlegend=False,
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=[start, end], y=[low, low], mode="lines",
                    line={"width": 0}, fill="tonexty",
                    fillcolor="rgba(124,92,255,0.20)",
                    name=f"Estimate range (±{abs(value - low):,.0f} kcal)",
                    hoverinfo="skip",
                )
            )

        figure.add_trace(
            go.Scatter(
                x=[start, end],
                y=[float(value), float(value)],
                mode="lines+markers",
                name="PNIE forecast (next week avg)",
                line={"color": COLOR_GOOD, "width": 3, "dash": "dash"},
                marker={"size": 8, "symbol": "diamond", "color": COLOR_GOOD},
                hovertemplate="forecast %{y:.0f} kcal/day<extra></extra>",
            )
        )
        # A date-axis vertical divider: drawn as a shape so the x value stays a
        # real date instead of an epoch number.
        figure.add_shape(
            type="line",
            xref="x",
            yref="paper",
            x0=start,
            x1=start,
            y0=0,
            y1=1,
            line={"color": COLOR_MUTED, "width": 1, "dash": "dot"},
        )
        figure.add_annotation(
            x=start,
            y=1.0,
            yref="paper",
            text="forecast →",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font={"color": COLOR_MUTED, "size": 11},
        )

    if target:
        figure.add_hline(
            y=float(target),
            line={"color": COLOR_WARN, "width": 2, "dash": "dash"},
            annotation_text=f"target {target:,.0f} kcal/day",
            annotation_position="bottom right",
            annotation_font={"color": COLOR_WARN, "size": 11},
        )

    layout = _base_layout(title, CHART_HEIGHT, y_title="kcal/day")
    figure.update_layout(**layout)
    figure.update_xaxes(tickformat="%d %b")
    return figure


def feature_importance_figure(
    importances: Mapping[str, float],
    descriptions: Mapping[str, str] | None = None,
    top: int = 8,
    title: str = "What the model relied on most",
) -> go.Figure:
    """Horizontal bars of the XGBoost feature importances (Phase 6).

    Importances come from the model that was actually fitted for this user, so
    the chart is a real explanation of *this* forecast — not a generic diagram.
    """
    if not importances:
        return _empty_figure(title, "The model has not been trained yet, so there is nothing to rank.")

    rows = sorted(importances.items(), key=lambda item: float(item[1]), reverse=True)[:top]
    rows.reverse()  # plotly draws the first row at the bottom of a horizontal chart
    names = [name.replace("_", " ") for name, _ in rows]
    shares = [float(share) * 100.0 for _, share in rows]
    hover = [
        (descriptions or {}).get(name, name.replace("_", " ")) for name, _ in rows
    ]

    figure = go.Figure(
        go.Bar(
            x=shares,
            y=names,
            orientation="h",
            marker={"color": COLOR_ACCENT, "line": {"width": 0}},
            customdata=hover,
            hovertemplate="%{customdata}<br>%{x:.1f}% of the model's decisions<extra></extra>",
        )
    )
    layout = _base_layout(title, max(CHART_HEIGHT_SMALL, 42 * len(rows) + 90), x_title="share of importance (%)")
    figure.update_layout(**layout, showlegend=False)
    figure.update_xaxes(ticksuffix="%")
    return figure
