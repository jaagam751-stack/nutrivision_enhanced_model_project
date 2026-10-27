"""Predictions — the PNIE next-week calorie forecast (Phase 6).

The page shows exactly what the engine produced and nothing more:

* a **forecast card** (next week's average daily calories, the chronological
  hold-out error as a range, and how it compares with the last recorded week and
  with the profile's target) — only when a real XGBoost model was trained;
* the **descriptive statistics** the engine computes from the same history,
  which are the honest answer whenever history is too short;
* an **explanation** of the model inputs, the evaluation and every guard that
  was applied;
* the **stored forecasts**, written to the ``predictions`` table only when the
  user presses the save button (mirroring the Save Meal rule from Phase 5).

Nothing on this page is fabricated: when the engine cannot predict, it says why
and what is missing.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import streamlit as st

from config import (
    APP_BUILD,
    APP_FOOTER_NOTE,
    APP_ICON,
    APP_NAME,
    CURRENT_PHASE,
    PNIE_HORIZON_DAYS,
    PNIE_LOOKBACK_DAYS,
    PNIE_MIN_HISTORY_DAYS,
    PNIE_MIN_LOGGED_DAYS,
    PNIE_MIN_TRAINING_ROWS,
    PNIE_ORIGIN_STRIDE_DAYS,
)
from pnie import FEATURE_DESCRIPTIONS, MODE_NO_DATA
from utils import analytics, ui
from utils.data_access import (
    pnie_daily_totals,
    require_database,
    require_stored_profile,
    run_pnie_for_user,
    store_prediction,
    stored_predictions,
)
from utils.helpers import (
    UserProfile,
    flash,
    format_number,
    friendly_timestamp,
    get_profile,
    init_session_state,
    render_pending_flash,
)
from utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

SESSION_SAVED_PREDICTION: str = "nv_saved_prediction_id"

st.set_page_config(
    page_title=f"{APP_NAME} · Predictions",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Forecast
# --------------------------------------------------------------------------- #
def render_forecast(profile: UserProfile, outcome) -> None:
    """The headline card: the forecast, or the reason there is none."""
    forecast = outcome.prediction
    if forecast is None:
        ui.section("Next week", "What the engine can say today.", icon="🔮")
        ui.notice(outcome.message, tone=outcome.tone, icon="🕒")
        return

    ui.section(
        "Next week",
        f"Forecast window {forecast.forecast_start:%d %b} – {forecast.forecast_end:%d %b} · "
        f"generated {forecast.generated_at:%d %b %Y %H:%M} UTC",
        icon="🔮",
    )

    target = float(profile.daily_calorie_target or 0.0)
    delta_target = forecast.value - target if target else None
    reference = forecast.recent_reference
    change = forecast.trend_change_kcal

    col1, col2, col3, col4 = st.columns(4, gap="medium")
    with col1:
        ui.metric_card(
            "Forecast average",
            format_number(forecast.value),
            "kcal/day",
            (
                f"estimated range {format_number(forecast.low)}–{format_number(forecast.high)} kcal"
                if forecast.low is not None and forecast.high is not None
                else "error estimate unavailable"
            ),
            tone="accent",
        )
    with col2:
        ui.metric_card(
            "Versus last week",
            f"{change:+,.0f}" if change is not None else "—",
            "kcal/day",
            (
                f"last recorded week averaged {format_number(reference)} kcal/day"
                if reference is not None
                else "no recent reference"
            ),
            tone={"rising": "warn", "falling": "good", "stable": "default"}.get(
                forecast.trend_direction, "default"
            ),
        )
    with col3:
        ui.metric_card(
            "Versus your target",
            f"{delta_target:+,.0f}" if delta_target is not None else "—",
            "kcal/day",
            f"target {format_number(target)} kcal/day · {profile.fitness_goal}",
            tone=(
                "good"
                if delta_target is not None and abs(delta_target) <= target * 0.10
                else ("warn" if delta_target is not None and abs(delta_target) <= target * 0.25 else "bad")
            ),
        )
    with col4:
        ui.metric_card(
            "Typical error",
            format_number(forecast.mae) if forecast.mae is not None else "—",
            "kcal/day",
            (
                f"from {forecast.test_samples} held-out week(s) · "
                f"{'better' if forecast.beats_baseline else 'not better'} than the mean baseline"
                if forecast.mae is not None
                else "no hold-out error available"
            ),
            tone="good" if forecast.beats_baseline else "warn",
        )

    ui.card(
        title="How to read this",
        icon="🧭",
        body_html=ui.html_raw(
            f"""
            <p style="margin-top:0" class="nv-muted">
            {forecast.explanation}
            </p>
            <p class="nv-muted" style="margin-bottom:0">
            Direction: <strong>{forecast.direction_label}</strong>
            ({'>' if forecast.trend_direction == 'rising' else ('<' if forecast.trend_direction == 'falling' else '≈')}
            the recent average). Model <code>{forecast.model_name}</code> v{forecast.model_version},
            trained on {forecast.training_samples} weekly window(s) of your own saved meals.
            </p>
            """
        ),
    )

    if forecast.beats_baseline is False:
        ui.notice(
            f"On the {forecast.test_samples} held-out week(s) this model did not beat the simplest "
            "baseline (always predicting your average intake), so treat the number as a rough "
            "continuation of your recent pattern. More logged weeks usually change that.",
            tone="warning",
            icon="📉",
        )

    if outcome.warnings:
        for warning in outcome.warnings:
            ui.notice(warning, tone="warning", icon="⚠️")


def render_forecast_chart(profile: UserProfile, outcome, daily_totals) -> None:
    """History plus the forecast window on one timeline."""
    ui.section("History and forecast", "One point per logged day; the forecast is the dashed line.",
               icon="📈")

    today = datetime.now(timezone.utc).date()
    totals = list(daily_totals or [])
    if totals:
        start = min(row["date"] for row in totals)
        frame = analytics.fill_missing_days(analytics.daily_frame(totals), start, today)
    else:
        frame = analytics.daily_frame(totals)

    forecast = outcome.prediction
    figure = analytics.prediction_forecast_figure(
        frame,
        value=forecast.value if forecast else None,
        low=forecast.low if forecast else None,
        high=forecast.high if forecast else None,
        forecast_start=forecast.forecast_start if forecast else None,
        forecast_end=forecast.forecast_end if forecast else None,
        target=float(profile.daily_calorie_target or 0) or None,
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    if forecast is None:
        st.caption(
            "The chart shows recorded history only — a forecast line appears once the model has "
            "enough weekly windows to train on."
        )
    else:
        st.caption(
            "Days without saved meals are not drawn as zero-calorie days; the dotted line averages "
            "logged days only."
        )


# --------------------------------------------------------------------------- #
# Explanation
# --------------------------------------------------------------------------- #
def render_model_card(outcome) -> None:
    """Explain the model: inputs, evaluation, guards. No black box."""
    ui.section("Why this number", "The model, its inputs and its measured error.", icon="🔬")

    training = outcome.training
    forecast = outcome.prediction
    dataset = outcome.dataset

    left, right = st.columns([1, 1], gap="large")
    with left:
        rows: list[tuple[str, str]] = [
            ("Model", "XGBoost gradient-boosted regression trees (one target)"),
            ("Target", "Average daily calories for the next 7 days"),
            ("Features", f"{len(FEATURE_DESCRIPTIONS)} inputs from the previous "
                         f"{PNIE_LOOKBACK_DAYS} days"),
            ("Training rows", str(training.rows) if training else "—"),
            ("Chronological split",
             f"{training.train_rows} oldest window(s) train · "
             f"{training.test_rows} newest window(s) held out"
             if training and training.trained else "not trained yet"),
        ]
        if training and training.trained:
            rows += [
                ("Hold-out MAE", f"{training.mae:,.0f} kcal/day"),
                ("Hold-out RMSE", f"{training.rmse:,.0f} kcal/day"),
                ("Baseline MAE (predict the mean)", f"{training.baseline_mae:,.0f} kcal/day"),
                ("Training MAE", f"{training.train_mae:,.0f} kcal/day"),
                ("Deployed model", "refit on all weekly windows after evaluation"),
            ]
        if dataset is not None:
            rows.append(("Windows discarded", str(dataset.dropped_total)))
        if forecast is not None and forecast.guards:
            rows.append(("Safety guards applied", str(len(forecast.guards))))
        ui.stat_rows(rows)

        if forecast is not None and forecast.guards:
            for guard in forecast.guards:
                ui.notice(guard, tone="warning", icon="🛡️")

    with right:
        importances = forecast.feature_importances if forecast else {}
        st.plotly_chart(
            analytics.feature_importance_figure(importances, descriptions=FEATURE_DESCRIPTIONS),
            width="stretch",
            config={"displayModeBar": False},
        )


def render_methodology() -> None:
    """The rules the engine follows, spelled out for the reader and the examiner."""
    with st.expander("How the PNIE works (method, leakage and limits)"):
        st.markdown(
            f"""
**One model, one question.** A single XGBoost regressor predicts the average
daily calories of the next {PNIE_HORIZON_DAYS} days. Everything else on this page
(the averages, the spread, the direction, the week-over-week change) is plain
arithmetic over your saved meals — there is no second model.

**Weekly windows.** A training row is built every
{PNIE_ORIGIN_STRIDE_DAYS} days: its features describe the
{PNIE_LOOKBACK_DAYS} days *before* that origin date, and its target is the
average of the *logged* days in the {PNIE_HORIZON_DAYS} days that follow it.
Days without saved meals are treated as missing data, never as zero-calorie days.

**No leakage.** Features are computed only from days strictly before the origin,
and the target only from days at or after it. The split is chronological — the
newest windows are held out — so the reported error comes from predicting weeks
the model never saw. A random split would let the model peek at the future.

**Honest thresholds.** Below {PNIE_MIN_HISTORY_DAYS} days of history and
{PNIE_MIN_LOGGED_DAYS} logged days you get statistics only; below
{PNIE_MIN_TRAINING_ROWS} usable weekly windows you get statistics and the exact
shortfall. The engine never fills that gap with a made-up forecast.

**Guards.** The prediction is clamped to a sane kcal range, and it is pulled
back if it strays more than 60 % from your recent average. Every clamp is listed
above instead of being applied silently.

**Limits.** The model learns from your own history, so it cannot know about a
festival week, illness or a new diet. Accuracy depends on how consistently you
log meals, and the reported error is an average — individual weeks will differ.
This is general wellness information, not medical advice.
            """
        )


# --------------------------------------------------------------------------- #
# Trends
# --------------------------------------------------------------------------- #
def render_trends(outcome, daily_totals) -> None:
    """Descriptive statistics — always available, and the fallback answer."""
    trends = outcome.trends
    ui.section("Your trends so far", "Descriptive statistics from the same saved meals.", icon="📊")

    if trends is None:
        ui.notice(
            "No saved meals yet, so there is nothing to summarise. Save your first meal and this "
            "section fills itself in.",
            tone="info",
            icon="🍽️",
        )
        return

    left, right = st.columns([1, 1], gap="large")
    with left:
        ui.stat_rows(trends.as_rows())
    with right:
        weekly = analytics.weekly_frame(analytics.daily_frame(daily_totals or []))
        if weekly.empty:
            ui.notice("Weekly averages appear once a few days are logged.", tone="info", icon="🗓️")
        else:
            table = weekly.tail(8).rename(
                columns={
                    "label": "Week of",
                    "avg_calories": "Avg kcal/day",
                    "avg_protein": "Protein (g)",
                    "avg_fiber": "Fibre (g)",
                    "logged_days": "Days logged",
                }
            )[["Week of", "Avg kcal/day", "Protein (g)", "Fibre (g)", "Days logged"]]
            st.dataframe(table, hide_index=True, width="stretch")

    ui.notice(trends.message, tone="info" if trends.has_enough_history else "warning", icon="🧮")

    if not trends.has_enough_history:
        requirements = outcome.requirements or {}
        history = requirements.get("history_days", {})
        logged = requirements.get("logged_days", {})
        ui.stat_rows(
            [
                ("Days of history", f"{history.get('have', trends.span_days)} / {history.get('need', PNIE_MIN_HISTORY_DAYS)} needed"),
                ("Logged days", f"{logged.get('have', trends.logged_days)} / {logged.get('need', PNIE_MIN_LOGGED_DAYS)} needed"),
            ]
        )


# --------------------------------------------------------------------------- #
# Stored forecasts
# --------------------------------------------------------------------------- #
def render_save_section(outcome, user_id: int) -> None:
    """Explicit save button — nothing is written to the ``predictions`` table without it."""
    ui.section("Stored forecasts", "Written to the predictions table only when you press save.",
               icon="💾")

    summary = outcome.prediction
    if summary is None:
        ui.notice(
            "There is no model forecast to store yet. Statistics and trends are never saved as "
            "predictions, so nothing is written until the model produces a real forecast.",
            tone="info",
            icon="🔒",
        )
        return

    saved_id = st.session_state.get(SESSION_SAVED_PREDICTION)
    if saved_id:
        ui.card(
            title=f"Saved as prediction #{saved_id}",
            icon="✅",
            body_html=ui.html_raw(
                f"""
                <p style="margin-top:0" class="nv-muted">
                {format_number(summary.value)} kcal/day for
                {summary.forecast_start:%d %b} – {summary.forecast_end:%d %b} is now stored with
                model <code>{summary.model_name}</code> v{summary.model_version},
                {summary.training_samples} training window(s) and MAE
                {format_number(summary.mae) if summary.mae is not None else '—'} kcal/day.
                </p>
                <p style="margin-bottom:0" class="nv-muted">
                The Gemini coach (Phase 7) reads the newest stored forecast when it writes your
                recommendation.
                </p>
                """
            ),
        )
        if st.button("Save another forecast", width="stretch"):
            st.session_state.pop(SESSION_SAVED_PREDICTION, None)
            st.rerun()
        return

    already = stored_predictions(user_id, limit=1)
    if already and already[0].predicted_next_week_avg_calories is not None:
        st.caption(
            f"Newest stored forecast: {format_number(already[0].predicted_next_week_avg_calories)} "
            f"kcal/day, saved {friendly_timestamp(already[0].prediction_date.isoformat())}."
        )

    if st.button("Save this forecast to my records", type="primary", width="stretch"):
        stored, message = store_prediction(user_id, outcome)
        if stored is not None:
            st.session_state[SESSION_SAVED_PREDICTION] = stored.prediction_id
            flash(message, "success")
        else:
            flash(message, "error")
        st.rerun()

    st.caption(
        "Saving is always an explicit action: the forecast is computed on every page load, but a row "
        "is only added here when you click the button."
    )


def render_stored_table(user_id: int) -> None:
    """Previous forecasts, newest first (also shows whether the model improved)."""
    rows = stored_predictions(user_id, limit=10)
    if not rows:
        st.caption("No forecasts stored yet.")
        return

    records = []
    for row in rows:
        stamp = row.prediction_date
        records.append(
            {
                "Stored (UTC)": stamp.strftime("%d %b %Y · %H:%M") if stamp else "—",
                "Forecast (kcal/day)": (
                    round(float(row.predicted_next_week_avg_calories), 1)
                    if row.predicted_next_week_avg_calories is not None
                    else None
                ),
                "Direction": row.trend_direction or "—",
                "Change (kcal)": (
                    round(float(row.trend_change_kcal), 1) if row.trend_change_kcal is not None else None
                ),
                "Model": f"{row.model_name or '—'} v{row.model_version or '—'}",
                "Windows": row.training_samples,
                "MAE (kcal)": round(float(row.mae), 1) if row.mae is not None else None,
            }
        )
    st.dataframe(pd.DataFrame(records), hide_index=True, width="stretch")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def render_no_profile() -> None:
    ui.hero(
        title="Predictions",
        subtitle="Create your profile and save a few meals first — the forecast needs your history.",
        eyebrow="PNIE · Step 0",
        icon="🔮",
    )
    ui.notice(
        "Open the main **NutriVision AI** page from the sidebar, complete onboarding, then save meals "
        "on the **Upload Meal** page. The engine needs saved meals before it will say anything about "
        "next week.",
        tone="info",
        icon="👤",
    )

    ui.page_footer(CURRENT_PHASE, APP_BUILD, APP_FOOTER_NOTE)


def render_intro(outcome, profile: UserProfile) -> None:
    requirements = outcome.requirements or {}
    history = requirements.get("history_days", {})
    logged = requirements.get("logged_days", {})
    training = requirements.get("training_rows", {})

    ui.hero(
        title="Next week, projected",
        subtitle=(
            f"{format_number(profile.daily_calorie_target)} kcal/day target · "
            f"{history.get('have', 0)} day(s) of history · {logged.get('have', 0)} logged day(s) · "
            f"{training.get('have') if training.get('have') is not None else '—'} weekly training window(s)"
        ),
        eyebrow=f"PNIE · {outcome.label}",
        icon="🔮",
    )


def main() -> None:
    init_session_state()
    ui.load_css()
    render_pending_flash()

    profile: Optional[UserProfile] = get_profile()
    if profile is None:
        render_no_profile()
        return
    if not require_database() or not require_stored_profile(profile):
        return

    user_id = int(profile.user_id)
    daily_totals = pnie_daily_totals(user_id)
    outcome = run_pnie_for_user(user_id, daily_totals=daily_totals)

    render_intro(outcome, profile)

    if outcome.mode == MODE_NO_DATA:
        ui.notice(outcome.message, tone="info", icon="🍽️")
        render_trends(outcome, daily_totals)
        return

    series = outcome.series
    if series is not None and series.start and series.end:
        st.caption(
            f"Engine status: **{outcome.label}** · history from "
            f"{series.start:%d %b %Y} to {series.end:%d %b %Y}"
        )
    else:
        st.caption(f"Engine status: **{outcome.label}**")

    render_forecast(profile, outcome)
    st.divider()
    render_forecast_chart(profile, outcome, daily_totals)
    st.divider()
    render_model_card(outcome)
    st.divider()
    render_trends(outcome, daily_totals)
    st.divider()
    render_save_section(outcome, user_id)
    render_stored_table(user_id)
    st.divider()
    render_methodology()

    ui.notice(
        "This forecast is a statistical estimate built from your own saved meals and is general "
        f"wellness information, not medical advice. It is not a substitute for a doctor or "
        f"dietitian. Today is treated as the UTC day, and the engine needs at least "
        f"{PNIE_MIN_HISTORY_DAYS} days of history before it predicts anything.",
        tone="warning",
        icon="⚠️",
    )
    ui.page_footer(CURRENT_PHASE, APP_BUILD, APP_FOOTER_NOTE)


main()
