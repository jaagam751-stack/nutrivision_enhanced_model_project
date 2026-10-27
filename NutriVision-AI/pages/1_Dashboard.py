"""Dashboard — today's nutrition, trends and history at a glance (Phase 5).

Reads only what has been **explicitly saved** to the database. Charts are built
with Plotly by ``utils/analytics.py``; every figure degrades to an explanatory
placeholder when there is no data, so the page is useful from the first day.

The "latest AI recommendation" card is honest by design: it shows the newest
message the Gemini coach actually wrote (Phase 7) and explains the next step when
there is none, instead of inventing advice.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import streamlit as st

from config import (
    APP_BUILD,
    APP_FOOTER_NOTE,
    APP_ICON,
    APP_NAME,
    CURRENT_PHASE,
    TREND_WINDOW_DAYS,
)
from utils import analytics, ui
from utils.data_access import (
    db_ready,
    db_status,
    history_for_range,
    latest_recommendation,
    latest_stored_prediction,
    recent_meals,
    require_database,
    require_stored_profile,
    today_summary,
    total_meals,
)
from utils.helpers import (
    UserProfile,
    format_number,
    friendly_timestamp,
    get_profile,
    init_session_state,
)
from utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title=f"{APP_NAME} · Dashboard",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def render_today(profile: UserProfile, summary, meals) -> None:
    """Today's intake against the profile's daily targets."""
    ui.section("Today", "Totals from the meals you have saved today (UTC day).", icon="📅")

    target = float(profile.daily_calorie_target)
    protein_target = float(profile.protein_target_g)
    fiber_target = float(profile.fiber_target_g)

    calorie_pct = (summary.calories / target * 100.0) if target else 0.0
    remaining = target - summary.calories

    col1, col2, col3, col4 = st.columns(4, gap="medium")
    with col1:
        ui.metric_card(
            "Energy today",
            format_number(summary.calories),
            "kcal",
            f"{calorie_pct:.0f}% of {format_number(target)} kcal target",
            tone="accent",
        )
    with col2:
        ui.metric_card(
            "Protein",
            format_number(summary.protein, 1),
            "g",
            f"{summary.protein / protein_target * 100:.0f}% of {protein_target:,.0f} g target"
            if protein_target
            else "no target set",
        )
    with col3:
        ui.metric_card(
            "Fibre",
            format_number(summary.fiber, 1),
            "g",
            f"{summary.fiber / fiber_target * 100:.0f}% of {fiber_target:,.0f} g target"
            if fiber_target
            else "no target set",
        )
    with col4:
        ui.metric_card(
            "Average health score",
            format_number(summary.avg_health_score, 1),
            "/ 100",
            f"{summary.meals} meal{'s' if summary.meals != 1 else ''} logged today",
            tone="good" if summary.avg_health_score >= 62 else ("warn" if summary.avg_health_score >= 45 else "bad"),
        )

    if summary.is_empty:
        ui.notice(
            "No meals saved today yet. Go to **Upload Meal**, analyse a photo and press "
            "<strong>Save Meal</strong> — nothing is stored automatically.",
            tone="info",
            icon="🍽️",
        )
        return

    # Progress toward the daily energy target, with an honest over/under message.
    progress = min(calorie_pct / 100.0, 1.0)
    st.progress(progress)
    if remaining >= 0:
        st.caption(
            f"{format_number(remaining)} kcal remaining of your {format_number(target)} kcal target "
            f"for today ({calorie_pct:.0f}% used)."
        )
    else:
        st.caption(
            f"{format_number(abs(remaining))} kcal above your {format_number(target)} kcal target today "
            f"({calorie_pct:.0f}% used). One day over target is normal — what matters is the weekly pattern."
        )

    left, right = st.columns([1, 1], gap="large")
    with left:
        ui.card(title="Macronutrient totals", icon="🧬", body_html=ui.rows_html(summary.as_rows()))
    with right:
        st.plotly_chart(
            analytics.macro_donut_figure(summary, title="Today's energy split"),
            width="stretch",
            key="nv_dash_donut_today",
        )

    ui.notice(
        "Fat and carbohydrate have no personal target in this project, so they are shown as totals "
        "and as a share of today's energy rather than as a pass/fail.",
        tone="info",
        icon="🧮",
    )


def render_trends(profile: UserProfile, user_id: int) -> None:
    """Daily and weekly trends over the trend window (and all history for weekly)."""
    ui.section(
        "Trends",
        f"Daily charts cover the last {TREND_WINDOW_DAYS} days; the weekly view uses your whole history.",
        icon="📈",
    )

    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=TREND_WINDOW_DAYS - 1)

    bundle = history_for_range(user_id, start=start, end=today)
    daily = analytics.daily_frame(bundle.daily)
    filled = analytics.fill_missing_days(daily, start, today)

    all_time = history_for_range(user_id)
    weekly = analytics.weekly_frame(analytics.daily_frame(all_time.daily))

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.plotly_chart(
            analytics.calorie_trend_figure(
                filled, target=profile.daily_calorie_target, title=f"Daily energy · last {TREND_WINDOW_DAYS} days"
            ),
            width="stretch",
            key="nv_dash_calorie_trend",
        )
    with right:
        st.plotly_chart(
            analytics.health_score_trend_figure(filled, title="Health score per day"),
            width="stretch",
            key="nv_dash_score_trend",
        )

    left2, right2 = st.columns([1.35, 1], gap="large")
    with left2:
        st.plotly_chart(
            analytics.weekly_average_figure(
                weekly, target=profile.daily_calorie_target, title="Weekly average energy (days with meals)"
            ),
            width="stretch",
            key="nv_dash_weekly",
        )
    with right2:
        st.plotly_chart(
            analytics.macro_stack_figure(filled, title="Macronutrients per day"),
            width="stretch",
            key="nv_dash_macro_stack",
        )

    if bundle.summary.is_empty and all_time.summary.is_empty:
        ui.notice(
            "These charts fill up as you save meals. With a few days of history, the Predictive "
            "Nutrition Intelligence Engine (Phase 6) will also be able to forecast your next week.",
            tone="info",
            icon="📊",
        )
        return

    # Textual summary of the same window, so the numbers are readable too.
    logged_days = int((filled["meals"] > 0).sum())
    avg_calories = float(filled.loc[filled["meals"] > 0, "calories"].mean()) if logged_days else 0.0
    target = float(profile.daily_calorie_target)
    if logged_days:
        comparison = "close to" if abs(avg_calories - target) <= target * 0.1 else (
            "below" if avg_calories < target else "above"
        )
        ui.notice(
            f"Over the last {TREND_WINDOW_DAYS} days you logged <strong>{logged_days}</strong> day"
            f"{'s' if logged_days != 1 else ''} and averaged <strong>{avg_calories:,.0f} kcal</strong> "
            f"on those days — {comparison} your {target:,.0f} kcal target. Average protein "
            f"{float(filled.loc[filled['meals'] > 0, 'protein'].mean()):,.0f} g and fibre "
            f"{float(filled.loc[filled['meals'] > 0, 'fiber'].mean()):,.0f} g per logged day.",
            tone="info",
            icon="🧾",
        )


def render_recent_meals(user_id: int) -> None:
    """The last few saved meals, with a link to the full history."""
    ui.section("Recent meals", "Newest first — the full history has filters and deletion.", icon="🍱")

    meals = recent_meals(user_id, limit=6)
    if not meals:
        ui.notice(
            "No meals saved yet. Once you save one, it appears here and in the Meal History page.",
            tone="info",
            icon="🧾",
        )
        return

    frame = analytics.meals_frame(meals)
    st.dataframe(frame, hide_index=True, width="stretch")
    st.caption(f"{len(meals)} of {total_meals(user_id)} saved meals shown. Open **Meal History** for filters.")


def render_prediction_card() -> None:
    """The newest *saved* PNIE forecast.

    Only a stored prediction is shown (one cheap read) — the full engine run,
    the explanation and the save button live on the **Predictions** page, so the
    dashboard stays fast. When nothing is stored yet the card says so and points
    there instead of showing an invented number.
    """
    ui.section("Next week (PNIE)", "The latest forecast you chose to store.", icon="🔮")

    profile = get_profile()
    stored = None
    if profile is not None and getattr(profile, "user_id", None) and db_ready():
        stored = latest_stored_prediction(int(profile.user_id))

    if stored is not None and stored.predicted_next_week_avg_calories is not None:
        target = float(profile.daily_calorie_target or 0.0)
        value = float(stored.predicted_next_week_avg_calories)
        delta = value - target
        rows: list[tuple[str, str]] = [
            ("Forecast", f"{format_number(value)} kcal/day (next 7 days)"),
            ("Stored", friendly_timestamp(stored.prediction_date.isoformat())),
            (
                "Direction",
                {"rising": "trending up", "falling": "trending down", "stable": "holding steady"}.get(
                    stored.trend_direction or "", stored.trend_direction or "—"
                ),
            ),
            (
                "Against your target",
                f"{delta:+,.0f} kcal/day (target {format_number(target)})" if target else "—",
            ),
            (
                "Model",
                f"{stored.model_name or '—'} v{stored.model_version or '—'}"
                + (f" · MAE {format_number(stored.mae)} kcal" if stored.mae is not None else ""),
            ),
        ]
        ui.card(title="Latest stored forecast", icon="📈", body_html=ui.rows_html(rows))
        st.caption("Open **Predictions** to refresh the forecast, see its inputs or store a new one.")
        return

    ui.card(
        title="No forecast stored yet",
        icon="🔮",
        body_html=ui.html_raw(
            """
            <p class="nv-muted" style="margin-top:0">
            The PNIE forecast is not shown here because nothing has been stored yet. Open the
            <strong>Predictions</strong> page to run the engine on your saved meals — it will explain
            its inputs and save a forecast only when you press the button.
            </p>
            <p class="nv-muted" style="margin-bottom:0">
            If your history is still short, that page shows descriptive trends and the exact number
            of days the model is still waiting for.
            </p>
            """
        ),
    )


def render_recommendation_card() -> None:
    """Newest stored Gemini coaching message (Phase 7).

    Same rule as the forecast card above: only what was actually generated and
    stored is shown. Without a key or without a stored answer the card explains
    the next step instead of displaying canned advice.
    """
    ui.section("Latest AI recommendation", "Written by the Gemini 2.5 Flash coach.", icon="🤖")

    profile = get_profile()
    stored = None
    if profile is not None and getattr(profile, "user_id", None) and db_ready():
        stored = latest_recommendation(int(profile.user_id))

    if stored is not None and str(stored.recommendation_text or "").strip():
        ui.card(
            title=f"Stored recommendation · {friendly_timestamp(stored.timestamp.isoformat())} · {stored.source}",
            icon="💡",
            body_html=f"<div style='white-space:pre-wrap'>{stored.recommendation_text}</div>",
        )
        if stored.prompt:
            st.caption(f"Asked: “{stored.prompt}”")
        st.caption(
            "AI-generated text stored with its model name — open **AI Coach** to ask a new question "
            "or refresh this after logging more meals. Not medical advice."
        )
        return

    ui.card(
        title="No recommendation yet",
        icon="🤖",
        body_html=ui.html_raw(
            """
            <p class="nv-muted" style="margin-top:0">
            Nothing is shown here because no coaching message has been generated yet. Open the
            <strong>AI Coach</strong> page: it shows the exact snapshot it will send (your profile,
            your meal averages and the PNIE forecast), answers one question per press, and stores the
            reply here.
            </p>
            <p class="nv-muted" style="margin-bottom:0">
            The coach needs a free Gemini API key. Without one, this card stays empty while every
            other feature — recognition, nutrition, saving, dashboard, history and the forecast —
            keeps working.
            </p>
            """
        ),
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def render_no_profile() -> None:
    ui.hero(
        title="Welcome to NutriVision AI",
        subtitle="Create your profile first — the dashboard is personalised around your goals.",
        eyebrow="Dashboard · Step 0",
        icon="📊",
    )
    ui.notice(
        "Open the main **NutriVision AI** page from the sidebar, complete the short onboarding form, "
        "then return here.",
        tone="info",
        icon="👤",
    )

    ui.page_footer(CURRENT_PHASE, APP_BUILD, APP_FOOTER_NOTE)


def main() -> None:
    init_session_state()
    ui.load_css()

    profile: Optional[UserProfile] = get_profile()

    if profile is None:
        render_no_profile()
        return

    if not require_database() or not require_stored_profile(profile):
        return

    user_id = int(profile.user_id)
    status = db_status()

    ui.hero(
        title=f"{profile.name.split()[0]}'s dashboard",
        subtitle=(
            f"Goal: {profile.fitness_goal} · target {format_number(profile.daily_calorie_target)} kcal, "
            f"{format_number(profile.protein_target_g)} g protein and "
            f"{format_number(profile.fiber_target_g)} g fibre per day."
        ),
        eyebrow=f"Dashboard · {status.label}",
        icon="📊",
    )

    summary, meals = today_summary(user_id)
    render_today(profile, summary, meals)
    st.divider()
    render_trends(profile, user_id)
    st.divider()
    render_recent_meals(user_id)
    st.divider()
    render_prediction_card()
    st.divider()
    render_recommendation_card()

    ui.notice(
        "Energy and nutrient values are estimates from food composition tables, and “today” follows "
        "the UTC calendar day. This dashboard is general wellness information, not medical advice.",
        tone="warning",
        icon="⚠️",
    )
    ui.page_footer(CURRENT_PHASE, APP_BUILD, APP_FOOTER_NOTE)


main()
