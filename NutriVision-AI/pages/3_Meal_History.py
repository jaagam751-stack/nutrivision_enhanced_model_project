"""Meal History — filter, review, export and delete saved meals (Phase 5).

Every row here is a meal the user explicitly saved from the Upload Meal page.
Filters: date range (presets or a custom range) and a food-name search. The page
also shows totals for exactly the filtered set, offers a CSV download, and allows
deleting a single meal when a mistake was made — with a confirmation step,
because deletion cannot be undone.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import streamlit as st

from config import (
    APP_BUILD,
    APP_FOOTER_NOTE,
    APP_ICON,
    APP_NAME,
    CURRENT_PHASE,
    HISTORY_RANGE_OPTIONS,
    TREND_WINDOW_DAYS,
)
from utils import analytics, ui
from utils.data_access import (
    db_status,
    delete_meal_by_id,
    history_for_range,
    require_database,
    require_stored_profile,
    resolve_range,
)
from utils.helpers import UserProfile, flash, format_number, get_profile, init_session_state, render_pending_flash
from utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title=f"{APP_NAME} · Meal History",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

SESSION_DELETE_MEAL: str = "nv_delete_meal_id"


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #
def render_filters() -> tuple[Optional[date], Optional[date], str, int]:
    """Render the filter controls and return ``(start, end, search, limit)``."""
    ui.section("Filters", "The summary, table and chart all respect these filters.", icon="🎛️")

    left, middle, right = st.columns([1, 1, 1.4], gap="medium")

    with left:
        option = st.selectbox("Date range", HISTORY_RANGE_OPTIONS, index=2, key="nv_history_range")

    custom: Optional[tuple[date, date]] = None
    if option == "Custom range":
        with middle:
            today = datetime.now(timezone.utc).date()
            custom = st.date_input(
                "Pick a range",
                value=(today - timedelta(days=6), today),
                max_value=today,
                key="nv_history_custom",
            )
            if not isinstance(custom, (tuple, list)) or len(custom) != 2:
                st.caption("Select both a start and an end date.")
                custom = None
    else:
        with middle:
            st.caption("Presets cover today going back from it.")

    with right:
        search = st.text_input(
            "Filter by food name",
            placeholder="e.g. dal, chapati, samosa… (leave empty for all)",
            key="nv_history_search",
        ).strip()

    limit = st.slider(
        "Maximum rows shown", min_value=10, max_value=500, value=100, step=10, key="nv_history_limit"
    )

    start, end = resolve_range(option, custom if isinstance(custom, tuple) else None)
    return start, end, search, limit


def describe_range(start: Optional[date], end: Optional[date]) -> str:
    if start is None and end is None:
        return "all saved meals"
    if start is None or end is None:
        return f"{start or end}"
    if start == end:
        return start.strftime("%d %b %Y")
    return f"{start:%d %b %Y} → {end:%d %b %Y}"


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def render_summary(bundle, search: str, start, end) -> None:
    """Totals for the filtered set, plus a text description of the filter."""
    summary = bundle.summary
    ui.section("Filtered summary", f"{summary.meals} meal(s) · {describe_range(start, end)}.", icon="🧾")

    col1, col2, col3, col4 = st.columns(4, gap="medium")
    with col1:
        ui.metric_card("Meals", f"{summary.meals}", "", "matching the current filters")
    with col2:
        ui.metric_card("Energy", format_number(summary.calories), "kcal", "total in this range", tone="accent")
    with col3:
        ui.metric_card(
            "Average per meal",
            format_number(summary.avg_calories_per_meal),
            "kcal",
            f"protein {format_number(summary.protein, 1)} g total",
        )
    with col4:
        ui.metric_card(
            "Average health score",
            format_number(summary.avg_health_score, 1),
            "/ 100",
            "mean of the rule-based scores",
            tone="good" if summary.avg_health_score >= 62 else ("warn" if summary.avg_health_score >= 45 else "bad"),
        )

    if search:
        st.caption(f"Food-name filter applied: “{search}”")

    if summary.is_empty:
        ui.notice(
            "No meals match these filters. Widen the date range, clear the search box, or save a meal "
            "from the **Upload Meal** page.",
            tone="info",
            icon="🔍",
        )


def render_table(bundle, search: str, limit: int) -> None:
    """The meals table, plus CSV export and single-meal deletion."""
    if bundle.summary.is_empty:
        return

    meals = bundle.meals
    if search:
        needle = search.lower()
        meals = [meal for meal in meals if needle in str(meal.food_name).lower()]

    if not meals:
        ui.notice(f"No saved meal matches “{search}”.", tone="warning", icon="🔍")
        return

    ui.section("Saved meals", f"Showing {min(len(meals), limit)} of {len(meals)} matching meal(s).", icon="📋")
    frame = analytics.meals_frame(meals[:limit])
    st.dataframe(frame, hide_index=True, width="stretch")

    export_col, delete_col = st.columns([1, 1.3], gap="large")

    with export_col:
        st.download_button(
            "⬇️ Download these rows as CSV",
            data=frame.to_csv(index=False).encode("utf-8"),
            file_name=f"nutrivision_meals_{datetime.now(timezone.utc).date()}.csv",
            mime="text/csv",
            width="stretch",
        )
        st.caption("The CSV contains exactly the rows above (filters applied).")

    with delete_col:
        with st.expander("🗑️ Delete a meal"):
            options = {
                f"#{meal.meal_id} · {meal.food_name} · {analytics._local(meal.timestamp):%d %b %Y %H:%M} · "
                f"{float(meal.calories):.0f} kcal": meal.meal_id
                for meal in meals[:limit]
            }
            chosen = st.selectbox("Which meal?", list(options), key="nv_delete_choice")
            meal_id = options[chosen]

            if st.session_state.get(SESSION_DELETE_MEAL) != meal_id:
                if st.button("Delete this meal", key="nv_delete_request"):
                    st.session_state[SESSION_DELETE_MEAL] = meal_id
                    st.rerun()
            else:
                st.warning(f"Delete meal #{meal_id}? This cannot be undone.")
                confirm_col, cancel_col = st.columns(2)
                if confirm_col.button("Yes, delete", key="nv_delete_confirm", type="primary", width="stretch"):
                    removed, message = delete_meal_by_id(meal_id)
                    st.session_state.pop(SESSION_DELETE_MEAL, None)
                    flash(message, "success" if removed else "error")
                    st.rerun()
                if cancel_col.button("Cancel", key="nv_delete_cancel", width="stretch"):
                    st.session_state.pop(SESSION_DELETE_MEAL, None)
                    st.rerun()


def render_chart(bundle, start, end, search: str) -> None:
    """Calorie trend for the filtered range (fills gaps with zero days)."""
    if bundle.summary.is_empty:
        return

    daily = analytics.daily_frame(bundle.daily)
    today = datetime.now(timezone.utc).date()
    window_start = start or (today - timedelta(days=TREND_WINDOW_DAYS - 1))
    window_end = end or today
    if (window_end - window_start).days > 400:
        # A very long custom range would render hundreds of bars; the table still
        # shows every row, so the chart falls back to the last year.
        window_start = window_end - timedelta(days=364)
    filled = analytics.fill_missing_days(daily, window_start, window_end)

    ui.section("Energy in this range", "Days without saved meals appear as zero.", icon="📈")
    st.plotly_chart(
        analytics.calorie_trend_figure(filled, title=f"Daily energy · {describe_range(start, end)}"),
        width="stretch",
        key="nv_history_chart",
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def render_no_profile() -> None:
    ui.hero(
        title="Create your profile first",
        subtitle="Saved meals are linked to your profile, so this page needs one.",
        eyebrow="Meal History · Step 0",
        icon="📋",
    )

    ui.page_footer(CURRENT_PHASE, APP_BUILD, APP_FOOTER_NOTE)


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
    status = db_status()

    ui.hero(
        title="Meal history",
        subtitle=(
            "Every meal you saved, with its estimated nutrition and health score. Filter, export or "
            "delete rows as needed."
        ),
        eyebrow=f"Meal History · {status.label}",
        icon="📋",
    )

    start, end, search, limit = render_filters()
    bundle = history_for_range(user_id, start=start, end=end, limit=None)

    st.divider()
    render_summary(bundle, search, start, end)
    st.divider()
    render_table(bundle, search, limit)
    st.divider()
    render_chart(bundle, start, end, search)

    ui.notice(
        "Nutrition values are estimates from food composition tables and follow the UTC calendar day. "
        "Deleting a meal removes it permanently from the database.",
        tone="warning",
        icon="⚠️",
    )
    ui.page_footer(CURRENT_PHASE, APP_BUILD, APP_FOOTER_NOTE)


main()
