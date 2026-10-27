"""Profile & data — view, edit, and delete (Phase 8).

The last page of the project answers the three questions a user asks about a
data-collecting app:

1. **What do you know about me?** — the stored profile (``users`` row), the
   derived targets, and a plain-language explanation of how each target is
   calculated (Mifflin-St Jeor BMR, activity multiplier, goal delta).
2. **Let me fix it.** — the same validated form as onboarding, prefilled, which
   updates the existing row **in place** (never a duplicate account) and
   recalculates every target immediately.
3. **Delete it.** — two explicit, confirmed actions: *delete my meal history*
   (profile stays) and *delete everything* (profile + meals + forecasts +
   recommendations), both requiring the user to type DELETE first.

Nothing on this page is automatic: every write follows the project rule that a
button press is what changes the database, and every delete reports exactly how
many rows it removed.
"""

from __future__ import annotations

import logging
from typing import Optional

import streamlit as st

from config import (
    ACTIVITY_LEVELS,
    APP_BUILD,
    APP_FOOTER_NOTE,
    APP_ICON,
    APP_NAME,
    CURRENT_PHASE,
    DIET_PREFERENCES,
    FITNESS_GOALS,
    GENDERS,
    PROFILE_DELETE_CONFIRM_WORD,
)
from utils import ui
from utils.data_access import (
    db_ready,
    delete_meal_history,
    delete_profile_and_data,
    persist_profile,
    profile_overview,
    require_database,
    require_stored_profile,
)
from utils.helpers import (
    UserProfile,
    build_profile,
    clear_profile,
    flash,
    format_number,
    get_profile,
    init_session_state,
    render_pending_flash,
    reset_cached_resources,
    save_profile,
    validate_profile_inputs,
)
from utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title=f"{APP_NAME} · Profile",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

SESSION_DELETE_MEALS: str = "nv_profile_confirm_delete_meals"
SESSION_DELETE_ALL: str = "nv_profile_confirm_delete_all"
SESSION_SAVED_NOTE: str = "nv_profile_saved_note"


# --------------------------------------------------------------------------- #
# Header and targets
# --------------------------------------------------------------------------- #
def render_targets(profile: UserProfile) -> None:
    """The four headline numbers plus the full profile card."""
    delta = profile.goal_calorie_delta
    delta_note = (
        f"{format_number(abs(delta))} kcal "
        f"{'deficit' if delta < 0 else ('surplus' if delta > 0 else 'at maintenance')}"
    )

    col1, col2, col3, col4 = st.columns(4, gap="medium")
    with col1:
        ui.metric_card(
            "Body Mass Index",
            f"{profile.bmi:g}",
            "",
            f"{profile.bmi_category} · Asian-Indian BMI bands",
            tone=profile.bmi_tone,
        )
    with col2:
        ui.metric_card(
            "Maintenance",
            format_number(profile.maintenance_calories),
            "kcal/day",
            "TDEE from your activity level",
        )
    with col3:
        ui.metric_card(
            "Daily target",
            format_number(profile.daily_calorie_target),
            "kcal/day",
            delta_note,
            tone="accent",
        )
    with col4:
        ui.metric_card(
            "Protein target",
            format_number(profile.protein_target_g),
            "g/day",
            f"Fibre target {format_number(profile.fiber_target_g)} g/day",
        )

    left, right = st.columns([1, 1], gap="large")
    with left:
        ui.section("Your profile", "The values used by every other page.", icon="👤")
        ui.card(body_html=ui.rows_html(profile.summary_rows()))
    with right:
        ui.section("How the targets are calculated", "Every number is reproducible.", icon="🧮")
        rows = [
            ("BMR — Mifflin-St Jeor", f"{format_number(profile.bmr)} kcal/day"),
            ("Activity multiplier", f"× {profile.activity_multiplier:g} ({profile.activity_level})"),
            ("Maintenance (TDEE)", f"{format_number(profile.maintenance_calories)} kcal/day"),
            ("Goal adjustment", f"{profile.fitness_goal} → {delta_note}"),
            ("Daily calorie target", f"{format_number(profile.daily_calorie_target)} kcal/day"),
            ("Protein rule", f"{profile.weight_kg:g} kg × goal factor → {format_number(profile.protein_target_g)} g/day"),
            ("Fibre rule", f"14 g per 1000 kcal → {format_number(profile.fiber_target_g)} g/day"),
        ]
        ui.card(body_html=ui.rows_html(rows))
        st.caption(
            "Equations: Mifflin-St Jeor for BMR, standard activity multipliers, and "
            "goal-based adjustments. Estimates for general wellness — not a medical assessment."
        )


# --------------------------------------------------------------------------- #
# Edit form
# --------------------------------------------------------------------------- #
def render_edit_form(profile: UserProfile) -> None:
    """Prefilled onboarding form; saving updates the stored row in place."""
    ui.section("Edit your profile", "Saving recalculates every target immediately.", icon="✏️")

    with st.form("nv_profile_edit_form", clear_on_submit=False):
        left, right = st.columns(2, gap="large")

        with left:
            st.markdown("##### About you")
            name = st.text_input("Name *", value=profile.name, max_chars=60)
            age = st.number_input("Age (years) *", min_value=1, max_value=120, value=int(profile.age), step=1)
            gender = st.selectbox(
                "Gender *", list(GENDERS), index=_index_of(list(GENDERS), profile.gender)
            )
            height_cm = st.number_input(
                "Height (cm) *", min_value=50.0, max_value=250.0, value=float(profile.height_cm), step=0.5
            )
            weight_kg = st.number_input(
                "Weight (kg) *", min_value=20.0, max_value=400.0, value=float(profile.weight_kg), step=0.1
            )

        with right:
            st.markdown("##### Lifestyle & goal")
            activity_level = st.selectbox(
                "Activity level *",
                list(ACTIVITY_LEVELS),
                index=_index_of(list(ACTIVITY_LEVELS), profile.activity_level),
            )
            diet_preference = st.selectbox(
                "Diet preference *",
                list(DIET_PREFERENCES),
                index=_index_of(list(DIET_PREFERENCES), profile.diet_preference),
            )
            fitness_goal = st.selectbox(
                "Fitness goal *",
                list(FITNESS_GOALS),
                index=_index_of(list(FITNESS_GOALS), profile.fitness_goal),
            )
            st.caption(
                "Nothing is written until you press **Save changes**. Your stored row is updated "
                "in place — saving twice never creates a second account."
            )

        submitted = st.form_submit_button("💾 Save changes", type="primary", use_container_width=True)

    if not submitted:
        return

    raw = {
        "name": name,
        "age": age,
        "gender": gender,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "activity_level": activity_level,
        "diet_preference": diet_preference,
        "fitness_goal": fitness_goal,
    }
    errors = validate_profile_inputs(raw)
    if errors:
        logger.warning("Profile edit validation failed with %d error(s)", len(errors))
        ui.error_list(errors.values())
        return

    updated = build_profile(raw)
    updated.created_at = profile.created_at          # keep the original creation time
    old_target = profile.daily_calorie_target

    written_id: Optional[int] = None
    message = "Kept in this session only — the database is not available."
    if db_ready():
        written_id, message = persist_profile(updated, previous_user_id=profile.user_id)
    updated.user_id = written_id

    save_profile(updated)

    delta = updated.daily_calorie_target - old_target
    detail = (
        f"Daily target {format_number(old_target)} → {format_number(updated.daily_calorie_target)} kcal "
        f"({'+' if delta >= 0 else '−'}{format_number(abs(delta))} kcal)."
    )
    st.session_state[SESSION_SAVED_NOTE] = {"message": message, "detail": detail}
    logger.info("Profile updated | user_id=%s | %s", written_id, detail)
    st.rerun()


def _index_of(options: list[str], value: Optional[str]) -> int:
    """Position of ``value`` in ``options`` (0 when it is missing)."""
    return options.index(value) if value in options else 0


# --------------------------------------------------------------------------- #
# Storage summary
# --------------------------------------------------------------------------- #
def render_storage(user_id: Optional[int]) -> None:
    """What is stored, where, and what is deliberately not stored."""
    overview = profile_overview(user_id)
    ui.section("What is stored", "Read straight from the database just now.", icon="🗄️")

    if overview.error:
        ui.notice(
            f"'What is stored' could not be read: {overview.error}",
            tone="warning",
            icon="⚠️",
        )

    ui.key_value_grid(overview.as_rows(), columns=2)

    if not overview.stored:
        ui.notice(
            "There is no database row for this profile yet, so meals, forecasts and "
            "recommendations have nothing to attach to. Use **Save profile to database** on the "
            "home screen (or save this form after fixing the connection).",
            tone="warning",
            icon="⬆️",
        )

    left, right = st.columns([1, 1], gap="large")
    with left:
        ui.card(
            title="Not stored at all",
            icon="🔒",
            body_html=ui.html_raw(
                """
                <ul class="nv-muted" style="margin:0 0 0 1rem;padding:0">
                  <li><strong>Uploaded photos</strong> — analysed in memory and never written to disk.</li>
                  <li><strong>Your Gemini API key</strong> — stays in the environment, never in the database.</li>
                  <li><strong>Anything you did not save</strong> — predictions and recommendations exist only after a button press.</li>
                </ul>
                """
            ),
        )
    with right:
        ui.card(
            title="Where it is stored",
            icon="☁️",
            body_html=ui.html_raw(
                f"""
                <p class="nv-muted" style="margin-top:0">Backend: <strong>{overview.backend or 'unknown'}</strong>
                · {overview.location or 'location unknown'}</p>
                <p class="nv-muted" style="margin-bottom:0">The Neon PostgreSQL free tier is shared with your
                other projects and suspends itself after inactivity — the first query after a pause can take
                a second or two. The SQLite fallback is for local development only: its file is not part of
                a deployment and disappears with the container.</p>
                """
            ),
        )


# --------------------------------------------------------------------------- #
# Destructive actions
# --------------------------------------------------------------------------- #
def render_danger_zone(profile: UserProfile) -> None:
    """Two explicit deletes, each behind a typed confirmation."""
    ui.section("Delete data", "Both actions are permanent and ask you to type a word first.", icon="🗑️")

    user_id = profile.user_id

    with st.container():
        st.markdown('<div class="nv-danger">', unsafe_allow_html=True)
        ui.card(
            title="Delete meal history",
            icon="🍽️",
            body_html=(
                "Removes every saved meal for this profile. The profile itself, its stored forecasts "
                "and the AI recommendations stay, and the PNIE cache is refreshed immediately."
            ),
        )
        if not user_id:
            st.caption("Nothing to delete — this profile has no database row.")
        elif not st.session_state.get(SESSION_DELETE_MEALS):
            if st.button("🗑️ Delete meal history", key="nv_ask_delete_meals"):
                st.session_state[SESSION_DELETE_MEALS] = True
                st.rerun()
        else:
            typed = st.text_input(
                f"Type {PROFILE_DELETE_CONFIRM_WORD} to confirm",
                key="nv_confirm_delete_meals_input",
                placeholder=PROFILE_DELETE_CONFIRM_WORD,
            )
            confirm_col, cancel_col = st.columns([1, 1], gap="small")
            if confirm_col.button("Delete meals", key="nv_do_delete_meals", use_container_width=True):
                if typed.strip().upper() != PROFILE_DELETE_CONFIRM_WORD:
                    st.error(f"Type {PROFILE_DELETE_CONFIRM_WORD} exactly to confirm.")
                else:
                    removed, message = delete_meal_history(int(user_id))
                    st.session_state[SESSION_DELETE_MEALS] = False
                    flash(message if removed else f"Nothing deleted. {message}", "success" if removed else "info")
                    st.rerun()
            if cancel_col.button("Cancel", key="nv_cancel_delete_meals", use_container_width=True):
                st.session_state[SESSION_DELETE_MEALS] = False
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="nv-danger">', unsafe_allow_html=True)
        ui.card(
            title="Delete everything",
            icon="💥",
            body_html=(
                "Removes the profile <strong>and</strong> every meal, forecast and AI recommendation "
                "linked to it. It reports the exact number of rows removed, and your session ends "
                "with a fresh onboarding form."
            ),
        )
        if not user_id:
            st.caption("Nothing to delete — this profile has no database row.")
        elif not st.session_state.get(SESSION_DELETE_ALL):
            if st.button("💥 Delete my profile and all data", key="nv_ask_delete_all"):
                st.session_state[SESSION_DELETE_ALL] = True
                st.rerun()
        else:
            typed = st.text_input(
                f"Type {PROFILE_DELETE_CONFIRM_WORD} to confirm",
                key="nv_confirm_delete_all_input",
                placeholder=PROFILE_DELETE_CONFIRM_WORD,
            )
            confirm_col, cancel_col = st.columns([1, 1], gap="small")
            if confirm_col.button("Delete everything", key="nv_do_delete_all", use_container_width=True):
                if typed.strip().upper() != PROFILE_DELETE_CONFIRM_WORD:
                    st.error(f"Type {PROFILE_DELETE_CONFIRM_WORD} exactly to confirm.")
                else:
                    removed, message = delete_profile_and_data(int(user_id))
                    if removed:
                        clear_profile()
                        reset_cached_resources()
                        flash(message, "success")
                    else:
                        st.session_state[SESSION_DELETE_ALL] = False
                        flash(message, "info")
                    st.rerun()
            if cancel_col.button("Cancel", key="nv_cancel_delete_all", use_container_width=True):
                st.session_state[SESSION_DELETE_ALL] = False
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    ui.notice(
        "Deleting rows in a free Neon database does not shrink the database file itself (PostgreSQL "
        "keeps the space for reuse). To reclaim it, use Neon's <strong>Vacuum</strong> control in the "
        "dashboard. Re-running <code>python database/init_db.py --sample-data</code> restores the "
        "demo data if you delete it by accident.",
        tone="info",
        icon="🧹",
    )


# --------------------------------------------------------------------------- #
# No profile / main
# --------------------------------------------------------------------------- #
def render_saved_note() -> None:
    """Show the outcome of the last save once, then forget it."""
    note = st.session_state.pop(SESSION_SAVED_NOTE, None)
    if not note:
        return
    ui.notice(f"{note['message']} {note['detail']}", tone="success", icon="✅")


def main() -> None:
    init_session_state()
    ui.load_css()
    render_pending_flash()

    profile: Optional[UserProfile] = get_profile()
    if profile is None:
        ui.hero(
            title="Profile & data",
            subtitle="Create your profile first — this page edits it and can delete it.",
            eyebrow="Profile · Step 0",
            icon="👤",
        )
        ui.notice(
            "Open the main **NutriVision AI** page from the sidebar and complete onboarding. "
            "Everything here (targets, storage summary, deletes) needs a profile to act on.",
            tone="info",
            icon="🧭",
        )
        ui.page_footer(CURRENT_PHASE, APP_BUILD, APP_FOOTER_NOTE)
        return

    ui.hero(
        title=f"Profile & data — {profile.name.split()[0]}",
        subtitle=(
            "Your stored profile, how every target is calculated, and full control over what is "
            "kept. Editing updates the existing row; deleting always asks first."
        ),
        eyebrow=f"Profile · user_id {profile.user_id}" if profile.user_id else "Profile · session only",
        icon="👤",
    )
    render_saved_note()

    if not require_database():
        st.caption("The database is unavailable, so only session data is shown and nothing can be deleted.")
    elif profile.user_id:
        require_stored_profile(profile)

    render_targets(profile)
    st.divider()
    render_edit_form(profile)
    st.divider()
    render_storage(profile.user_id)
    st.divider()
    render_danger_zone(profile)

    ui.page_footer(CURRENT_PHASE, APP_BUILD, APP_FOOTER_NOTE)


main()
