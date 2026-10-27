"""NutriVision AI — application entry point.

Phases delivered so far:
  * Phase 1 — dark-themed Streamlit shell, onboarding form with validation,
    session-state profile handling, and derived health values (BMI, BMR, TDEE,
    goal-adjusted calorie target, protein and fibre targets).
  * Phase 2 — Neon PostgreSQL persistence through SQLAlchemy: profile is saved
    to the ``users`` table, restored when a known name signs in, and the app
    degrades gracefully (setup panel) when no database is configured.

Run locally with:

    streamlit run app.py
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Sequence

import streamlit as st

from config import (
    ACTIVITY_LEVELS,
    AGE_RANGE,
    APP_BUILD,
    APP_FOOTER_NOTE,
    APP_ICON,
    APP_NAME,
    APP_TAGLINE,
    CURRENT_PHASE,
    DIET_PREFERENCES,
    FITNESS_GOALS,
    GENDERS,
    HEIGHT_RANGE_CM,
    SQLITE_URL,
    WEIGHT_RANGE_KG,
)
from database import (
    DatabaseNotConfiguredError,
    DatabaseStatus,
    check_database,
    find_user_by_name,
    get_user,
    list_users,
    reset_engine,
    save_user,
    table_counts,
    update_user,
)
from models import Meal, Prediction, Recommendation, User
from sqlalchemy import delete
from sqlalchemy.exc import SQLAlchemyError

from utils import ui
from utils.helpers import (
    SESSION_CONFIRM_RESET,
    UserProfile,
    build_profile,
    clear_profile,
    flash,
    form_epoch,
    format_number,
    get_profile,
    init_session_state,
    is_editing_profile,
    render_pending_flash,
    reset_cached_resources,
    save_profile,
    set_editing_profile,
    validate_profile_inputs,
)
from utils.logging_config import setup_logging

# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
setup_logging()
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title=f"{APP_NAME} · Setup",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Static content for this phase
# --------------------------------------------------------------------------- #
# Sample data used ONLY to preview the UI. It is clearly separated from real
# user records and can be removed later without touching the app logic.
DEMO_PROFILE: Dict[str, Any] = {
    "name": "Demo User",
    "age": 28,
    "gender": "Female",
    "height_cm": 163.0,
    "weight_kg": 64.5,
    "activity_level": "Moderately active (moderate exercise 3-5 days/week)",
    "diet_preference": "Vegetarian",
    "fitness_goal": "Weight Loss",
}

# Build roadmap shown on the home screen. Statuses are honest: a phase is marked
# Complete only after it ships verified, so Phase 8 is the last one still Planned.
ROADMAP: Sequence[tuple[str, str, str, str]] = (
    ("Phase 1", "Streamlit UI & user profile", "Complete",
     "Onboarding, validation, dark theme, reusable UI components."),
    ("Phase 2", "PostgreSQL database (Neon) + SQLAlchemy models", "Complete",
     "Users, Meals, Predictions and Recommendations tables + data-access layer."),
    ("Phase 3", "Food image recognition (pretrained model)", "Complete",
     "Upload Meal page: MobileNetV2/Food-101, validation, confidence + top-5."),
    ("Phase 4", "Nutrition engine (Indian food CSV + health score)", "Complete",
     "134 Indian foods, documented matching rules, portion scaling, transparent 0-100 score."),
    ("Phase 5", "Save meal + dashboard", "Complete",
     "Explicit Save Meal with duplicate guards, Plotly dashboard, filterable history."),
    ("Phase 6", "PNIE — XGBoost calorie-trend prediction", "Complete",
     "One XGBoost regressor, chronological evaluation, honest minimum-history fallbacks."),
    ("Phase 7", "Gemini 2.5 Flash nutrition coach", "Complete",
     "One snapshot per press, stored recommendations, every failure classified and explained."),
    ("Phase 8", "Final UI polish & deployment", "Complete",
     "Profile & data page, consistent headers/footers, deployment guide and checklist."),
)


# --------------------------------------------------------------------------- #
# Page catalogue (Phase 8): the home screen is a hub, not a duplicate dashboard
# --------------------------------------------------------------------------- #
PAGES: Sequence[tuple[str, str, str]] = (
    ("pages/1_Dashboard.py", "📊 Today's dashboard",
     "Calories, macros and health score for today, weekly trends, the saved forecast and the latest AI recommendation."),
    ("pages/2_Upload_Meal.py", "📷 Analyse a meal",
     "Upload a photo or type a dish: food recognition, portion size, nutrition lookup, health score and Save Meal."),
    ("pages/3_Meal_History.py", "🗂️ Meal history",
     "Every saved meal with filters, per-day rollups, charts, CSV export and single-row delete."),
    ("pages/4_Predictions.py", "📈 Calorie forecast",
     "PNIE predicts next week's average daily calories from your own log, with the typical error and the features behind it."),
    ("pages/5_AI_Coach.py", "🤖 AI coach",
     "One short Gemini-written message from your numbers. You can read the exact prompt before you send it."),
    ("pages/6_Profile.py", "👤 Profile & data",
     "Targets and how they are calculated, edit the profile, and delete meal history or your whole record."),
    ("pages/7_Model_Comparison.py", "⚖️ Model comparison",
     "Run the original Food-101 classifier, the separately trained enhanced classifier, or both side by side."),
)

# --------------------------------------------------------------------------- #
# Database access helpers (Phase 2)
# --------------------------------------------------------------------------- #
SESSION_DB_AVAILABLE: str = "nv_db_available"


def clear_database_flag() -> None:
    """Forget the cached availability result (call after a config change)."""
    st.session_state.pop(SESSION_DB_AVAILABLE, None)


def get_database_status() -> DatabaseStatus:
    """Run the connectivity + schema check once per session and remember it."""
    if SESSION_DB_AVAILABLE not in st.session_state:
        st.session_state[SESSION_DB_AVAILABLE] = check_database()
    return st.session_state[SESSION_DB_AVAILABLE]


def db_available() -> bool:
    """True when the database is reachable *and* the schema exists."""
    return bool(get_database_status().ok)


def _reset_database_connection() -> None:
    """Dispose the engine and forget every cached database result."""
    reset_engine()
    reset_cached_resources()
    clear_database_flag()


def activate_sqlite_fallback() -> None:
    """Point the app at the local SQLite file for this process only.

    The original configuration is remembered so it can be restored exactly.
    """
    if "nv_original_database_url" not in os.environ:
        os.environ["nv_original_database_url"] = os.environ.get("DATABASE_URL", "")
    os.environ["DATABASE_URL"] = SQLITE_URL
    _reset_database_connection()
    logger.warning("SQLite development fallback activated for this process")


def deactivate_sqlite_fallback() -> None:
    """Restore the PostgreSQL configuration that was active before the override."""
    original = os.environ.pop("nv_original_database_url", "")
    if original:
        os.environ["DATABASE_URL"] = original
    else:
        os.environ.pop("DATABASE_URL", None)
    _reset_database_connection()
    logger.info("SQLite fallback deactivated; PostgreSQL configuration restored")


def render_db_status_panel(compact: bool = True) -> None:
    """Small sidebar block that always tells the truth about the backend."""
    status = get_database_status()
    tone = "good" if status.ok else ("warn" if status.backend != "not configured" else "neutral")
    label = status.label if status.backend != "not configured" else "Not configured"
    detail = status.location if status.backend != "not configured" else "no DATABASE_URL"
    note = (
        f"{len(status.tables)}/{len(status.missing_tables) + len(status.tables)} tables ready"
        if status.ok
        else ("Tables missing — run init_db.py" if status.backend != "not configured" else "See Database Setup below")
    )
    ui.card(
        body_html=ui.rows_html([("Backend", label), ("Target", detail), ("Schema", note)]),
        title=f"Database {ui.pill('online' if status.ok else 'offline', tone)}",
        icon="🗄️",
    )
    if compact and not status.ok and status.error:
        st.caption(status.error[:220])


def render_database_setup(status) -> None:
    """Read-only setup instructions shown when the database is unusable."""
    ui.section("Phase 2 setup needed", "The app works, but profiles and meals cannot be stored yet.", icon="🗄️")

    if status.backend == "not configured":
        ui.notice(
            "No <code>DATABASE_URL</code> is configured. Create a free Neon project, then set the "
            "connection string in your local <code>.env</code> file (or Streamlit secrets) and reload.",
            tone="warning",
            icon="🧭",
        )
    else:
        ui.notice(
            f"Connected to <strong>{status.label}</strong> but the schema is incomplete. "
            f"Run the initialisation command below, then reload the app.",
            tone="warning",
            icon="🧭",
        )
        if status.error:
            st.code(status.error, language="text")

    left, right = st.columns([1.2, 1], gap="large")

    with left:
        ui.card(
            title="1 · Configure the database",
            icon="🔑",
            body_html=ui.html_raw(
                """
                <ol class="nv-steps">
                  <li>Create a free project at <code>neon.tech</code> and copy its connection string.</li>
                  <li>Copy <code>.env.example</code> to <code>.env</code> and paste it as
                      <code>DATABASE_URL=postgresql+psycopg2://user:password@host/db?sslmode=require</code>.</li>
                  <li>Reload this page — the sidebar panel should turn green.</li>
                </ol>
                """
            ),
        )

        with st.expander("Optional: use the local SQLite fallback instead"):
            st.markdown(
                "Choose this **deliberately** if you do not have a Neon account yet. "
                "It creates a local file (`nutrivision.db`) and is meant for development only — "
                "Streamlit Community Cloud has an ephemeral filesystem, so PostgreSQL is required for deployment."
            )
            sqlite_active = get_database_status().backend == "sqlite"
            if sqlite_active:
                if st.button("↩️ Switch back to PostgreSQL", key="nv_back_to_postgres"):
                    deactivate_sqlite_fallback()
                    flash("Switched back to the PostgreSQL configuration.", "info")
                    st.rerun()
            else:
                if st.button("Use SQLite for this session", key="nv_use_sqlite"):
                    activate_sqlite_fallback()
                    flash(
                        "SQLite fallback enabled for this session only. Run "
                        "`python database/init_db.py --sqlite` to create its tables.",
                        "info",
                    )
                    st.rerun()
            st.caption(f"Fallback file: `{SQLITE_URL.removeprefix('sqlite:///')}`")

    with right:
        ui.card(
            title="2 · Create the tables",
            icon="🛠️",
            body_html=ui.html_raw(
                """
                <div class="nv-code">python database/init_db.py --check<br>
                python database/init_db.py<br>
                python database/init_db.py --sample-data   # optional demo rows</div>
                <p class="nv-muted">Uses <code>SQLAlchemy Base.metadata.create_all()</code>:
                it only creates missing tables and never alters existing ones.</p>
                """
            ),
        )
        status_after = check_database()
        if status_after.ok:
            ui.notice("Schema is complete. Reload the page to continue.", tone="success", icon="✅")


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar(profile: Optional[UserProfile]) -> None:
    """Branding, database status, profile summary and profile actions."""
    with st.sidebar:
        ui.sidebar_brand(APP_NAME, APP_TAGLINE, APP_ICON, CURRENT_PHASE)
        render_db_status_panel()
        st.divider()

        if profile is None:
            st.caption("Finish onboarding to see your nutrition snapshot.")

            if db_available():
                # Phase 2 has no authentication yet: a known name is matched to its
                # saved user row. This makes persistence visible without pretending
                # to be a login system (documented in README).
                recent = [u.name for u in (list_users(limit=5) or [])]
                if recent:
                    st.markdown("**Continue as**")
                    choice = st.selectbox("Saved profiles", recent, label_visibility="collapsed", key="nv_returning_user")
                    if st.button(f"↩️ Load {choice}'s data", use_container_width=True):
                        stored = find_user_by_name(choice)
                        if stored is None:
                            st.error("That profile could not be loaded (row disappeared).")
                        else:
                            save_profile(stored.to_profile())
                            flash(f"Loaded saved profile for {stored.name} (user_id={stored.user_id}).", "success")
                            st.rerun()
                    st.divider()

            st.markdown("**Quick preview**")
            st.caption("Loads sample values so you can see the app layout before typing anything.")
            if st.button("🧪 Load demo profile", use_container_width=True):
                try:
                    demo = build_profile(DEMO_PROFILE)
                    save_profile(demo)
                    flash("Demo profile loaded (sample data, not a real user record).", "info")
                    st.rerun()
                except ValueError as exc:  # should never happen with static data
                    logger.error("Demo profile failed validation: %s", exc)
                    st.error(f"Demo profile is invalid: {exc}")
        else:
            st.markdown(
                f"**{profile.name}** &nbsp; {ui.pill(profile.fitness_goal, 'accent')}",
                unsafe_allow_html=True,
            )
            ui.stat_rows(
                [
                    ("BMI", f"{profile.bmi:g} ({profile.bmi_category})"),
                    ("Daily target", f"{format_number(profile.daily_calorie_target)} kcal"),
                ]
            )

            if st.button("✏️ Edit profile", use_container_width=True):
                set_editing_profile(True)
                st.rerun()

            if st.button("🗑️ Reset profile", use_container_width=True):
                st.session_state[SESSION_CONFIRM_RESET] = True
                st.rerun()

            if st.session_state.get(SESSION_CONFIRM_RESET):
                st.warning("This removes the profile from this session.")
                confirm_col, cancel_col = st.columns(2)
                if confirm_col.button("Yes, reset", key="nv_reset_yes", use_container_width=True):
                    clear_profile()
                    flash("Profile cleared. Complete onboarding again to continue.", "info")
                    st.rerun()
                if cancel_col.button("Cancel", key="nv_reset_no", use_container_width=True):
                    st.session_state[SESSION_CONFIRM_RESET] = False
                    st.rerun()

        st.divider()
        if profile is not None and getattr(profile, "user_id", None):
            st.caption(
                f"**Storage:** this profile is stored in the database as "
                f"<code>user_id {profile.user_id}</code>. Meals, forecasts and AI recommendations "
                "are written only when you press their button — see **Profile & data** to review "
                "or delete them.",
                unsafe_allow_html=True,
            )
        else:
            st.caption(
                "**Storage:** this profile lives in the Streamlit session only (it disappears when "
                "the tab is closed). Open the home screen to write it to the database."
            )
        st.caption(f"{APP_NAME} · build {APP_BUILD}")


# --------------------------------------------------------------------------- #
# Onboarding / profile form
# --------------------------------------------------------------------------- #
def _index_of(options: Sequence[str], value: Optional[str]) -> int:
    """Safe index lookup for select boxes (0 when the value is unknown)."""
    if value is None:
        return 0
    try:
        return list(options).index(value)
    except ValueError:
        return 0


def render_profile_form(mode: str, prefill: Optional[UserProfile] = None) -> None:
    """Render the create/update profile form.

    Args:
        mode: ``"create"`` for onboarding, ``"edit"`` when updating an
            existing profile (widget keys are prefixed so the two states never
            leak values into each other).
        prefill: existing profile used to pre-populate the inputs.
    """
    editing = mode == "edit"
    prefix = f"{mode}{form_epoch()}"  # bump on reset -> blank form

    ui.hero(
        title="Create your nutrition profile" if not editing else "Update your profile",
        subtitle=(
            "Ten seconds of setup lets NutriVision personalise calorie targets, meal "
            "analysis and AI recommendations."
        ),
        eyebrow="Step 1 · Onboarding" if not editing else "Profile settings",
        icon="🥗",
    )

    with st.form("nv_profile_form", clear_on_submit=False):
        left, right = st.columns(2, gap="large")

        with left:
            st.markdown("##### About you")
            name = st.text_input(
                "Full name *",
                value=prefill.name if prefill else "",
                max_chars=60,
                placeholder="e.g. Aarav Sharma",
                key=f"{prefix}_name",
                help="Used to personalise the dashboard and AI recommendations.",
            )
            age = st.number_input(
                "Age (years) *",
                min_value=int(AGE_RANGE[0]),
                max_value=int(AGE_RANGE[1]),
                value=int(prefill.age) if prefill else 28,
                step=1,
                key=f"{prefix}_age",
            )
            gender = st.selectbox(
                "Gender *",
                GENDERS,
                index=_index_of(GENDERS, prefill.gender if prefill else None),
                key=f"{prefix}_gender",
                help="Used only for energy calculations (Mifflin-St Jeor equation).",
            )
            height_cm = st.number_input(
                "Height (cm) *",
                min_value=float(HEIGHT_RANGE_CM[0]),
                max_value=float(HEIGHT_RANGE_CM[1]),
                value=float(prefill.height_cm) if prefill else 168.0,
                step=1.0,
                format="%.1f",
                key=f"{prefix}_height",
            )

        with right:
            st.markdown("##### Lifestyle & goal")
            weight_kg = st.number_input(
                "Weight (kg) *",
                min_value=float(WEIGHT_RANGE_KG[0]),
                max_value=float(WEIGHT_RANGE_KG[1]),
                value=float(prefill.weight_kg) if prefill else 68.0,
                step=0.5,
                format="%.1f",
                key=f"{prefix}_weight",
            )
            activity_level = st.selectbox(
                "Activity level *",
                list(ACTIVITY_LEVELS.keys()),
                index=_index_of(list(ACTIVITY_LEVELS.keys()), prefill.activity_level if prefill else None),
                key=f"{prefix}_activity",
                help=ACTIVITY_LEVELS.get(
                    prefill.activity_level if prefill else "",
                    {},
                ).get("description", "Pick the option that best matches a typical week."),
            )
            diet_preference = st.selectbox(
                "Diet preference *",
                DIET_PREFERENCES,
                index=_index_of(DIET_PREFERENCES, prefill.diet_preference if prefill else None),
                key=f"{prefix}_diet",
            )
            fitness_goal = st.radio(
                "Fitness goal *",
                FITNESS_GOALS,
                index=_index_of(FITNESS_GOALS, prefill.fitness_goal if prefill else None),
                horizontal=True,
                key=f"{prefix}_goal",
            )

        st.caption("* Required. All fields are validated before your profile is created.")

        submitted = st.form_submit_button(
            "Create profile" if not editing else "Save changes",
            type="primary",
            use_container_width=True,
        )

    if submitted:
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
            logger.warning("Profile validation failed with %d error(s)", len(errors))
            ui.error_list(errors.values())
            return

        new_profile = build_profile(raw)
        if editing and prefill is not None:
            new_profile.created_at = prefill.created_at  # keep the original creation time

        # Persist through SQLAlchemy when the database is configured and ready.
        # A database problem must never lose the profile: the object is always
        # kept in session state and the user is told exactly what happened.
        persisted = False
        if db_available():
            try:
                if editing and prefill is not None and prefill.user_id is not None:
                    row = update_user(prefill.user_id, **raw)
                    persisted = row is not None
                    new_profile.user_id = prefill.user_id if persisted else None
                else:
                    existing = find_user_by_name(new_profile.name)
                    if existing is not None:
                        row = update_user(existing.user_id, **raw)
                        persisted = row is not None
                        new_profile.user_id = existing.user_id if persisted else None
                    else:
                        row = save_user(User.from_profile(new_profile))
                        if row is not None:
                            new_profile.user_id = row.user_id
                            persisted = True
            except (SQLAlchemyError, DatabaseNotConfiguredError) as exc:
                logger.error("Profile persistence failed: %s", exc)
                st.error(
                    "Your profile was kept in this session, but it could not be written to the "
                    f"database: {exc.__class__.__name__}: {exc}"
                )

        save_profile(new_profile)
        flash(
            (
                f"Profile saved for {new_profile.name} (user_id={new_profile.user_id}) — "
                f"daily target {format_number(new_profile.daily_calorie_target)} kcal."
                if persisted
                else f"Profile kept in this session for {new_profile.name}. "
                     f"Daily target {format_number(new_profile.daily_calorie_target)} kcal."
            ),
            "success",
        )
        st.rerun()

    if editing:
        if st.button("← Cancel and go back", use_container_width=False):
            set_editing_profile(False)
            st.rerun()

    ui.notice(
        "Your profile data is used to estimate energy and nutrient targets. "
        "These are general wellness estimates, not a medical assessment.",
        tone="info",
        icon="🩺",
    )

    ui.page_footer(CURRENT_PHASE, APP_BUILD, APP_FOOTER_NOTE)


# --------------------------------------------------------------------------- #
# Home screen
# --------------------------------------------------------------------------- #
def render_database_panel(profile: UserProfile) -> None:
    """Phase 2 panel: persistence state, counts and the explicit save action.

    Meals (Phase 5) and PNIE forecasts (Phase 6) follow the exact same pattern:
    nothing is written until the user presses a button, and the button cannot
    double-save because the row is persisted before the rerun.
    """
    status = get_database_status()

    ui.section(
        "Database (Phase 2)",
        "Neon PostgreSQL via SQLAlchemy — the profile is written only when you press the button.",
        icon="🗄️",
    )

    if not status.ok:
        render_database_setup(status)
        return

    counts = table_counts() or {}
    stored = get_user(profile.user_id) if profile.user_id else None

    left, right = st.columns([1, 1.25], gap="large")

    with left:
        ui.metric_card(
            "Saved profile",
            f"user_id {profile.user_id}" if profile.user_id else "not stored",
            "",
            (
                f"Last write: {stored.created_at.strftime('%d %b %Y · %H:%M')} UTC"
                if stored is not None and stored.created_at
                else "Session-only profile"
            ),
            tone="good" if profile.user_id else "warn",
        )
        ui.card(
            title="Row counts",
            icon="🔢",
            body_html=ui.rows_html(
                [(table, format_number(value)) for table, value in counts.items()]
            ),
        )
        st.caption(f"Backend: {status.label} · {status.location}")

    with right:
        if profile.user_id and stored is not None:
            ui.notice(
                f"This profile is stored in the <code>users</code> table as "
                f"<strong>user_id {profile.user_id}</strong>. Meals, predictions and "
                "recommendations will reference it by foreign key.",
                tone="success",
                icon="✅",
            )
            if st.button("🗑️ Remove saved copy from the database", key="nv_delete_saved_profile"):
                from database import session_scope

                try:
                    # Explicit child-first deletion so the behaviour is identical
                    # on PostgreSQL (ON DELETE CASCADE) and on SQLite.
                    with session_scope() as session:
                        for table in (Meal, Prediction, Recommendation):
                            session.execute(delete(table).where(table.user_id == profile.user_id))
                        session.execute(delete(User).where(User.user_id == profile.user_id))
                    updated = UserProfile.from_dict(profile.to_dict())
                    updated.user_id = None
                    save_profile(updated)
                    flash("The stored copy was deleted. Your session profile is still active.", "info")
                    st.rerun()
                except SQLAlchemyError as exc:
                    st.error(f"Delete failed: {exc.__class__.__name__}: {exc}")
        else:
            ui.notice(
                "The profile exists in this session but has no database row yet. Saving it now "
                "creates the user record that meals, predictions and recommendations will point to.",
                tone="warning",
                icon="⬆️",
            )
            if st.button("💾 Save profile to database", type="primary", key="nv_persist_profile"):
                try:
                    row = save_user(User.from_profile(profile))
                    if row is None:
                        st.error("Nothing was written — the database reported a schema problem.")
                    else:
                        updated = UserProfile.from_dict(profile.to_dict())
                        updated.user_id = row.user_id
                        save_profile(updated)
                        flash(f"Profile stored in the database as user_id={row.user_id}.", "success")
                        st.rerun()
                except SQLAlchemyError as exc:
                    logger.exception("Manual profile save failed")
                    st.error(f"Could not save the profile: {exc.__class__.__name__}: {exc}")

        st.markdown("##### Tables created by Phase 2")
        st.markdown(
            "`users` · `meals` · `predictions` · `recommendations` — with primary keys, "
            "foreign keys to `users.user_id` (`ON DELETE CASCADE`) and indexes on "
            "`user_id`, `timestamp` and `prediction_date`."
        )


def render_home(profile: UserProfile) -> None:
    """Profile-aware landing screen with real derived health values."""
    ui.hero(
        title=f"Welcome, {profile.name.split()[0]} 👋",
        subtitle=(
            "Your profile is ready. This is the hub: your baseline numbers below, and every page "
            "of the pipeline one click away."
        ),
        eyebrow=f"Home · {CURRENT_PHASE}",
    )

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
            "TDEE from activity level",
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

    ui.notice(
        f"Signed in as <strong>{profile.name}</strong> — daily target "
        f"<strong>{format_number(profile.daily_calorie_target)} kcal</strong>, protein "
        f"<strong>{format_number(profile.protein_target_g)} g</strong>, fibre "
        f"<strong>{format_number(profile.fiber_target_g)} g</strong>. Everything is recalculated "
        "the moment you edit your profile on the <strong>Profile &amp; data</strong> page. "
        "No authentication yet: whoever opens this app sees the profile in the session.",
        tone="info",
        icon="🧭",
    )

    ui.section("Your pages", "Seven screens, one pipeline — open the one you need.", icon="🧩")
    columns = st.columns(3, gap="medium")
    for index, (path, label, description) in enumerate(PAGES):
        with columns[index % 3]:
            st.page_link(path, label=label, use_container_width=True)
            st.caption(description)

    render_database_panel(profile)

    ui.section("Build roadmap", "Where the project stands right now.", icon="🗺️")
    ui.roadmap(ROADMAP)

    ui.notice(
        "Calorie and nutrient targets are estimates produced from standard equations and the "
        "profile you entered. They are general wellness information, not medical advice.",
        tone="warning",
        icon="⚠️",
    )
    ui.notice(
        "Your profile, meals and PNIE forecasts are stored in the database, and every write is an "
        "explicit action: meals are saved only with the <strong>Save Meal</strong> button, forecasts "
        "only with the <strong>Save this forecast</strong> button, and AI text only when you press "
        "the coach's button. <strong>Profile &amp; data</strong> shows what is stored and can delete it.",
        tone="info",
        icon="🔐",
    )
    ui.page_footer(CURRENT_PHASE, APP_BUILD, APP_FOOTER_NOTE)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    init_session_state()
    profile = get_profile()

    render_sidebar(profile)
    render_pending_flash()

    editing = is_editing_profile()
    if profile is None or editing:
        render_profile_form(mode="edit" if editing and profile else "create", prefill=profile)
    else:
        render_home(profile)


if __name__ == "__main__":
    main()
