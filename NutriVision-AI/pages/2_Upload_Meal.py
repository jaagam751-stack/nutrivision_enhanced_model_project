"""Upload Meal — food recognition (Phase 3) + nutrition estimate (Phase 4).

Flow on this page:

**Step 1 — recognition (Phase 3)**
1. the user uploads a JPG/JPEG/PNG photo (validated: type, size, decodability),
2. presses **Analyze Meal** (nothing runs automatically),
3. the pretrained MobileNetV2 / Food-101 model predicts the food and a confidence
   score, shown together with the top-5 alternatives,
4. the result is cached in session state so Streamlit reruns do not re-run inference.

**Step 2 — nutrition (Phase 4)**
5. the recognised label is matched to the Indian food CSV using the documented
   matching rules; uncertain matches are flagged and the user confirms the food,
6. the portion size is chosen (presets, slider or exact grams),
7. nutrition for that portion is calculated by linear scaling from per-100 g
   values, alongside the daily-target context from the user's profile,
8. a transparent rule-based Nutrition Health Score (0-100) is shown with its full
   point-by-point breakdown.

Nothing is written to the database on this page until the explicit **Save Meal**
button is pressed (its own section at the bottom, with duplicate guards). If the
recognition model is unavailable the page shows a real error with repair
instructions and never invents a prediction.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional

import streamlit as st

from config import (
    ALLOWED_IMAGE_TYPES,
    APP_BUILD,
    APP_FOOTER_NOTE,
    APP_ICON,
    APP_NAME,
    CURRENT_PHASE,
    DUPLICATE_GUARD_SECONDS,
    MAX_UPLOAD_MB,
    MODEL_IMAGE_SIZE,
    DEFAULT_PORTION_GRAMS,
    PORTION_GRAMS_RANGE,
)
import pandas as pd

from food_recognition import FoodPrediction, FoodPredictor, ModelUnavailableError, UnsupportedImageError
from food_recognition.model_loader import load_food_model, model_status
from food_recognition.preprocessing import ImageInfo, load_image_from_upload, validate_upload
from nutrition import (
    NUTRIENT_LABELS,
    NUTRIENT_UNITS,
    NutritionDataError,
    get_nutrition_engine,
    score_item,
)
from utils import ui
from utils.data_access import db_ready, db_status, store_meal
from utils.helpers import (
    UserProfile,
    flash,
    format_number,
    get_profile,
    init_session_state,
    render_pending_flash,
)
from utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title=f"{APP_NAME} · Upload Meal",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

SESSION_ANALYSIS: str = "nv_food_analysis"      # cached prediction for the current upload
SESSION_UPLOAD_KEY: str = "nv_upload_key"
SESSION_MATCHED_FOOD: str = "nv_matched_food_id"   # confirmed CSV row (food_id)
SESSION_PORTION: str = "nv_portion_grams"          # portion size in grams
SESSION_SAVED_MEAL_ID: str = "nv_saved_meal_id"     # locks the Save Meal button once written
SESSION_SAVE_MESSAGE: str = "nv_save_message"       # last save/duplicate message

TIER_TONE = {"high": "good", "medium": "warn", "low": "bad"}
TIER_LABEL = {"high": "High confidence", "medium": "Medium confidence", "low": "Low confidence"}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def upload_fingerprint(uploaded_file) -> str:
    """Cheap, stable identity for an upload (name + size + content hash)."""
    digest = hashlib.md5()                       # noqa: S324 - not used for security
    digest.update((uploaded_file.name or "").encode("utf-8"))
    digest.update(uploaded_file.getvalue())
    return digest.hexdigest()


def confidence_bar(label: str, confidence: float, tone: str = "accent") -> str:
    """One row of the top-5 ranking, as HTML (no charting dependency needed)."""
    width = max(1.0, min(100.0, float(confidence)))
    return (
        f'<div class="nv-rank-row">'
        f'<span class="nv-rank-label">{label}</span>'
        f'<span class="nv-rank-bar"><span class="nv-rank-fill nv-fill-{tone}" style="width:{width:.1f}%"></span></span>'
        f'<span class="nv-rank-value">{confidence:.1f}%</span>'
        f"</div>"
    )


def render_model_status() -> None:
    """Show what the app would load, before any upload happens."""
    with st.expander("🧠 Model status", expanded=False):
        status = model_status()
        st.markdown(
            "\n".join(f"- **{key}:** `{value}`" for key, value in status.items())
        )
        st.caption(
            "The model is loaded lazily on the first analysis and then cached for the whole "
            "server session, so only the very first prediction is slow."
        )


def render_model_error(exc: Exception) -> None:
    """Explain a model-loading failure with concrete repair steps."""
    ui.notice(
        "The food recognition model could not be loaded, so no prediction was produced. "
        "No substitute or placeholder result is shown.",
        tone="danger",
        icon="🚫",
    )
    st.error(f"{exc.__class__.__name__}: {exc}")
    st.markdown("**How to fix it**")
    st.markdown(
        "1. **PyTorch backend** (downloads the checkpoint on first run, ~19 MB):\n"
        "   ```bash\n"
        "   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu\n"
        "   ```\n"
        "2. **Light ONNX backend** (no PyTorch at runtime):\n"
        "   ```bash\n"
        "   pip install onnxruntime\n"
        "   python -m food_recognition.export_onnx --out assets/models\n"
        "   ```\n"
        "3. Offline / restricted network: download `pytorch_model.bin` from "
        "`huggingface.co/AlexKoff88/mobilenet_v2_food101` and point `FOOD_MODEL_PATH` at it, "
        "or place an ONNX file in `assets/models/`."
    )
    render_model_status()


def render_analysis_result(prediction, info: ImageInfo) -> None:
    """Show the prediction: label, confidence, alternatives and caveats."""
    tone = TIER_TONE.get(prediction.tier, "default")

    ui.section("Recognition result", "What the pretrained model saw in your photo.", icon="🔍")

    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        ui.metric_card(
            "Predicted food",
            prediction.label,
            "",
            f"{ui.pill(TIER_LABEL.get(prediction.tier, prediction.tier), tone)}",
            tone=tone,
        )
    with col2:
        ui.metric_card(
            "Confidence",
            prediction.confidence_text,
            "",
            f"Runner-up is {prediction.margin:.1f} points behind",
            tone=tone,
        )
    with col3:
        ui.metric_card(
            "Inference time",
            format_number(prediction.inference_ms, 1),
            "ms",
            f"Backend: {prediction.backend} · CPU",
        )

    left, right = st.columns([1.1, 1], gap="large")

    with left:
        ui.section("Top 5 possibilities", "The model's full shortlist, not just the winner.", icon="📊")
        rows = "".join(
            confidence_bar(label, score, tone if index == 0 else "muted")
            for index, (label, score) in enumerate(prediction.alternatives)
        )
        ui.card(body_html=f'<div class="nv-rank">{rows}</div>')

        if prediction.is_uncertain:
            if prediction.tier == "low":
                reason = (
                    f"the best score is only {prediction.confidence_text}"
                )
            else:
                reason = (
                    f"the top two foods are only {prediction.margin:.1f} points apart "
                    f"({prediction.alternatives[0][0]} vs {prediction.alternatives[1][0]})"
                )
            ui.notice(
                f"<strong>Please confirm this food yourself.</strong> {prediction.explanation()} "
                f"Confirmation is needed because {reason}. Phase 4 will let you pick the correct "
                "dish from this list before nutrition is looked up.",
                tone="warning",
                icon="⚠️",
            )
        else:
            ui.notice(prediction.explanation(), tone="success", icon="✅")

    with right:
        ui.section("Image & model", "Provenance of this result.", icon="🖼️")
        ui.card(
            body_html=ui.rows_html(
                [
                    ("Image", info.describe()),
                    ("Model", prediction.model_name),
                    ("Backend", prediction.backend),
                    ("Classes", "101 Food-101 categories"),
                    ("Input size", f"{MODEL_IMAGE_SIZE} × {MODEL_IMAGE_SIZE} px"),
                    ("Confidence tier", TIER_LABEL.get(prediction.tier, prediction.tier)),
                ]
            )
        )
        ui.notice(
            "Food-101 covers 101 dishes (pizza, samosa, chicken curry, sushi…). Indian "
            "home-cooked staples such as dal, chapati, idli or paneer are <em>not</em> classes in "
            "this model, so a photo of them will land on the closest Western dish. That is a real "
            "limitation of the pretrained model, not a bug — and it is why you are asked to "
            "confirm uncertain labels.",
            tone="info",
            icon="🧠",
        )




# --------------------------------------------------------------------------- #
# Phase 4 — nutrition estimate
# --------------------------------------------------------------------------- #
def _engine():
    """Load the nutrition dataset once (cached by the engine factory)."""
    return get_nutrition_engine()


def render_food_match(prediction: FoodPrediction, profile: UserProfile):
    """Match the recognised label to the dataset and let the user confirm it.

    Returns the chosen :class:`FoodItem` (or ``None`` when nothing is chosen).

    Confirmation rules (documented in ``nutrition_engine.match``): exact, alias
    and curated model-label matches are accepted automatically; token and fuzzy
    matches, low recognition confidence, and unknown labels all require an
    explicit choice by the user before nutrition values are shown.
    """
    engine = _engine()
    result = engine.match(prediction.raw_label, top_k=6)
    must_confirm = result.needs_confirmation or prediction.is_uncertain

    ui.section(
        "Step 2 · Nutrition estimate",
        "Values come from a local Indian food composition table (per 100 g) — never from the AI model.",
        icon="🥗",
    )

    if must_confirm:
        reason = (
            "the photo recognition was uncertain"
            if prediction.is_uncertain
            else "the food name could not be matched to the dataset with confidence"
        )
        ui.notice(
            f"<strong>Please confirm the food before looking up nutrition.</strong> "
            f"NutriVision asks because {reason}. The suggested row is pre-selected below — "
            "change it if it is wrong.",
            tone="warning",
            icon="⚠️",
        )
    else:
        ui.notice(
            f"Matched automatically — {result.message or result.method_label.lower()}.",
            tone="success",
            icon="✅",
        )

    left, right = st.columns([1.05, 1], gap="large")

    # -- shortlist from the recognition step --------------------------------- #
    with left:
        candidates: list[tuple[int, str]] = []
        if result.food is not None:
            candidates.append(
                (result.food.food_id, f"{result.food.name} — {result.method_label.lower()} ({result.score:.0f}% match)")
            )
        for food, score in result.alternatives:
            if all(food.food_id != existing for existing, _label in candidates):
                label = f"{food.name} — similar dish ({score:.0f}% text similarity)" if score else food.name
                candidates.append((food.food_id, label))

        options = [label for _id, label in candidates]

        if result.food is None:
            # Nothing matched: never pre-select a random food for an unknown dish.
            # The user must choose explicitly before any values are shown.
            ui.notice(
                f"<strong>'{prediction.label}' is not in the Indian food dataset.</strong> "
                "NutriVision will not invent nutrition values for an unrecognised dish. "
                "Choose the closest food below (or search the full dataset) — nothing is "
                "estimated until you pick one.",
                tone="danger",
                icon="🚫",
            )

        selected_label = st.radio(
            "Suggested matches from the dataset",
            options,
            index=0 if result.food is not None else None,
            key=f"nv_match_choice_{prediction.raw_label}",
            help="These are lookups in the CSV, ranked by how closely the names match.",
        ) if options else None

        chosen_id = candidates[options.index(selected_label)][0] if selected_label else None

        with st.expander("🔎 Search the full dataset instead"):
            query = st.text_input(
                "Search foods", placeholder="e.g. dal, chapati, paneer, idli…", key="nv_food_search"
            )
            category_names = ("All categories",) + engine.categories
            category = st.selectbox("Category", category_names, key="nv_food_category")
            results = engine.search(query, limit=40) if query else engine.list_foods(
                None if category == "All categories" else category
            )
            if category != "All categories":
                results = [item for item in results if item.category == category]
            if not results:
                st.caption("No food matched that search. Try a simpler word like 'rice' or 'dal'.")
            else:
                labels = {f"{item.name} · {item.category} · {item.calories:.0f} kcal/100 g": item.food_id for item in results}
                picked = st.selectbox("Results", list(labels), key="nv_food_search_result")
                if st.button("Use this food", key="nv_use_search_result"):
                    st.session_state[SESSION_MATCHED_FOOD] = labels[picked]
                    st.rerun()

    stored_id = st.session_state.get(SESSION_MATCHED_FOOD)
    food = engine.get(stored_id) if stored_id else None
    if food is None and chosen_id is not None:
        food = engine.get(chosen_id)

    with right:
        if food is None:
            ui.notice(
                "No food is selected yet, so no nutrition values are shown. Pick a food from the "
                "list on the left — NutriVision will not invent values for an unmatched dish.",
                tone="danger",
                icon="🚫",
            )
            return None
        ui.card(
            title="Dataset row in use",
            icon="📋",
            body_html=ui.rows_html(
                [
                    ("Food", food.name),
                    ("Category", food.category),
                    ("Match method", result.method_label),
                    ("Match score", f"{result.score:.0f}%"),
                    ("Typical serving", f"{food.serving_label} ({food.serving_grams:g} g)"),
                    ("Data source", "Indian food composition table values, rounded"),
                ]
            ),
        )
        if food.note:
            st.caption(f"Note on this row: {food.note}")

        compatible, explanation = food.matches_diet(profile.diet_preference)
        if not compatible:
            ui.notice(
                f"<strong>Diet check:</strong> {explanation} If the match is wrong, choose a different "
                "food above.",
                tone="warning",
                icon="🥘",
            )

    st.session_state[SESSION_MATCHED_FOOD] = food.food_id
    return food


def render_portion_controls(food, profile: UserProfile) -> float:
    """Portion selection: presets, slider and exact gram entry."""
    low, high = PORTION_GRAMS_RANGE
    engine = _engine()
    default = (
        float(food.serving_grams)
        if low <= food.serving_grams <= high
        else DEFAULT_PORTION_GRAMS
    )
    current = float(st.session_state.get(SESSION_PORTION, default))
    if not (low <= current <= high):
        current = default

    ui.section("Portion size", "Nutrition scales linearly with the weight you enter.", icon="⚖️")

    presets = engine.portion_presets(food)
    if presets:
        columns = st.columns(len(presets))
        for column, (label, grams) in zip(columns, presets):
            if column.button(label, key=f"nv_preset_{food.food_id}_{grams}", use_container_width=True):
                st.session_state[SESSION_PORTION] = grams
                st.rerun()

    left, right = st.columns([1, 1], gap="large")
    with left:
        grams = st.number_input(
            "Portion weight (g)",
            min_value=float(low),
            max_value=float(high),
            value=float(current),
            step=5.0,
            key=f"nv_portion_input_{food.food_id}",
            help=f"Accepted range: {low:g}–{high:g} g. Typical serving for this food: "
            f"{food.serving_label} ({food.serving_grams:g} g).",
        )
    with right:
        slider = st.slider(
            "Adjust",
            min_value=float(low),
            max_value=float(min(high, 600.0)),
            value=float(min(max(current, low), min(high, 600.0))),
            step=5.0,
            key=f"nv_portion_slider_{food.food_id}",
            label_visibility="collapsed",
        )

    chosen = float(grams if abs(grams - current) >= abs(slider - current) else slider)
    st.session_state[SESSION_PORTION] = chosen

    if st.button("↩️ Reset to the typical serving", key=f"nv_reset_portion_{food.food_id}"):
        st.session_state[SESSION_PORTION] = default
        st.rerun()

    return chosen


def render_nutrition_results(breakdown, food, profile: UserProfile) -> None:
    """Results table: per-portion values, per-100 g reference and daily context."""
    ui.section(
        "Estimated nutrition",
        f"For {breakdown.portion_label()} of {food.name}.",
        icon="📊",
    )

    daily_rows = breakdown.daily_context(profile)
    daily_lookup = {label.lower(): (value, target, percent) for label, value, target, percent in daily_rows}

    table_rows = []
    for key in ("calories", "protein", "fat", "carbs", "fiber", "sugar"):
        portion_value = breakdown.values.get(key)
        per_100g = breakdown.per_100g.get(key)
        if portion_value is None and per_100g is None:
            continue
        unit = NUTRIENT_UNITS[key]
        context = ""
        if key == "calories" and "energy" in daily_lookup:
            _value, _target, percent = daily_lookup["energy"]
            context = f"{percent:.0f}% of your daily target"
        elif key == "protein" and "protein" in daily_lookup:
            _value, _target, percent = daily_lookup["protein"]
            context = f"{percent:.0f}% of your protein target"
        elif key == "fiber" and "fibre" in daily_lookup:
            _value, _target, percent = daily_lookup["fibre"]
            context = f"{percent:.0f}% of your fibre target"
        elif key == "fat":
            context = f"{breakdown.macro_energy_pct.get('fat', 0):.0f}% of this portion's energy"
        elif key == "carbs":
            context = f"{breakdown.macro_energy_pct.get('carbs', 0):.0f}% of this portion's energy"

        table_rows.append(
            {
                "Nutrient": f"{NUTRIENT_LABELS[key]}",
                f"In your portion ({breakdown.portion_grams:g} g)": (
                    "—" if portion_value is None else f"{portion_value:,.1f} {unit}"
                ),
                "Per 100 g": "—" if per_100g is None else f"{per_100g:,.1f} {unit}",
                "Daily context": context,
            }
        )

    st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)

    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        ui.metric_card("Energy", format_number(breakdown.calories or 0, 0), "kcal", f"{food.energy_density()} density")
    with col2:
        protein = breakdown.values.get("protein") or 0.0
        ui.metric_card("Protein", format_number(protein, 1), "g", f"{breakdown.macro_energy_pct.get('protein', 0):.0f}% of energy")
    with col3:
        fiber = breakdown.values.get("fiber") or 0.0
        ui.metric_card("Fibre", format_number(fiber, 1), "g", f"{breakdown.macro_energy_pct.get('carbs', 0):.0f}% of energy is carbohydrate")

    ui.notice(
        "Portion values are <strong>linear extrapolations</strong> of per-100 g table values — "
        "no cooking-loss or density correction is applied. Home cooking varies with oil, salt and "
        "portion size, so treat every number as an estimate, not a measurement, and not as medical advice.",
        tone="info",
        icon="🧮",
    )


def render_health_score(food) -> None:
    """Show the rule-based score with its full breakdown."""
    health = score_item(food)
    ui.section(
        "Nutrition Health Score",
        "Rule-based, 0–100, calculated from the composition of 100 g — every point is explained below.",
        icon="🏅",
    )

    left, right = st.columns([1, 1.35], gap="large")
    with left:
        ui.metric_card(
            f"Health score · {food.name}",
            f"{health.total:.1f}",
            "/ 100",
            f"{ui.pill(health.band, health.tone)} {health.headline}",
            tone=health.tone,
        )
        st.progress(min(max(health.total / 100.0, 0.0), 1.0))
    with right:
        with st.expander("📐 How this score was calculated (all rules)", expanded=False):
            st.dataframe(
                pd.DataFrame(health.as_rows(), columns=["Rule", "Points", "Explanation"]),
                hide_index=True,
                use_container_width=True,
            )
            for note in health.notes:
                st.caption(f"• {note}")
            st.caption(
                "The score describes the food's composition, not how much you ate. It is a "
                "transparent wellness heuristic, not a medical or dietary prescription."
            )

    if health.tone == "bad":
        ui.notice(
            f"{food.name} scores {health.total:.0f}/100. It can still be part of a healthy pattern — "
            "the score simply means most of its energy comes with little protein, fibre or micronutrients.",
            tone="warning",
            icon="🍰",
        )


# --------------------------------------------------------------------------- #
# Phase 5 — explicit Save Meal
# --------------------------------------------------------------------------- #
def render_save_section(profile: UserProfile, food, portion: float, breakdown, health) -> None:
    """The only place in the app that writes a meal to the database.

    Duplicate protection works in three layers:

    1. the button disappears once a meal has been saved for this analysis
       (:data:`SESSION_SAVED_MEAL_ID` in session state) — a rerun cannot save twice,
    2. ``store_meal`` asks the database for an identical meal written in the last
       ``DUPLICATE_GUARD_SECONDS`` and returns that row instead of inserting, which
       also covers a double click racing a rerun,
    3. the user has to press the button explicitly; nothing else in the page writes.
    """
    ui.section(
        "Save this meal",
        "Meals are stored only when you press the button — analysis alone never writes to the database.",
        icon="💾",
    )

    already_saved = st.session_state.get(SESSION_SAVED_MEAL_ID)
    if already_saved:
        ui.notice(
            f"This meal is saved in your history as meal <strong>#{already_saved}</strong>. "
            "Open the **Dashboard** or **Meal History** page to review it.",
            tone="success",
            icon="✅",
        )
        action_columns = st.columns([1, 1, 2])
        with action_columns[0]:
            if st.button("🍽️ Save another meal", key="nv_save_another", width="stretch"):
                for key in (SESSION_SAVED_MEAL_ID, SESSION_SAVE_MESSAGE, SESSION_ANALYSIS):
                    st.session_state.pop(key, None)
                st.session_state.pop(SESSION_MATCHED_FOOD, None)
                st.session_state.pop(SESSION_PORTION, None)
                st.rerun()
        with action_columns[1]:
            try:
                st.page_link("pages/1_Dashboard.py", label="📊 Dashboard", icon="📊")
            except Exception:  # pragma: no cover - standalone execution
                st.caption("See the Dashboard page in the sidebar.")
        return

    if not db_ready():
        status = db_status()
        ui.notice(
            "The database is not available, so this meal cannot be stored yet. Your analysis stays on "
            f"screen and nothing is lost. Backend: <strong>{status.label}</strong>"
            + (f" — {status.error}" if status.error else ""),
            tone="warning",
            icon="🗄️",
        )
        st.caption("Run `python database/init_db.py --check`, fix the connection, then reload the page.")
        return

    if not getattr(profile, "user_id", None):
        ui.notice(
            "Your profile is not stored in the database yet. Meals are linked to a stored profile, so "
            "first open the main **NutriVision AI** page and press **Save profile to database**.",
            tone="warning",
            icon="👤",
        )
        return

    summary_rows = [
        ("Food", food.name),
        ("Portion", breakdown.portion_label()),
        ("Energy", f"{breakdown.values.get('calories') or 0:,.0f} kcal"),
        ("Protein / Fat / Carbs", (
            f"{breakdown.values.get('protein') or 0:,.1f} g / "
            f"{breakdown.values.get('fat') or 0:,.1f} g / "
            f"{breakdown.values.get('carbs') or 0:,.1f} g"
        )),
        ("Fibre", f"{breakdown.values.get('fiber') or 0:,.1f} g"),
        ("Nutrition Health Score", f"{health.total:.1f} / 100 ({health.band})"),
        ("Linked to", f"user_id {profile.user_id} · {profile.name}"),
    ]
    left, right = st.columns([1, 1.1], gap="large")
    with left:
        ui.card(title="What will be stored", icon="🧾", body_html=ui.rows_html(summary_rows))
    with right:
        ui.notice(
            "Saving stores the food name, portion and the values you see here. The photo itself is "
            "never uploaded or stored — it is processed in memory only.",
            tone="info",
            icon="🔐",
        )
        if st.button("💾 Save Meal", type="primary", key="nv_save_meal", width="stretch"):
            meal, message = store_meal(
                profile,
                food_name=food.name,
                portion_grams=float(portion),
                values=breakdown.values,
                health_score=health.total,
            )
            if meal is None:
                st.error(message)
                st.session_state[SESSION_SAVE_MESSAGE] = message
            else:
                st.session_state[SESSION_SAVED_MEAL_ID] = meal.meal_id
                flash(f"{message} Your dashboard now includes it.", "success")
                st.rerun()
        st.caption(
            "The button disables itself once the meal is stored, and an identical meal saved within "
            f"{DUPLICATE_GUARD_SECONDS} seconds is treated as a duplicate and ignored."
        )


def render_nutrition_step(prediction: FoodPrediction, profile: UserProfile) -> None:
    """The full Phase 4 flow: match → confirm → portion → results → score."""
    try:
        food = render_food_match(prediction, profile)
    except NutritionDataError as exc:
        st.error(f"The nutrition dataset could not be loaded: {exc}")
        return

    if food is None:
        return

    portion = render_portion_controls(food, profile)
    try:
        breakdown = _engine().calculate(food, portion)
    except ValueError as exc:
        st.error(str(exc))
        return

    st.session_state[SESSION_ANALYSIS + "_breakdown"] = {
        "food_id": food.food_id,
        "food_name": food.name,
        "portion_grams": breakdown.portion_grams,
        "values": breakdown.values,
        "health_score": score_item(food).total,
    }

    render_nutrition_results(breakdown, food, profile)
    render_health_score(food)
    render_save_section(profile, food, portion, breakdown, score_item(food))


def render_no_profile() -> None:
    """Phase guard: this page needs a profile from onboarding."""
    ui.hero(
        title="Create your profile first",
        subtitle="Meal analysis is personalised, so NutriVision needs your profile before you upload a photo.",
        eyebrow="Upload Meal · Step 0",
        icon="🥗",
    )
    ui.notice(
        "Open the main **NutriVision AI** page from the sidebar, complete the short onboarding "
        "form and come back to this page.",
        tone="info",
        icon="👤",
    )
    try:  # a direct link is nicer, but it only resolves inside the multipage app
        st.page_link("app.py", label="← Go to onboarding", icon="🥗")
    except Exception:  # pragma: no cover - e.g. when the page is run standalone
        st.caption("Sidebar → **NutriVision AI** → Create profile.")

    ui.page_footer(CURRENT_PHASE, APP_BUILD, APP_FOOTER_NOTE)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    init_session_state()
    ui.load_css()
    render_pending_flash()

    profile: Optional[UserProfile] = get_profile()
    if profile is None:
        render_no_profile()
        return

    ui.hero(
        title="Upload a meal photo",
        subtitle=(
            f"Hi {profile.name.split()[0]} — upload a photo of a single dish and NutriVision will "
            "recognise it with a pretrained Food-101 model, showing exactly how confident it is."
        ),
        eyebrow="Upload Meal · Phase 3",
        icon="📷",
    )

    ui.notice(
        "Only one dish per photo for now: the model performs single-food classification. "
        "Multi-item detection on a full thali plate is a documented future enhancement.",
        tone="info",
        icon="🍽️",
    )

    uploaded_file = st.file_uploader(
        "Choose a food photo",
        type=list(ALLOWED_IMAGE_TYPES),
        accept_multiple_files=False,
        help=f"JPG, JPEG or PNG up to {MAX_UPLOAD_MB:g} MB. Uploaded images are processed in memory "
        "and are never written to disk by the app.",
    )

    analyze_col, hint_col = st.columns([1, 3], gap="medium")
    with analyze_col:
        analyze_clicked = st.button(
            "🔍 Analyze Meal",
            type="primary",
            use_container_width=True,
            disabled=uploaded_file is None,
        )
    with hint_col:
        if uploaded_file is None:
            st.caption("Select a photo first, then press **Analyze Meal**.")
        else:
            st.caption("Analysis runs only when you press the button — nothing is automatic.")

    if uploaded_file is not None:
        try:
            info = validate_upload(uploaded_file, filename=uploaded_file.name)
        except UnsupportedImageError as exc:
            st.error(f"{exc}")
            st.session_state.pop(SESSION_ANALYSIS, None)
            render_model_status()
            return

        preview_col, meta_col = st.columns([1.15, 1], gap="large")
        with preview_col:
            st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)
        with meta_col:
            ui.section("Upload summary", "Validated before any model runs.", icon="📄")
            ui.card(
                body_html=ui.rows_html(
                    [
                        ("File", uploaded_file.name),
                        ("Image", info.describe()),
                        ("Size limit", f"{MAX_UPLOAD_MB:g} MB (accepted)"),
                        ("Allowed types", ", ".join(t.upper() for t in ALLOWED_IMAGE_TYPES)),
                    ]
                )
            )

        fingerprint = upload_fingerprint(uploaded_file)

        # Reuse a previous result for the same file (Streamlit reruns happen often).
        cached: Optional[dict[str, Any]] = st.session_state.get(SESSION_ANALYSIS)
        if cached and cached.get("key") == fingerprint and not analyze_clicked:
            from food_recognition import FoodPrediction  # local import keeps the import list small

            prediction = FoodPrediction.from_dict(cached["prediction"])
            render_analysis_result(prediction, info)
            render_nutrition_step(prediction, profile)
            st.caption(
                "Showing the cached result for this file. Press **Analyze Meal** again to re-run "
                "the model."
            )
            return

        if not analyze_clicked:
            render_model_status()
            return

        # ---- run the model -------------------------------------------------- #
        try:
            with st.spinner("Loading the food recognition model…"):
                load_food_model()
            with st.spinner("Analysing the photo…"):
                image = load_image_from_upload(uploaded_file)
                predictor = FoodPredictor()
                prediction = predictor.predict(image)
        except ModelUnavailableError as exc:
            logger.error("Model unavailable: %s", exc)
            st.session_state.pop(SESSION_ANALYSIS, None)
            render_model_error(exc)
            return
        except UnsupportedImageError as exc:
            st.error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - never crash the page on an unexpected error
            logger.exception("Unexpected error during prediction")
            st.error(f"Unexpected error while analysing the image: {exc.__class__.__name__}: {exc}")
            render_model_status()
            return

        st.session_state[SESSION_ANALYSIS] = {"key": fingerprint, "prediction": prediction.to_dict()}
        st.session_state.pop(SESSION_MATCHED_FOOD, None)    # new photo → new match
        st.session_state.pop(SESSION_PORTION, None)
        st.session_state.pop(SESSION_SAVED_MEAL_ID, None)   # ...and a fresh save slot
        st.session_state.pop(SESSION_SAVE_MESSAGE, None)
        logger.info("Stored analysis for %s (session only, not persisted)", uploaded_file.name)
        render_analysis_result(prediction, info)
        render_nutrition_step(prediction, profile)

        if st.button("🔁 Clear this result", use_container_width=False):
            st.session_state.pop(SESSION_ANALYSIS, None)
            st.rerun()

        render_model_status()
    ui.page_footer(CURRENT_PHASE, APP_BUILD, APP_FOOTER_NOTE)


main()
