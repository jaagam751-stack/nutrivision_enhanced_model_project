"""Gemini 2.5 Flash nutrition coach (Phase 7).

What this module does
---------------------
It turns the app's *own* numbers — the profile, its calculated targets, the saved
meals and the PNIE forecast — into one factual snapshot, sends that snapshot to
Gemini 2.5 Flash with a strict system instruction, and returns the answer. The
answer is advisory text; every figure inside it comes from the snapshot, never
from the model's imagination.

Rules this module enforces
--------------------------
* **The key never leaks.** It is read from Streamlit secrets / the environment,
  is never logged, and is scrubbed out of any error message the API returns.
* **Failures are reported, not dressed up.** Quota limits, a bad key, a timeout,
  a blocked response or an empty answer all come back as a
  :class:`CoachResult` with ``ok=False`` and a specific ``error_kind`` plus a
  plain-language explanation. There is **no** canned fallback text pretending to
  be a recommendation.
* **One call, on demand.** Nothing here runs on page load; the UI calls
  :meth:`GeminiCoach.generate` only after the user presses a button, so quota is
  never spent on reruns.
* **Offline-safe.** The SDK is imported defensively: without it (or without a
  key) the rest of the app keeps working and the coach explains what is missing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Mapping, Optional, Sequence

from config import (
    GEMINI_MAX_MEALS_IN_CONTEXT,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MAX_PROMPT_CHARS,
    GEMINI_MAX_TOP_FOODS,
    GEMINI_MODEL_NAME,
    GEMINI_REQUEST_TIMEOUT_MS,
    GEMINI_TARGET_WORDS,
    GEMINI_TEMPERATURE,
    GEMINI_THINKING_BUDGET,
    GEMINI_TOP_P,
    SECRET_GEMINI_API_KEY,
    get_secret,
    COACH_MIN_LOGGED_DAYS,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Optional SDK
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - exercised only when the dependency is missing
    from google import genai as google_genai
    from google.genai import errors as genai_errors
    from google.genai import types as genai_types
except Exception as exc:  # noqa: BLE001 - reported to the user, never hidden
    google_genai = None  # type: ignore[assignment]
    genai_errors = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]
    _SDK_IMPORT_ERROR = f"{exc.__class__.__name__}: {exc}"
else:
    _SDK_IMPORT_ERROR = ""

#: ``error_kind`` values used by :class:`CoachResult` (stable, so tests and the
#: UI can rely on them).
ERROR_NOT_CONFIGURED = "not_configured"
ERROR_SDK_MISSING = "sdk_missing"
ERROR_INVALID_KEY = "invalid_key"
ERROR_PERMISSION = "permission_denied"
ERROR_MODEL_NOT_FOUND = "model_not_found"
ERROR_QUOTA = "quota_exceeded"
ERROR_SERVER = "server_error"
ERROR_TIMEOUT = "timeout"
ERROR_NETWORK = "network_error"
ERROR_BLOCKED = "blocked"
ERROR_EMPTY = "empty_response"
ERROR_UNKNOWN = "unknown_error"

#: What the user can do about each failure (shown in the UI, keeps the page honest).
ERROR_ADVICE: dict[str, str] = {
    ERROR_NOT_CONFIGURED: (
        "Add GEMINI_API_KEY to your Streamlit secrets (cloud) or your local .env file, then reload "
        "this page. Get a free key at https://aistudio.google.com/app/apikey"
    ),
    ERROR_SDK_MISSING: "Install the SDK with `pip install -r requirements.txt` and restart the app.",
    ERROR_INVALID_KEY: "The API key was rejected — check it for typos or create a new one in Google AI Studio.",
    ERROR_PERMISSION: (
        "The key is valid but not allowed to use this model. Enable the Generative Language API for "
        "the project that owns the key, or use a key from a project where it is enabled."
    ),
    ERROR_MODEL_NOT_FOUND: "The configured model name is not available for this key — check GEMINI_MODEL_NAME.",
    ERROR_QUOTA: "The free-tier quota or rate limit was reached. Wait a few minutes and try again.",
    ERROR_SERVER: "Google's service returned an error. This is usually temporary — try again shortly.",
    ERROR_TIMEOUT: "The request took too long. Try again; a shorter question or a smaller history helps.",
    ERROR_NETWORK: "The app could not reach Google's API. Check your internet connection and try again.",
    ERROR_BLOCKED: (
        "The response was blocked by the model's safety filters, so there is nothing to show. "
        "Rephrase the question if you asked something unusual."
    ),
    ERROR_EMPTY: "The model returned an empty answer, so nothing is shown rather than invented text.",
    ERROR_UNKNOWN: "Something unexpected happened. The full error is shown in the app logs.",
}


class CoachError(RuntimeError):
    """Raised only by helper APIs that prefer an exception over a result object."""


# --------------------------------------------------------------------------- #
# The snapshot sent to the model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class NutritionContext:
    """Everything the coach is allowed to see — built from the app's own data.

    Every field is produced by the existing engines (profile maths, nutrition
    engine, PNIE). Nothing here is invented, and anything missing stays ``None``
    and is described as missing in the prompt.
    """

    # profile
    name: str = ""
    age: Optional[int] = None
    gender: str = ""
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    bmi: Optional[float] = None
    bmi_category: str = ""
    activity_level: str = ""
    diet_preference: str = ""
    fitness_goal: str = ""

    # targets
    daily_calorie_target: Optional[float] = None
    protein_target_g: Optional[float] = None
    fiber_target_g: Optional[float] = None

    # logged history
    window_start: Optional[date] = None
    window_end: Optional[date] = None
    days_covered: int = 0
    logged_days: int = 0
    meals_logged: int = 0
    coverage_pct: float = 0.0
    avg_calories: Optional[float] = None
    avg_protein: Optional[float] = None
    avg_fat: Optional[float] = None
    avg_carbohydrates: Optional[float] = None
    avg_fiber: Optional[float] = None
    avg_health_score: Optional[float] = None
    energy_share: Optional[dict[str, float]] = None
    latest_week_avg: Optional[float] = None
    previous_week_avg: Optional[float] = None
    slope_per_week: Optional[float] = None
    trend_direction: str = ""

    # PNIE forecast (optional)
    prediction_value: Optional[float] = None
    prediction_low: Optional[float] = None
    prediction_high: Optional[float] = None
    prediction_direction: str = ""
    prediction_mae: Optional[float] = None
    prediction_start: Optional[date] = None
    prediction_end: Optional[date] = None
    prediction_mode: str = ""

    recent_meals: tuple[dict[str, Any], ...] = ()
    top_foods: tuple[tuple[str, int], ...] = ()
    notes: tuple[str, ...] = ()

    # ------------------------------------------------------------------ #
    @property
    def has_prediction(self) -> bool:
        return self.prediction_value is not None

    @property
    def has_meal_history(self) -> bool:
        return self.logged_days > 0

    def to_prompt(self, question: str = "", max_chars: int = GEMINI_MAX_PROMPT_CHARS) -> str:
        """Render the snapshot as the plain-text prompt (deterministic)."""
        lines: list[str] = ["USER SNAPSHOT (every number comes from the app's own calculations)"]

        profile_bits = []
        if self.age is not None:
            profile_bits.append(f"{self.age} years old")
        if self.gender:
            profile_bits.append(self.gender)
        if self.height_cm is not None:
            profile_bits.append(f"{self.height_cm:.0f} cm")
        if self.weight_kg is not None:
            profile_bits.append(f"{self.weight_kg:.1f} kg")
        if self.bmi is not None:
            profile_bits.append(f"BMI {self.bmi:.1f} ({self.bmi_category})")
        # The user's name is deliberately NOT sent: the coach does not need it,
        # so the snapshot carries no direct identifier at all.
        lines.append("Profile: " + (" · ".join(profile_bits) if profile_bits else "not provided"))
        if self.activity_level or self.diet_preference or self.fitness_goal:
            lines.append(
                "Lifestyle: " + " · ".join(
                    part for part in (self.activity_level, self.diet_preference,
                                      f"Goal: {self.fitness_goal}" if self.fitness_goal else "")
                    if part
                )
            )

        targets = []
        if self.daily_calorie_target:
            targets.append(f"{self.daily_calorie_target:,.0f} kcal/day")
        if self.protein_target_g:
            targets.append(f"{self.protein_target_g:,.0f} g protein")
        if self.fiber_target_g:
            targets.append(f"{self.fiber_target_g:,.0f} g fibre")
        lines.append("Daily targets: " + (" · ".join(targets) if targets else "not available"))

        if self.has_meal_history:
            period = ""
            if self.window_start and self.window_end:
                period = f" ({self.window_start:%d %b %Y} – {self.window_end:%d %b %Y})"
            lines.append(
                f"Logging{period}: {self.logged_days} day(s) with meals out of {self.days_covered} "
                f"day(s) covered ({self.coverage_pct:.0f}%), {self.meals_logged} meal(s) saved"
            )
            averages = []
            for label, value, unit in (
                ("energy", self.avg_calories, "kcal"),
                ("protein", self.avg_protein, "g"),
                ("fat", self.avg_fat, "g"),
                ("carbohydrates", self.avg_carbohydrates, "g"),
                ("fibre", self.avg_fiber, "g"),
            ):
                if value is not None:
                    averages.append(f"{value:,.0f} {unit} {label}")
            if averages:
                lines.append("Average per logged day: " + " · ".join(averages))
            if self.avg_health_score is not None:
                lines.append(f"Average meal health score: {self.avg_health_score:,.0f} / 100")
            if self.energy_share:
                lines.append(
                    "Macro energy split: "
                    f"{self.energy_share['protein'] * 100:.0f}% protein · "
                    f"{self.energy_share['fat'] * 100:.0f}% fat · "
                    f"{self.energy_share['carbohydrates'] * 100:.0f}% carbohydrate"
                )
            if self.latest_week_avg is not None:
                change = ""
                if self.previous_week_avg is not None:
                    change = (
                        f" vs {self.previous_week_avg:,.0f} kcal/day the week before "
                        f"({self.latest_week_avg - self.previous_week_avg:+,.0f} kcal)"
                    )
                lines.append(f"Last 7 days: {self.latest_week_avg:,.0f} kcal/day{change}")
            if self.slope_per_week is not None:
                lines.append(
                    f"Trend direction: {self.trend_direction or 'unknown'} "
                    f"({self.slope_per_week:+,.0f} kcal/week over the logged period)"
                )
            if self.top_foods:
                lines.append(
                    "Most logged foods: "
                    + ", ".join(f"{name} (x{count})" for name, count in self.top_foods)
                )
            if self.recent_meals:
                lines.append("Most recent meals:")
                for meal in self.recent_meals:
                    when = meal.get("when") or "—"
                    lines.append(
                        f"  - {when}: {meal.get('food', 'food')} "
                        f"{meal.get('grams', 0):,.0f} g, {meal.get('calories', 0):,.0f} kcal"
                        + (f", health score {meal['score']:,.0f}" if meal.get("score") is not None else "")
                    )
        else:
            lines.append("Logging: no meals have been saved yet")

        if self.has_prediction:
            lines.append(
                f"PNIE forecast (XGBoost, next week): {self.prediction_value:,.0f} kcal/day average"
                + (
                    f" (estimated range {self.prediction_low:,.0f}–{self.prediction_high:,.0f})"
                    if self.prediction_low is not None and self.prediction_high is not None
                    else ""
                )
                + (
                    f", direction {self.prediction_direction}"
                    if self.prediction_direction
                    else ""
                )
                + (
                    f", typical error ± {self.prediction_mae:,.0f} kcal/day"
                    if self.prediction_mae is not None
                    else ""
                )
            )
            if self.prediction_start and self.prediction_end:
                lines.append(
                    f"Forecast window: {self.prediction_start:%d %b} – {self.prediction_end:%d %b}"
                )
        else:
            lines.append(
                "PNIE forecast: not available yet "
                f"({self.prediction_mode or 'insufficient history'})"
            )

        for note in self.notes:
            lines.append(f"Note: {note}")

        if question:
            lines.append("")
            lines.append(f"QUESTION FROM THE USER: {question.strip()}")

        prompt = "\n".join(lines)
        if len(prompt) > max_chars:
            prompt = (
                prompt[: max_chars - 240].rstrip()
                + "\n[Snapshot truncated: the history was longer than the app sends. "
                "Answer with what is above.]"
            )
        return prompt

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe snapshot (for logs, tests and the "what the coach sees" card)."""
        payload = {
            key: (value.isoformat() if isinstance(value, date) else value)
            for key, value in self.__dict__.items()
            if key not in ("recent_meals", "top_foods", "notes")
        }
        payload["recent_meals"] = [dict(meal) for meal in self.recent_meals]
        payload["top_foods"] = [list(item) for item in self.top_foods]
        payload["notes"] = list(self.notes)
        return payload

    def as_rows(self) -> list[tuple[str, str]]:
        """Compact label/value rows for the "what the coach sees" card."""
        rows: list[tuple[str, str]] = []
        if self.daily_calorie_target:
            rows.append(("Daily targets", f"{self.daily_calorie_target:,.0f} kcal · goal {self.fitness_goal}"))
        if self.has_meal_history:
            rows.append(
                ("History", f"{self.logged_days} logged day(s) of {self.days_covered} · "
                            f"{self.meals_logged} meal(s)")
            )
            if self.avg_calories is not None:
                rows.append(("Average intake", f"{self.avg_calories:,.0f} kcal per logged day"))
            if self.avg_health_score is not None:
                rows.append(("Average health score", f"{self.avg_health_score:,.0f} / 100"))
            if self.top_foods:
                rows.append(("Top foods", ", ".join(f"{name} (x{count})" for name, count in self.top_foods)))
        else:
            rows.append(("History", "no meals saved yet"))
        if self.has_prediction:
            rows.append(
                ("PNIE forecast",
                 f"{self.prediction_value:,.0f} kcal/day"
                 + (f" ({self.prediction_direction})" if self.prediction_direction else ""))
            )
        else:
            rows.append(("PNIE forecast", f"not available ({self.prediction_mode or 'no data'})"))
        if self.recent_meals:
            rows.append(("Meals included", f"last {len(self.recent_meals)}"))
        return rows


def build_context(
    profile: Any = None,
    daily_totals: Optional[Sequence[Mapping[str, Any]]] = None,
    meals: Optional[Sequence[Any]] = None,
    outcome: Any = None,
    max_meals: int = GEMINI_MAX_MEALS_IN_CONTEXT,
    max_foods: int = GEMINI_MAX_TOP_FOODS,
    min_logged_days: int = COACH_MIN_LOGGED_DAYS,
) -> NutritionContext:
    """Assemble the snapshot from the profile, saved meals and PNIE outcome.

    ``outcome`` is a :class:`pnie.PNIEOutcome` (or ``None``). Its descriptive
    trends and its forecast are copied verbatim — the coach never computes
    nutrition numbers of its own.
    """
    totals = list(daily_totals or [])
    logged_days = len(totals)
    window_start = min((row.get("date") for row in totals), default=None)
    window_end = max((row.get("date") for row in totals), default=None)
    days_covered = (
        (window_end - window_start).days + 1
        if window_start is not None and window_end is not None
        else 0
    )

    def _mean(key: str) -> Optional[float]:
        values = [float(row[key]) for row in totals if row.get(key) is not None]
        return round(sum(values) / len(values), 1) if values else None

    trends = getattr(outcome, "trends", None)
    prediction = getattr(outcome, "prediction", None)

    notes: list[str] = []
    if logged_days == 0:
        notes.append(
            "No meals have been saved yet, so there is no intake history. Give general, "
            "profile-based guidance and encourage saving meals — do not invent intake numbers."
        )
    elif logged_days < min_logged_days:
        notes.append(
            f"Only {logged_days} day(s) of meals have been saved; the averages are based on very "
            "little data and should be described as provisional."
        )
    if not getattr(prediction, "value", None):
        notes.append(
            "No calorie forecast is available yet, so do not mention a predicted intake; explain "
            "that the forecast needs more logged weeks."
        )
    if days_covered and logged_days < days_covered:
        notes.append(
            f"{days_covered - logged_days} day(s) in the period have no saved meals — average "
            "figures describe the days that were logged, not every day."
        )

    return NutritionContext(
        name=str(getattr(profile, "name", "") or ""),
        age=getattr(profile, "age", None),
        gender=str(getattr(profile, "gender", "") or ""),
        height_cm=getattr(profile, "height_cm", None),
        weight_kg=getattr(profile, "weight_kg", None),
        bmi=round(float(getattr(profile, "bmi", 0.0)), 1) if getattr(profile, "bmi", None) else None,
        bmi_category=str(getattr(profile, "bmi_category", "") or ""),
        activity_level=str(getattr(profile, "activity_level", "") or ""),
        diet_preference=str(getattr(profile, "diet_preference", "") or ""),
        fitness_goal=str(getattr(profile, "fitness_goal", "") or ""),
        daily_calorie_target=getattr(profile, "daily_calorie_target", None),
        protein_target_g=getattr(profile, "protein_target_g", None),
        fiber_target_g=getattr(profile, "fiber_target_g", None),
        window_start=window_start,
        window_end=window_end,
        days_covered=days_covered,
        logged_days=logged_days,
        meals_logged=int(sum(int(row.get("meals") or 0) for row in totals)),
        coverage_pct=round(logged_days / days_covered * 100.0, 1) if days_covered else 0.0,
        avg_calories=_mean("calories"),
        avg_protein=_mean("protein"),
        avg_fat=_mean("fat"),
        avg_carbohydrates=_mean("carbohydrates"),
        avg_fiber=_mean("fiber"),
        avg_health_score=_mean("avg_health_score"),
        energy_share=dict(trends.energy_share) if getattr(trends, "energy_share", None) else None,
        latest_week_avg=getattr(trends, "latest_week_avg", None),
        previous_week_avg=getattr(trends, "previous_week_avg", None),
        slope_per_week=getattr(trends, "slope_per_week", None),
        trend_direction=str(getattr(trends, "direction", "") or ""),
        prediction_value=getattr(prediction, "value", None),
        prediction_low=getattr(prediction, "low", None),
        prediction_high=getattr(prediction, "high", None),
        prediction_direction=str(getattr(prediction, "trend_direction", "") or ""),
        prediction_mae=getattr(prediction, "mae", None),
        prediction_start=getattr(prediction, "forecast_start", None),
        prediction_end=getattr(prediction, "forecast_end", None),
        prediction_mode=str(getattr(outcome, "mode", "") or ""),
        recent_meals=_recent_meals(meals, max_meals),
        top_foods=_top_foods(meals, max_foods),
        notes=tuple(notes),
    )


def _recent_meals(meals: Optional[Sequence[Any]], limit: int) -> tuple[dict[str, Any], ...]:
    """The newest ``limit`` meals, as compact dictionaries for the prompt."""
    rows: list[dict[str, Any]] = []
    for meal in list(meals or [])[:limit]:
        stamp = getattr(meal, "timestamp", None)
        rows.append(
            {
                "when": stamp.strftime("%d %b %H:%M") if hasattr(stamp, "strftime") else None,
                "food": str(getattr(meal, "food_name", "") or "food")[:60],
                "grams": float(getattr(meal, "portion_grams", 0) or 0),
                "calories": float(getattr(meal, "calories", 0) or 0),
                "score": float(getattr(meal, "health_score", 0) or 0),
            }
        )
    return tuple(rows)


def _top_foods(meals: Optional[Sequence[Any]], limit: int) -> tuple[tuple[str, int], ...]:
    """Most frequently saved foods (counts only — no invented nutrition)."""
    counts: dict[str, int] = {}
    for meal in meals or []:
        name = str(getattr(meal, "food_name", "") or "").strip()
        if name:
            counts[name] = counts.get(name, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return tuple(ranked[:limit])


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class CoachStatus:
    """Whether the coach can run at all, and where the key came from."""

    configured: bool
    key_source: str = "none"
    sdk_available: bool = False
    model: str = GEMINI_MODEL_NAME
    reason: str = ""

    @property
    def label(self) -> str:
        if self.configured:
            return f"Ready · {self.model}"
        if not self.sdk_available:
            return "SDK not installed"
        return "API key not configured"


@dataclass(frozen=True, eq=False)
class CoachResult:
    """Outcome of one coach call. Never raises, never invents text."""

    ok: bool
    text: str = ""
    model: str = ""
    error_kind: str = ""
    error_message: str = ""
    advice: str = ""
    question: str = ""
    prompt: str = ""
    latency_ms: int = 0
    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    finish_reason: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def failed(self) -> bool:
        return not self.ok

    @property
    def advice_text(self) -> str:
        return self.advice or ERROR_ADVICE.get(self.error_kind, "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "model": self.model,
            "error_kind": self.error_kind,
            "error_message": self.error_message,
            "advice": self.advice_text,
            "question": self.question,
            "latency_ms": self.latency_ms,
            "total_tokens": self.total_tokens,
            "finish_reason": self.finish_reason,
            "created_at": self.created_at.isoformat(),
            "text_length": len(self.text),
        }


# --------------------------------------------------------------------------- #
# The coach
# --------------------------------------------------------------------------- #
SYSTEM_INSTRUCTION = f"""You are the nutrition coach inside NutriVision AI, a wellness app used by
people in India who photograph their meals and track them.

You receive ONE user's snapshot: their profile, the app's calculated daily targets, the averages
of the meals they saved, and (sometimes) a machine-learning forecast of next week's average
calories.

Follow these rules exactly:
- Use ONLY the numbers in the snapshot. Never invent meals, portions, dates, nutrients or weights.
- If something is missing, say what is missing instead of guessing.
- Refer to Indian foods and eating patterns when the snapshot mentions them.
- Be practical and specific: mention foods, meal timings and swaps that fit the snapshot.
- Reply in at most {GEMINI_TARGET_WORDS} words: one short opening sentence, then 3-5 concise
  bullets with concrete actions, then one short encouraging closing line.
- Plain text with simple bullets only. No tables, no emoji, no markdown headings.
- General wellness information only: no diagnosis, no treatment, no promises of weight change.
- Never recommend eating less than the calorie target in the snapshot; instead suggest where the
  energy comes from and, if the user is already below target, suggest nutrient-dense additions.
- Say "these are estimates" once when you quote numbers.
- Do not add a medical disclaimer paragraph; the app displays its own.
"""


class GeminiCoach:
    """Thin, defensive wrapper around ``google-genai`` for one coaching call."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = GEMINI_MODEL_NAME,
        temperature: float = GEMINI_TEMPERATURE,
        top_p: float = GEMINI_TOP_P,
        max_output_tokens: int = GEMINI_MAX_OUTPUT_TOKENS,
        timeout_ms: int = GEMINI_REQUEST_TIMEOUT_MS,
        thinking_budget: Optional[int] = GEMINI_THINKING_BUDGET,
        client: Any = None,
    ) -> None:
        self._api_key, self.key_source = _resolve_api_key(api_key)
        self.model = model
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_output_tokens = int(max_output_tokens)
        self.timeout_ms = int(timeout_ms)
        self.thinking_budget = thinking_budget
        self._client = client          # injected client wins (used by the tests)
        self._owns_client = client is None

    # ------------------------------------------------------------------ #
    @property
    def status(self) -> CoachStatus:
        sdk_available = google_genai is not None
        if self._client is not None:
            return CoachStatus(configured=True, key_source="injected client",
                               sdk_available=sdk_available, model=self.model)
        if not sdk_available:
            return CoachStatus(configured=False, sdk_available=False, model=self.model,
                               reason=f"The google-genai package is not installed ({_SDK_IMPORT_ERROR}).")
        if not self._api_key:
            return CoachStatus(configured=False, sdk_available=True, model=self.model,
                               reason="No GEMINI_API_KEY found in Streamlit secrets or the environment.")
        return CoachStatus(configured=True, key_source=self.key_source,
                           sdk_available=True, model=self.model)

    @property
    def is_configured(self) -> bool:
        return self.status.configured

    # ------------------------------------------------------------------ #
    def _get_client(self) -> Any:
        """Build (once) and return a client, or raise a classified error."""
        if self._client is not None:
            return self._client
        if google_genai is None:  # pragma: no cover - guarded by status
            raise CoachError(_SDK_IMPORT_ERROR or "google-genai is not installed")
        options = genai_types.HttpOptions(timeout=self.timeout_ms)
        self._client = google_genai.Client(api_key=self._api_key, http_options=options)
        return self._client

    def _generation_config(self) -> Any:
        kwargs: dict[str, Any] = {
            "system_instruction": SYSTEM_INSTRUCTION,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_output_tokens": self.max_output_tokens,
        }
        if self.thinking_budget is not None:
            kwargs["thinking_config"] = genai_types.ThinkingConfig(thinking_budget=int(self.thinking_budget))
        return genai_types.GenerateContentConfig(**kwargs)

    def generate(
        self,
        context: NutritionContext,
        question: str = "",
        prompt_override: Optional[str] = None,
    ) -> CoachResult:
        """Ask the model once. **Never raises** — failures come back in the result.

        Args:
            context: the snapshot built by :func:`build_context`.
            question: optional question from the user (stored with the answer).
            prompt_override: used by tests to send a fixed prompt.
        """
        question = (question or "").strip()
        prompt = prompt_override or context.to_prompt(question)

        status = self.status
        if not status.configured:
            kind = ERROR_SDK_MISSING if not status.sdk_available else ERROR_NOT_CONFIGURED
            logger.warning("Coach not configured (%s)", kind)
            return CoachResult(ok=False, model=self.model, error_kind=kind,
                               error_message=status.reason, question=question, prompt=prompt)

        started = time.perf_counter()
        try:
            response = self._get_client().models.generate_content(
                model=self.model, contents=prompt, config=self._generation_config()
            )
        except Exception as exc:  # noqa: BLE001 - every failure is classified, none escape
            kind, message = _classify_error(exc, self._api_key)
            logger.warning("Coach call failed (%s): %s", kind, message)
            return CoachResult(
                ok=False, model=self.model, error_kind=kind, error_message=message,
                question=question, prompt=prompt,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )

        latency_ms = int((time.perf_counter() - started) * 1000)
        usage = getattr(response, "usage_metadata", None)
        tokens = {
            "prompt_tokens": _token(getattr(usage, "prompt_token_count", None)),
            "output_tokens": _token(getattr(usage, "candidates_token_count", None)),
            "total_tokens": _token(getattr(usage, "total_token_count", None)),
        }
        finish_reason = _finish_reason(response)

        block_reason = _block_reason(response)
        if block_reason:
            logger.info("Coach response blocked: %s", block_reason)
            return CoachResult(
                ok=False, model=self.model, error_kind=ERROR_BLOCKED,
                error_message=f"The response was blocked by the model's safety filters ({block_reason}).",
                question=question, prompt=prompt, latency_ms=latency_ms,
                finish_reason=finish_reason, **tokens,
            )

        text = _extract_text(response)
        if not text:
            logger.info("Coach returned no text (finish_reason=%s)", finish_reason)
            return CoachResult(
                ok=False, model=self.model, error_kind=ERROR_EMPTY,
                error_message="The model returned no text"
                + (f" (finish reason: {finish_reason})" if finish_reason else "") + ".",
                question=question, prompt=prompt, latency_ms=latency_ms,
                finish_reason=finish_reason, **tokens,
            )

        logger.info("Coach answered | model=%s chars=%s latency_ms=%s", self.model, len(text), latency_ms)
        return CoachResult(
            ok=True, text=text, model=self.model, question=question, prompt=prompt,
            latency_ms=latency_ms, finish_reason=finish_reason, **tokens,
        )


def _resolve_api_key(explicit: Optional[str] = None) -> tuple[str, str]:
    """Return ``(key, source)`` without ever logging the key itself."""
    if explicit:
        return str(explicit).strip(), "argument"
    from_streamlit = None
    try:  # Streamlit secrets first (deployment), then the environment
        import streamlit as st

        secrets = getattr(st, "secrets", None)
        if secrets is not None and SECRET_GEMINI_API_KEY in secrets:
            from_streamlit = str(secrets[SECRET_GEMINI_API_KEY]).strip()
    except Exception:  # no missing secrets.toml, no runtime — all fine
        from_streamlit = None
    if from_streamlit:
        return from_streamlit, "Streamlit secrets"
    value = get_secret(SECRET_GEMINI_API_KEY, "")
    if value:
        return str(value).strip(), "environment variable"
    return "", "none"


def _token(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _finish_reason(response: Any) -> str:
    """Best-effort finish reason (``STOP`` means the model completed normally)."""
    try:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return ""
        reason = getattr(candidates[0], "finish_reason", None)
        return str(getattr(reason, "name", reason) or "")
    except Exception:  # noqa: BLE001
        return ""


def _block_reason(response: Any) -> str:
    """Detect a safety block from either the prompt feedback or the candidate."""
    try:
        feedback = getattr(response, "prompt_feedback", None)
        reason = getattr(feedback, "block_reason", None) if feedback else None
        if reason:
            return str(getattr(reason, "name", reason))
        finish = _finish_reason(response)
        if finish.upper() in {"SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "RECITATION", "SPII"}:
            return finish
        return ""
    except Exception:  # noqa: BLE001
        return ""


def _extract_text(response: Any) -> str:
    """Pull the answer text out of a response, tolerating every shape the SDK uses.

    ``response.text`` raises when the response has no text parts, so it is wrapped;
    the fallback walks the candidate parts directly.
    """
    try:
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
    except Exception:  # noqa: BLE001 - the SDK raises ValueError for tool-only responses
        pass

    chunks: list[str] = []
    try:
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                piece = getattr(part, "text", None)
                if isinstance(piece, str) and piece.strip():
                    chunks.append(piece.strip())
    except Exception:  # noqa: BLE001
        return ""
    return "\n".join(chunks).strip()


def _redact(message: str, secret: str) -> str:
    """Remove anything that looks like the API key from a message before showing it."""
    text = str(message)
    if secret:
        text = text.replace(secret, "***")
        if len(secret) > 8:
            text = text.replace(secret[:8], "***")
    return text


def _classify_error(exc: Exception, api_key: str = "") -> tuple[str, str]:
    """Map an SDK/network exception onto a stable ``error_kind`` + safe message."""
    name = type(exc).__name__
    text = _redact(str(exc), api_key)
    lowered = text.lower()

    code = getattr(exc, "code", None)
    status = str(getattr(exc, "status", "") or "").upper()

    if code == 429 or "resource_exhausted" in lowered or "quota" in lowered or "rate limit" in lowered:
        return ERROR_QUOTA, _short(text, "The Gemini API rate limit or quota was reached.")
    if code in (401,):
        return ERROR_INVALID_KEY, _short(text, "The API key was rejected.")
    if code == 400 and ("api key not valid" in lowered or "api_key_invalid" in lowered
                        or "invalid api key" in lowered or "api key" in lowered):
        return ERROR_INVALID_KEY, _short(text, "The API key was rejected.")
    if "api key not valid" in lowered or "api_key_invalid" in lowered:
        return ERROR_INVALID_KEY, _short(text, "The API key was rejected.")
    if code == 403 or "permission" in lowered:
        return ERROR_PERMISSION, _short(text, "The API key is not permitted to use this model.")
    if code == 404 or "not found" in lowered and "model" in lowered:
        return ERROR_MODEL_NOT_FOUND, _short(text, "The configured model was not found.")
    if code is not None and isinstance(code, int) and 500 <= code < 600:
        return ERROR_SERVER, _short(text, f"Google's API returned HTTP {code}.")
    if "deadline" in lowered or "timeout" in lowered or "timed out" in lowered:
        return ERROR_TIMEOUT, _short(text, "The request timed out.")
    if isinstance(exc, (TimeoutError,)):
        return ERROR_TIMEOUT, _short(text, "The request timed out.")
    if isinstance(exc, (ConnectionError, OSError)) or "connection" in lowered or "network" in lowered:
        return ERROR_NETWORK, _short(text, "The app could not reach the Gemini API.")
    if status == "INVALID_ARGUMENT":
        return ERROR_UNKNOWN, _short(text, "The request was rejected as invalid.")
    if name in {"ServerError"}:
        return ERROR_SERVER, _short(text, "Google's API returned a server error.")
    if name in {"ClientError", "APIError"}:
        return ERROR_UNKNOWN, _short(text, "The Gemini API rejected the request.")
    return ERROR_UNKNOWN, _short(text, f"{name} while calling the Gemini API.")


def _short(message: str, fallback: str, limit: int = 240) -> str:
    """Keep the first meaningful line of an error, capped for the UI."""
    first_line = next((line.strip() for line in str(message).splitlines() if line.strip()), "")
    text = first_line or fallback
    return text[:limit] + ("…" if len(text) > limit else "")


def coach_status(**kwargs: Any) -> CoachStatus:
    """Convenience helper used by the page: is the coach ready?"""
    return GeminiCoach(**kwargs).status


def generate_recommendation(
    context: NutritionContext, question: str = "", coach: Optional[GeminiCoach] = None, **kwargs: Any
) -> CoachResult:
    """One-shot helper: ``generate_recommendation(context, question)``."""
    return (coach or GeminiCoach(**kwargs)).generate(context, question=question)
