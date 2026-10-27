"""Phase 7 tests: the Gemini coach, its snapshot, its failure handling, storage.

Nothing here touches the network or needs an API key: the SDK client is injected
with a fake, so the request shape, every failure mode and the database writes are
all testable offline. One test additionally builds the *real* SDK error class to
prove the classifier understands it.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Optional

import pytest

from ai import (
    ERROR_ADVICE,
    ERROR_BLOCKED,
    ERROR_EMPTY,
    ERROR_INVALID_KEY,
    ERROR_MODEL_NOT_FOUND,
    ERROR_NETWORK,
    ERROR_NOT_CONFIGURED,
    ERROR_PERMISSION,
    ERROR_QUOTA,
    ERROR_SERVER,
    ERROR_TIMEOUT,
    ERROR_UNKNOWN,
    GeminiCoach,
    build_context,
    coach_status,
)
from ai.gemini_coach import ERROR_SDK_MISSING, SYSTEM_INSTRUCTION
from config import GEMINI_MAX_OUTPUT_TOKENS, GEMINI_MODEL_NAME, GEMINI_TEMPERATURE
from database import repository as repo
from models import Meal, Recommendation
from utils import data_access
from utils.helpers import build_profile

# Reuse the throw-away SQLite database and the synthetic-history helpers.
from tests.test_pnie import ANCHOR, daily_rows

ANSWER = (
    "Your protein average is 71 g against a 103 g target. Try these: add 150 g of curd to "
    "breakfast, swap one chapati for 100 g of paneer at dinner, and keep a boiled egg as an "
    "evening snack. These are estimates from the meals you logged. Small changes add up."
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeUsage:
    def __init__(self, prompt: int = 120, output: int = 180) -> None:
        self.prompt_token_count = prompt
        self.candidates_token_count = output
        self.total_token_count = prompt + output


class FakePart:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeContent:
    def __init__(self, text: str) -> None:
        self.parts = [FakePart(text)]


class FakeCandidate:
    def __init__(self, text: str = "", finish_reason: str = "STOP") -> None:
        self.content = FakeContent(text) if text else None
        self.finish_reason = finish_reason


class FakeFeedback:
    def __init__(self, block_reason: Optional[str]) -> None:
        self.block_reason = block_reason


class FakeResponse:
    """Mimics the shape of ``GenerateContentResponse``."""

    def __init__(
        self,
        text: str = ANSWER,
        block_reason: Optional[str] = None,
        finish_reason: str = "STOP",
        usage: Optional[FakeUsage] = None,
    ) -> None:
        self.text = text
        self.candidates = [FakeCandidate(text=text, finish_reason=finish_reason)]
        self.prompt_feedback = FakeFeedback(block_reason)
        self.usage_metadata = usage if usage is not None else FakeUsage()


class NoTextField(FakeResponse):
    """A response whose ``.text`` raises, exactly like the SDK when there are no parts."""

    def __init__(self, text: str = ANSWER) -> None:
        self.candidates = [FakeCandidate(text=text)]
        self.prompt_feedback = FakeFeedback(None)
        self.usage_metadata = FakeUsage()

    @property
    def text(self):  # type: ignore[override]
        raise ValueError("Response has no text parts")


class FakeApiError(Exception):
    """Stand-in for the SDK's APIError (code/status attributes)."""

    def __init__(self, code: int, status: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


class FakeModels:
    def __init__(self, response=None, error: Optional[Exception] = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, response=None, error: Optional[Exception] = None) -> None:
        self.models = FakeModels(response=response, error=error)

    @property
    def calls(self) -> list[dict]:
        return self.models.calls


def coach_with(response=None, error: Optional[Exception] = None) -> GeminiCoach:
    return GeminiCoach(client=FakeClient(response=response, error=error))


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture()
def profile():
    return build_profile(
        {
            "name": "Coach Tester",
            "age": 29,
            "gender": "Female",
            "height_cm": 163.0,
            "weight_kg": 64.5,
            "activity_level": "Moderately active (moderate exercise 3-5 days/week)",
            "diet_preference": "Vegetarian",
            "fitness_goal": "Weight Loss",
        }
    )


@pytest.fixture()
def outcome():
    prediction = SimpleNamespace(
        value=1847.6, low=1813.0, high=1882.0, trend_direction="stable", mae=34.7,
        forecast_start=date(2026, 7, 1), forecast_end=date(2026, 7, 7),
    )
    trends = SimpleNamespace(
        energy_share={"protein": 0.15, "fat": 0.28, "carbohydrates": 0.57},
        latest_week_avg=1865.3, previous_week_avg=1849.1, slope_per_week=-18.0, direction="stable",
    )
    return SimpleNamespace(mode="model", trends=trends, prediction=prediction)


def meal(name: str, calories: float = 300.0, when: Optional[datetime] = None) -> Meal:
    return Meal(
        user_id=1,
        timestamp=when or datetime(2026, 6, 30, 20, 0, tzinfo=timezone.utc),
        food_name=name,
        portion_grams=150.0,
        calories=calories,
        protein=10.0,
        fat=8.0,
        carbohydrates=40.0,
        fiber=3.0,
        health_score=65.0,
    )


# --------------------------------------------------------------------------- #
# Snapshot / prompt
# --------------------------------------------------------------------------- #
def test_context_carries_the_profile_and_targets(profile, outcome) -> None:
    context = build_context(profile, daily_rows(14), [], outcome)
    assert context.age == 29 and context.gender == "Female"
    assert context.height_cm == pytest.approx(163.0)
    assert context.bmi == pytest.approx(24.3, abs=0.1)
    assert context.bmi_category == "Overweight"
    assert context.fitness_goal == "Weight Loss"
    assert context.daily_calorie_target == pytest.approx(profile.daily_calorie_target)
    assert context.protein_target_g == pytest.approx(profile.protein_target_g)


def test_context_averages_come_from_the_saved_days(profile) -> None:
    rows = daily_rows(14, calories=2000.0)
    rows[3]["calories"] = 1000.0          # one unusual day
    context = build_context(profile, rows, [], None)
    expected = sum(row["calories"] for row in rows) / len(rows)
    assert context.avg_calories == pytest.approx(expected, abs=0.1)
    assert context.logged_days == 14
    assert context.meals_logged == 42
    assert context.coverage_pct == pytest.approx(100.0)


def test_context_counts_unlogged_days_in_the_period(profile) -> None:
    context = build_context(profile, daily_rows(14, skip=(4, 5, 6, 7)), [], None)
    assert context.logged_days == 10
    assert context.days_covered == 14
    assert context.coverage_pct == pytest.approx(71.4, abs=0.1)
    assert any("no saved meals" in note for note in context.notes)


def test_context_caps_recent_meals_and_ranks_top_foods(profile) -> None:
    meals = [meal("Chapati") for _ in range(5)] + [meal("Poha") for _ in range(3)] + [meal("Idli")]
    context = build_context(profile, daily_rows(7), meals, None, max_meals=3, max_foods=2)
    assert len(context.recent_meals) == 3
    assert context.top_foods == (("Chapati", 5), ("Poha", 3))
    assert context.recent_meals[0]["food"] == "Chapati"
    assert context.recent_meals[0]["grams"] == pytest.approx(150.0)


def test_context_copies_the_forecast_verbatim(profile, outcome) -> None:
    context = build_context(profile, daily_rows(14), [], outcome)
    assert context.prediction_value == pytest.approx(1847.6)
    assert context.prediction_low == pytest.approx(1813.0)
    assert context.prediction_mae == pytest.approx(34.7)
    assert context.prediction_direction == "stable"
    assert context.prediction_mode == "model"
    assert context.has_prediction is True


def test_context_without_meals_is_explicitly_empty(profile) -> None:
    context = build_context(profile, [], [], None)
    assert context.logged_days == 0 and context.avg_calories is None
    assert context.has_meal_history is False
    assert any("No meals have been saved yet" in note for note in context.notes)
    assert any("No calorie forecast is available" in note for note in context.notes)


def test_context_flags_thin_history_and_missing_forecast(profile) -> None:
    context = build_context(profile, daily_rows(2), [meal("Poha")], None)
    assert any("provisional" in note for note in context.notes)
    assert any("forecast needs more logged weeks" in note for note in context.notes)


def test_prompt_contains_the_numbers_the_model_may_use(profile, outcome) -> None:
    context = build_context(profile, daily_rows(14), [meal("Chapati", 356.0)], outcome)
    prompt = context.to_prompt()
    assert "1,684 kcal/day" in prompt or "1,684 kcal" in prompt
    assert "24 g fibre" in prompt or "fibre" in prompt
    assert "1,848 kcal/day" in prompt
    assert "Chapati" in prompt
    assert "1,880 kcal" in prompt or "1,800 kcal" in prompt
    assert prompt.startswith("USER SNAPSHOT")


def test_prompt_never_contains_the_user_name(profile, outcome) -> None:
    """Privacy: the snapshot carries no direct identifier."""
    context = build_context(profile, daily_rows(7), [meal("Poha")], outcome)
    assert context.name == "Coach Tester"          # kept for the UI
    assert "Coach Tester" not in context.to_prompt()
    assert "Coach Tester" not in json.dumps(context.to_dict()["recent_meals"])


def test_prompt_is_deterministic(profile, outcome) -> None:
    context = build_context(profile, daily_rows(14), [meal("Chapati")], outcome)
    assert context.to_prompt() == context.to_prompt()
    assert context.to_prompt("Any question?") == context.to_prompt("Any question?")


def test_prompt_includes_the_question_only_when_asked(profile, outcome) -> None:
    context = build_context(profile, daily_rows(14), [], outcome)
    assert "QUESTION FROM THE USER" not in context.to_prompt()
    assert "QUESTION FROM THE USER: More protein please?" in context.to_prompt("More protein please?")


def test_prompt_is_truncated_for_absurdly_long_histories(profile) -> None:
    context = build_context(profile, daily_rows(14), [meal(f"Food {i}") for i in range(50)], None,
                            max_meals=50)
    prompt = context.to_prompt(max_chars=1200)
    assert len(prompt) <= 1200
    assert "Snapshot truncated" in prompt


def test_context_dict_is_json_safe(profile, outcome) -> None:
    context = build_context(profile, daily_rows(5), [meal("Poha")], outcome)
    payload = json.dumps(context.to_dict())
    assert "Coach Tester" in payload
    assert "NaN" not in payload and "Infinity" not in payload
    assert context.as_rows() and all(isinstance(value, str) for _, value in context.as_rows())


# --------------------------------------------------------------------------- #
# Configuration / status
# --------------------------------------------------------------------------- #
def test_status_without_a_key_is_not_configured(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    status = coach_status()
    assert status.configured is False
    assert status.sdk_available is True
    assert "GEMINI_API_KEY" in status.reason
    assert status.label == "API key not configured"


def test_status_reads_the_key_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "env-test-key")
    status = coach_status()
    assert status.configured is True
    assert status.key_source == "environment variable"
    assert "env-test-key" not in status.label       # the key never appears in the UI


def test_status_accepts_the_sdk_missing_case(monkeypatch) -> None:
    import ai.gemini_coach as module

    monkeypatch.setattr(module, "google_genai", None)
    monkeypatch.setenv("GEMINI_API_KEY", "env-test-key")
    status = coach_status()
    assert status.configured is False
    assert status.sdk_available is False
    assert "google-genai" in status.reason


def test_not_configured_never_builds_a_client(profile, outcome, monkeypatch) -> None:
    """Without a key the coach stops before any network client exists."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    coach = GeminiCoach(api_key="")
    result = coach.generate(build_context(profile, daily_rows(7), [], outcome))
    assert result.ok is False
    assert result.error_kind == ERROR_NOT_CONFIGURED
    assert result.advice_text
    assert coach._client is None, "no SDK client may be created without a key"
    assert result.prompt.startswith("USER SNAPSHOT")   # the snapshot is still available


# --------------------------------------------------------------------------- #
# A successful call
# --------------------------------------------------------------------------- #
def test_successful_call_returns_text_and_builds_the_request(profile, outcome) -> None:
    coach = coach_with(response=FakeResponse())
    context = build_context(profile, daily_rows(14), [meal("Chapati")], outcome)
    result = coach.generate(context, question="How do I add protein?")

    assert result.ok is True and result.failed is False
    assert result.text == ANSWER
    assert result.model == GEMINI_MODEL_NAME
    assert result.question == "How do I add protein?"

    call = coach._client.calls[0]
    assert call["model"] == GEMINI_MODEL_NAME
    assert "QUESTION FROM THE USER: How do I add protein?" in call["contents"]
    assert "Coach Tester" not in call["contents"]
    config = call["config"]
    assert config.temperature == pytest.approx(GEMINI_TEMPERATURE)
    assert config.max_output_tokens == GEMINI_MAX_OUTPUT_TOKENS
    assert "Never invent" in str(config.system_instruction)


def test_success_records_tokens_latency_and_finish_reason(profile, outcome) -> None:
    coach = coach_with(response=FakeResponse(usage=FakeUsage(100, 250)))
    result = coach.generate(build_context(profile, daily_rows(7), [], outcome))
    assert result.ok is True
    assert result.prompt_tokens == 100
    assert result.output_tokens == 250
    assert result.total_tokens == 350
    assert result.latency_ms >= 0
    assert result.finish_reason == "STOP"
    payload = result.to_dict()
    assert payload["ok"] is True and payload["text_length"] == len(ANSWER)


def test_system_instruction_forbids_fabrication_and_medical_claims() -> None:
    lowered = SYSTEM_INSTRUCTION.lower()
    assert "never invent" in lowered
    assert "no diagnosis" in lowered
    assert "general wellness information" in lowered
    assert "disclaimer" in lowered          # the app shows its own, the model must not
    assert "estimates" in lowered


# --------------------------------------------------------------------------- #
# Failure handling
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (FakeApiError(400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key."),
         ERROR_INVALID_KEY),
        (FakeApiError(429, "RESOURCE_EXHAUSTED", "429 RESOURCE_EXHAUSTED. Quota exceeded"),
         ERROR_QUOTA),
        (FakeApiError(403, "PERMISSION_DENIED", "403 PERMISSION_DENIED"), ERROR_PERMISSION),
        (FakeApiError(404, "NOT_FOUND", "models/gemini-9 not found"), ERROR_MODEL_NOT_FOUND),
        (FakeApiError(503, "UNAVAILABLE", "503 UNAVAILABLE. The model is overloaded"), ERROR_SERVER),
        (TimeoutError("deadline exceeded"), ERROR_TIMEOUT),
        (ConnectionError("connection reset by peer"), ERROR_NETWORK),
        (ValueError("something strange"), ERROR_UNKNOWN),
    ],
)
def test_failures_are_classified_and_never_raise(profile, outcome, error, expected) -> None:
    coach = coach_with(error=error)
    result = coach.generate(build_context(profile, daily_rows(7), [], outcome))
    assert result.ok is False
    assert result.error_kind == expected
    assert result.text == ""            # no fallback text is ever produced
    assert result.error_message
    assert result.advice_text           # every kind has advice for the user


def test_quota_message_is_recognised_even_without_a_code(profile, outcome) -> None:
    coach = coach_with(error=RuntimeError("429 Too Many Requests: rate limit exceeded"))
    result = coach.generate(build_context(profile, daily_rows(7), [], outcome))
    assert result.error_kind == ERROR_QUOTA


def test_the_api_key_is_scrubbed_from_error_messages(profile, outcome) -> None:
    secret = "AIzaSyFAKE-KEY-FOR-TESTS-1234567890"
    coach = coach_with(
        error=FakeApiError(400, "INVALID_ARGUMENT", f"API key not valid. Rejected key: {secret}")
    )
    coach._api_key = secret
    result = coach.generate(build_context(profile, daily_rows(7), [], outcome))
    assert result.error_kind == ERROR_INVALID_KEY
    assert secret not in result.error_message
    assert "***" in result.error_message
    assert secret not in result.prompt


def test_real_sdk_client_error_is_classified(profile, outcome) -> None:
    """The classifier must understand the SDK's own exception class, not just our fake."""
    from google.genai import errors as genai_errors

    error = genai_errors.ClientError(
        code=400,
        response_json={"error": {"code": 400, "message": "API key not valid. Please pass a valid API key.",
                                 "status": "INVALID_ARGUMENT"}},
    )
    result = coach_with(error=error).generate(build_context(profile, daily_rows(7), [], outcome))
    assert result.error_kind == ERROR_INVALID_KEY


def test_blocked_response_is_reported(profile, outcome) -> None:
    coach = coach_with(response=FakeResponse(text="", block_reason="SAFETY", finish_reason="SAFETY"))
    result = coach.generate(build_context(profile, daily_rows(7), [], outcome))
    assert result.ok is False
    assert result.error_kind == ERROR_BLOCKED
    assert "safety filters" in result.error_message


def test_empty_response_is_reported(profile, outcome) -> None:
    coach = coach_with(response=FakeResponse(text=""))
    result = coach.generate(build_context(profile, daily_rows(7), [], outcome))
    assert result.ok is False
    assert result.error_kind == ERROR_EMPTY


def test_response_whose_text_raises_falls_back_to_the_parts(profile, outcome) -> None:
    coach = coach_with(response=NoTextField(text=ANSWER))
    result = coach.generate(build_context(profile, daily_rows(7), [], outcome))
    assert result.ok is True
    assert result.text == ANSWER


def test_every_error_kind_has_advice() -> None:
    kinds = [
        ERROR_NOT_CONFIGURED, ERROR_SDK_MISSING, ERROR_INVALID_KEY, ERROR_PERMISSION,
        ERROR_MODEL_NOT_FOUND, ERROR_QUOTA, ERROR_SERVER, ERROR_TIMEOUT, ERROR_NETWORK,
        ERROR_BLOCKED, ERROR_EMPTY, ERROR_UNKNOWN,
    ]
    for kind in kinds:
        assert ERROR_ADVICE.get(kind), f"missing advice for {kind}"


# --------------------------------------------------------------------------- #
# Storage (recommendations table)
# --------------------------------------------------------------------------- #
@pytest.fixture()
def stored_user(user_factory):
    user = user_factory("Coach User")
    meals = []
    for offset in range(10):
        day = ANCHOR - timedelta(days=offset)
        for hour in (8, 20):
            meals.append(
                Meal(
                    user_id=user.user_id,
                    timestamp=datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)
                    + timedelta(hours=hour),
                    food_name="Chapati (whole wheat roti)" if hour == 8 else "Dal (cooked toor dal)",
                    portion_grams=120.0,
                    calories=356.0,
                    protein=11.0,
                    fat=8.5,
                    carbohydrates=59.0,
                    fiber=7.0,
                    health_score=72.0,
                )
            )
    assert repo.save_meals_bulk(meals) == len(meals)
    return user


def test_coach_context_is_built_from_real_meals_and_the_pnie(stored_user, profile) -> None:
    profile.user_id = stored_user.user_id
    context = data_access.coach_context(stored_user.user_id, profile)
    assert context.logged_days == 10
    assert context.meals_logged == 20
    assert context.avg_calories == pytest.approx(712.0)          # 2 x 356 kcal per day
    assert context.top_foods[0] == ("Chapati (whole wheat roti)", 10)
    assert context.has_prediction is False                       # 10 days is too short to forecast
    assert context.prediction_mode in {"insufficient_history", "descriptive_only", "no_data"}


def test_request_recommendation_stores_the_answer(stored_user, profile) -> None:
    profile.user_id = stored_user.user_id
    context = data_access.coach_context(stored_user.user_id, profile)
    coach = coach_with(response=FakeResponse())

    outcome = data_access.request_recommendation(stored_user.user_id, context, "More protein?", coach)
    assert outcome.ok is True
    assert outcome.text == ANSWER

    row = repo.get_latest_recommendation(stored_user.user_id)
    assert row is not None
    assert row.recommendation_text == ANSWER
    assert row.source == GEMINI_MODEL_NAME
    assert row.prompt == "More protein?"
    assert row.user_id == stored_user.user_id
    assert len(data_access.stored_recommendations(stored_user.user_id)) == 1
    assert data_access.coach_is_stale(row) is False


def test_a_failed_call_stores_nothing(stored_user, profile) -> None:
    profile.user_id = stored_user.user_id
    context = data_access.coach_context(stored_user.user_id, profile)
    outcome = data_access.request_recommendation(
        stored_user.user_id, context, "hello", coach_with(error=FakeApiError(503, "UNAVAILABLE", "boom"))
    )
    assert outcome.ok is False
    assert outcome.result.error_kind == ERROR_SERVER
    assert repo.list_recommendations(stored_user.user_id) == []


def test_the_same_question_is_not_asked_twice_in_a_row(stored_user, profile) -> None:
    profile.user_id = stored_user.user_id
    context = data_access.coach_context(stored_user.user_id, profile)
    coach = coach_with(response=FakeResponse())

    first = data_access.request_recommendation(stored_user.user_id, context, "Same question", coach)
    second = data_access.request_recommendation(stored_user.user_id, context, "Same question", coach)

    assert first.ok and second.ok
    assert second.reused is True
    assert second.text == ANSWER
    assert len(coach._client.calls) == 1, "the guard must prevent a second API call"
    assert len(repo.list_recommendations(stored_user.user_id)) == 1


def test_a_different_question_calls_the_model_again(stored_user, profile) -> None:
    profile.user_id = stored_user.user_id
    context = data_access.coach_context(stored_user.user_id, profile)
    coach = coach_with(response=FakeResponse())

    data_access.request_recommendation(stored_user.user_id, context, "Question one", coach)
    second = data_access.request_recommendation(stored_user.user_id, context, "Question two", coach)

    assert second.reused is False
    assert len(coach._client.calls) == 2
    assert len(repo.list_recommendations(stored_user.user_id)) == 2


def test_stored_recommendation_is_marked_stale_after_a_week(stored_user) -> None:
    row = Recommendation(
        user_id=stored_user.user_id,
        timestamp=datetime.now(timezone.utc) - timedelta(days=10),
        recommendation_text="Old advice",
        source=GEMINI_MODEL_NAME,
    )
    saved = repo.save_recommendation(row)
    assert saved is not None
    assert data_access.coach_is_stale(saved) is True
    assert data_access.coach_is_stale(None) is False
