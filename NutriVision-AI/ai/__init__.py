"""AI layer (Phase 7) — the Gemini 2.5 Flash nutrition coach.

Public API
----------
* :class:`GeminiCoach` — one defensive wrapper around the ``google-genai`` SDK:
  builds the request, classifies every failure, never raises.
* :func:`build_context` / :class:`NutritionContext` — the factual snapshot sent
  to the model (profile, targets, saved-meal averages, PNIE forecast).
* :class:`CoachResult` / :class:`CoachStatus` — what the UI renders, including
  the exact reason when the coach cannot run.
* :func:`generate_recommendation` — one-shot helper.
* ``ERROR_*`` constants + :data:`ERROR_ADVICE` — stable failure kinds and the
  plain-language advice shown for each one.

Nothing in this package runs on page load: a call is made only when the user
presses a button, so API quota is never spent on reruns.
"""

from ai.gemini_coach import (
    ERROR_ADVICE,
    ERROR_BLOCKED,
    ERROR_EMPTY,
    ERROR_INVALID_KEY,
    ERROR_MODEL_NOT_FOUND,
    ERROR_NETWORK,
    ERROR_NOT_CONFIGURED,
    ERROR_PERMISSION,
    ERROR_QUOTA,
    ERROR_SDK_MISSING,
    ERROR_SERVER,
    ERROR_TIMEOUT,
    ERROR_UNKNOWN,
    SYSTEM_INSTRUCTION,
    CoachError,
    CoachResult,
    CoachStatus,
    GeminiCoach,
    NutritionContext,
    build_context,
    coach_status,
    generate_recommendation,
)

__all__ = [
    "GeminiCoach",
    "NutritionContext",
    "CoachResult",
    "CoachStatus",
    "CoachError",
    "build_context",
    "generate_recommendation",
    "coach_status",
    "SYSTEM_INSTRUCTION",
    "ERROR_NOT_CONFIGURED",
    "ERROR_SDK_MISSING",
    "ERROR_INVALID_KEY",
    "ERROR_PERMISSION",
    "ERROR_MODEL_NOT_FOUND",
    "ERROR_QUOTA",
    "ERROR_SERVER",
    "ERROR_TIMEOUT",
    "ERROR_NETWORK",
    "ERROR_BLOCKED",
    "ERROR_EMPTY",
    "ERROR_UNKNOWN",
    "ERROR_ADVICE",
]
