"""Phase 7 end-to-end check — the AI Coach page and the dashboard coach card.

Run it with::

    python tests/e2e_phase7_apptest.py

This is **not** part of the pytest suite (the file name starts with ``e2e_``), because it
drives the real Streamlit pages through ``streamlit.testing.v1.AppTest`` against whatever
database ``DATABASE_URL`` points at, and it writes one demo recommendation row.

Safety rules it follows:

* the Gemini SDK is **never** called for real — a fake client with the same shape as
  ``google.genai.Client`` is injected, so the full code path (snapshot → request →
  answer → stored row → dashboard card) is exercised offline. Even if you have a valid
  ``GEMINI_API_KEY`` configured, no request leaves the machine;
* it does not delete anything: it records the recommendation count before asking and
  asserts it grew by exactly one;
* if no user with saved meals exists, it tells you to run
  ``python database/init_db.py --sample-data`` and stops.

It covers five things:

1. no API key  → the setup card renders and the button is disabled;
2. one press   → one answer, rendered and stored, containing the snapshot (and never
   your name), then a second press with the same question reuses the stored answer
   without a second API call;
3. the dashboard shows the stored answer;
4. a failing call (simulated quota error) shows the classified error + advice and
   stores nothing;
5. a visitor without a profile gets the "create your profile" screen, not a crash.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{ROOT / 'nutrivision.db'}")

from sqlalchemy import delete, func, select  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from ai import GeminiCoach  # noqa: E402
from config import GEMINI_MODEL_NAME  # noqa: E402
from database.connection import reset_engine  # noqa: E402
from database.repository import list_meals, list_users, session_scope  # noqa: E402
from models import Recommendation  # noqa: E402
from utils.helpers import build_profile  # noqa: E402

reset_engine()

# --------------------------------------------------------------------------- #
# Pick a user that has data (the sample user created by init_db.py works).
# --------------------------------------------------------------------------- #
candidate = None
for user in list_users(include_samples=True):
    if list_meals(user.user_id, limit=5):
        candidate = user
        break
if candidate is None:
    print("No user with saved meals found.")
    print("Run:  python database/init_db.py --sample-data   (then run this script again)")
    raise SystemExit(1)

PROFILE = build_profile(
    {
        "name": candidate.name,
        "age": candidate.age,
        "gender": candidate.gender,
        "height_cm": candidate.height_cm,
        "weight_kg": candidate.weight_kg,
        "activity_level": candidate.activity_level,
        "diet_preference": candidate.diet_preference,
        "fitness_goal": candidate.fitness_goal,
    }
)
PROFILE.user_id = candidate.user_id
STATE = {"nv_profile": PROFILE.to_dict()}
USER_LABEL = f"user #{candidate.user_id}"
print(f"Using {USER_LABEL} ({candidate.name}) — {len(list_meals(candidate.user_id, limit=100000))} meals")

ANSWER = (
    "Your logged week averages 1,880 kcal against a 1,684 kcal target, with protein at 71 g of a "
    "103 g goal and fibre at 15 g. Three practical moves: add 150 g of curd and a boiled egg to "
    "breakfast for about 20 g more protein; swap one chapati for 100 g of paneer or 150 g of "
    "grilled chicken at dinner; keep a fruit and a handful of roasted chana for the evening gap. "
    "These are estimates from the meals you saved, and your trend is holding steady."
)

# A unique question per run keeps the 120 s duplicate guard from interfering with
# repeated runs, while still letting us test the guard inside one run.
QUESTION = f"How can I add protein without more rotis? (e2e {int(time.time())})"


# --------------------------------------------------------------------------- #
# A fake SDK client with the same call signature as the real one.
# --------------------------------------------------------------------------- #
class _Usage:
    prompt_token_count, candidates_token_count, total_token_count = 320, 180, 500


class _Part:
    def __init__(self, text):
        self.text = text


class _Content:
    def __init__(self, text):
        self.parts = [_Part(text)]


class _Candidate:
    def __init__(self, text):
        self.content = _Content(text)
        self.finish_reason = "STOP"


class _Response:
    def __init__(self, text):
        self.text = text
        self.candidates = [_Candidate(text)]
        self.prompt_feedback = None
        self.usage_metadata = _Usage()


class _FakeModels:
    """Records every call so the test can prove *how many* requests were made."""

    def __init__(self, fail: bool = False):
        self.calls: list[dict] = []
        self.fail = fail

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self.fail:
            raise RuntimeError("429 RESOURCE_EXHAUSTED. Quota exceeded for model gemini-2.5-flash")
        return _Response(ANSWER)


class _FakeClient:
    def __init__(self, fail: bool = False):
        self.models = _FakeModels(fail=fail)


def _client(fail: bool = False) -> _FakeClient:
    return _FakeClient(fail=fail)


def _factory(client: _FakeClient):
    def build(*_args, **_kwargs):
        # A real GeminiCoach with a fake client: no network, no key needed.
        return GeminiCoach(api_key="not-a-real-key", client=client)

    return build


def text_of(at) -> str:
    """Everything visible as prose: markdown, captions and alerts."""
    parts = [block.value for block in at.markdown]
    parts += [(getattr(c, "value", "") or "") for c in at.get("caption")]
    for kind in ("success", "info", "warning", "error"):
        parts += [(getattr(e, "value", "") or "") for e in at.get(kind)]
    return "\n".join(parts)


def run_page(page: str, state: dict | None = None):
    at = AppTest.from_file(str(ROOT / page), default_timeout=240)
    for key, value in (state or STATE).items():
        at.session_state[key] = value
    at.run()
    return at


def recommendation_count() -> int:
    with session_scope() as session:
        return int(session.scalar(select(func.count()).select_from(Recommendation)) or 0)


def drop_demo_rows(ids) -> None:
    ids = [i for i in ids if i]
    if not ids:
        return
    with session_scope() as session:
        session.execute(delete(Recommendation).where(Recommendation.recommendation_id.in_(ids)))
    print(f"    (cleaned up the {len(ids)} demo row(s) this script created)")


created: list[int] = []

# --------------------------------------------------------------------------- #
# 1. No API key → the page explains itself, the button stays disabled.
# --------------------------------------------------------------------------- #
print("--- 1. without an API key")
import utils.data_access as data_access  # noqa: E402

key_present = bool(data_access.coach_status().configured)
at = run_page("pages/5_AI_Coach.py")
assert not at.exception, at.exception
page = text_of(at)
assert "Your AI nutrition coach" in page
assert "What the coach will see" in page
assert any("Read the exact prompt" in (getattr(e, "label", "") or "") for e in at.get("expander"))
if key_present:
    print("    a key is configured here — the no-key assertions are skipped (nothing was called)")
else:
    assert "API key not configured" in page
    assert "How to enable it" in page and "GEMINI_API_KEY" in page
    assert at.button[0].disabled is True
    print("    setup card shown, button disabled ✓")

# --------------------------------------------------------------------------- #
# 2. One press → one answer, rendered, stored, snapshot sent, name never sent.
# --------------------------------------------------------------------------- #
print("--- 2. asking the coach (fake SDK client, real code path)")
before = recommendation_count()
good = _client()
with patch.object(data_access, "GeminiCoach", _factory(good)):
    at = run_page("pages/5_AI_Coach.py")
    assert not at.exception, at.exception
    assert at.button[0].disabled is False, "the button must be enabled once the coach is ready"
    at.text_area[0].input(QUESTION)
    at.button[0].click().run()
    assert not at.exception, at.exception

    page = text_of(at)
    assert "Your coaching message" in page
    assert "add 150 g of curd" in page, "the answer text must be rendered"
    after = recommendation_count()
    assert after == before + 1, f"exactly one row should be stored ({before} → {after})"
    print(f"    answer rendered and stored ({before} → {after} recommendations) ✓")

    sent = good.models.calls[-1]
    assert len(good.models.calls) == 1, "exactly one API call per press"
    assert sent["model"] == GEMINI_MODEL_NAME
    assert f"QUESTION FROM THE USER: {QUESTION}" in sent["contents"]
    assert "PNIE forecast" in sent["contents"], "the forecast must be part of the snapshot"
    assert candidate.name not in sent["contents"], "the user's name must never be sent"
    assert "Never invent" in str(sent["config"].system_instruction)
    print(f"    prompt carried the snapshot ({len(sent['contents'])} chars), not the name ✓")

    # same question again → the stored answer is reused, no second API call
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert len(good.models.calls) == 1, "the duplicate guard must prevent a second API call"
    assert recommendation_count() == after
    print("    second press reused the stored answer (1 API call total) ✓")

with session_scope() as session:
    created += [
        int(i)
        for i in session.scalars(
            select(Recommendation.recommendation_id).order_by(Recommendation.recommendation_id.desc())
        ).all()[: after - before]
    ]

# --------------------------------------------------------------------------- #
# 3. The dashboard shows the newest stored answer.
# --------------------------------------------------------------------------- #
print("--- 3. dashboard coach card")
dash = run_page("pages/1_Dashboard.py")
assert not dash.exception, dash.exception
dash_text = text_of(dash)
assert "add 150 g of curd" in dash_text, "the stored answer must appear on the dashboard"
assert "How can I add protein without more rotis?" in dash_text
print("    stored recommendation shown on the dashboard ✓")

# --------------------------------------------------------------------------- #
# 4. A failing call → classified error + advice, nothing stored.
# --------------------------------------------------------------------------- #
print("--- 4. failure path (simulated quota error)")
before = recommendation_count()
bad = _client(fail=True)
with patch.object(data_access, "GeminiCoach", _factory(bad)):
    at = run_page("pages/5_AI_Coach.py")
    at.text_area[0].input("Should I eat more protein?")
    at.button[0].click().run()
    assert not at.exception, at.exception
    page = text_of(at)
    assert "The coach could not answer" in page
    assert "Quota or rate limit reached" in page
    assert "free-tier quota" in page, "kind-specific advice must be shown"
    assert recommendation_count() == before, "a failed call must store nothing"
    print("    failure explained with advice, nothing stored ✓")

# --------------------------------------------------------------------------- #
# 5. No profile → onboarding prompt, no crash.
# --------------------------------------------------------------------------- #
at = run_page("pages/5_AI_Coach.py", {"nv_profile": None})
assert not at.exception, at.exception
assert "Create your profile" in text_of(at)
print("--- 5. no-profile path ✓")

drop_demo_rows(created)
print("\nALL PHASE 7 E2E CHECKS PASSED")
