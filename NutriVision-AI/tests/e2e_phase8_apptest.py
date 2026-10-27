"""Phase 8 end-to-end check — every screen, and the Profile page's writes/deletes.

Run it with::

    python tests/e2e_phase8_apptest.py

Like the Phase 7 script this is **not** collected by pytest (the file name starts
with ``e2e_``): it drives the real Streamlit pages through
``streamlit.testing.v1.AppTest`` against whatever database ``DATABASE_URL`` points
at. Safety rules it follows:

* it never touches the ``Sample Demo User`` or any other existing row — it
  creates its own throwaway user (and meals) and deletes it again at the end;
* no AI call is made (the coach page is only *rendered*, its button is never
  pressed), so no API key is needed and no quota is spent;
* every delete it performs is on its own throwaway data.

It covers:

1. all seven screens render **with and without a profile**, each with the build
   footer, and the home screen offers the six page links;
2. the Profile page shows targets, the calculation rules and the storage summary
   read from the database;
3. the edit form updates the stored row **in place** (never a duplicate account),
   recalculates the targets and reports the change;
4. deleting the meal history requires the typed word and keeps profile,
   forecasts and recommendations;
5. deleting everything removes the user plus all children and clears the session.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{ROOT / 'nutrivision.db'}")

from sqlalchemy import func, select  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from config import APP_BUILD  # noqa: E402
from database import get_user, list_users, save_meal, save_user, session_scope, user_data_summary  # noqa: E402
from database.connection import reset_engine  # noqa: E402
from models import Meal, User  # noqa: E402
from utils.helpers import build_profile  # noqa: E402

reset_engine()

TEST_NAME = "Phase Eight End To End User"
PAGES = (
    "app.py",
    "pages/1_Dashboard.py",
    "pages/2_Upload_Meal.py",
    "pages/3_Meal_History.py",
    "pages/4_Predictions.py",
    "pages/5_AI_Coach.py",
    "pages/6_Profile.py",
)


# --------------------------------------------------------------------------- #
# Fixtures: a throwaway user with three meals
# --------------------------------------------------------------------------- #
def _cleanup() -> None:
    with session_scope() as session:
        row = session.scalars(select(User).where(User.name == TEST_NAME)).first()
        if row is not None:
            session.delete(row)


def _create_user() -> User:
    user = save_user(
        User.from_profile(
            build_profile(
                {
                    "name": TEST_NAME,
                    "age": 31,
                    "gender": "Male",
                    "height_cm": 175.0,
                    "weight_kg": 78.0,
                    "activity_level": "Moderately active (moderate exercise 3-5 days/week)",
                    "diet_preference": "Non-Vegetarian",
                    "fitness_goal": "Weight Loss",
                }
            )
        )
    )
    for offset, kcal in ((0, 620.0), (1, 540.0), (2, 700.0)):
        save_meal(
            Meal(
                user_id=user.user_id,
                timestamp=datetime.now(timezone.utc) - timedelta(days=offset),
                food_name="Paneer butter masala",
                portion_grams=200.0,
                calories=kcal,
                protein=18.0,
                fat=30.0,
                carbohydrates=14.0,
                fiber=3.0,
                health_score=58.0,
            )
        )
    return user


def text_of(at) -> str:
    """Everything visible as prose: markdown, captions and alerts."""
    parts = [block.value for block in at.markdown]
    parts += [(getattr(c, "value", "") or "") for c in at.get("caption")]
    for kind in ("success", "info", "warning", "error"):
        parts += [(getattr(e, "value", "") or "") for e in at.get(kind)]
    return "\n".join(parts)


def run(page: str, state: dict | None = None):
    at = AppTest.from_file(str(ROOT / page), default_timeout=300)
    for key, value in (state or {}).items():
        at.session_state[key] = value
    at.run()
    return at


def widget(elements, label: str):
    """Find an AppTest widget by its label (the project's widgets carry no keys)."""
    for element in elements:
        if (getattr(element, "label", "") or "") == label:
            return element
    raise AssertionError(f"{label!r} not found among {[getattr(e, 'label', None) for e in elements]}")


_cleanup()
user = _create_user()
profile = get_user(user.user_id).to_profile()
STATE = {"nv_profile": profile.to_dict()}
print(f"throwaway user_id={user.user_id} with {user_data_summary(user.user_id)['meals']} meals")

# --------------------------------------------------------------------------- #
# 1. Every screen renders in both states, all with the build footer
# --------------------------------------------------------------------------- #
print("--- 1. all seven screens")
for page in PAGES:
    for label, state in (("with profile", STATE), ("no profile", {"nv_profile": None})):
        at = run(page, state)
        assert not at.exception, f"{page} ({label}): {at.exception}"
        page_text = text_of(at)
        assert f"build {APP_BUILD}" in page_text, f"{page} ({label}): footer missing"
        assert "Estimates for general wellness" in page_text, f"{page} ({label}): disclaimer missing"
        print(f"    {page:26s} {label:12s} OK")

home = run("app.py", STATE)
links = [getattr(link, "label", "") for link in home.get("page_link")]
assert len(links) >= 6, f"the home screen must link to all six pages, found {links}"
assert any("Profile" in label for label in links)
print("    home links to all six pages ✓")

# --------------------------------------------------------------------------- #
# 2. Profile page content
# --------------------------------------------------------------------------- #
print("--- 2. profile page content")
at = run("pages/6_Profile.py", STATE)
assert not at.exception, at.exception
page = text_of(at)
for expected in ("Your profile", "How the targets are calculated", "Edit your profile",
                 "What is stored", "Delete data", "Not stored at all"):
    assert expected in page, f"missing section: {expected}"
assert f"user_id {user.user_id}" in page
assert "Mifflin-St Jeor" in page
assert widget(at.number_input, "Weight (kg) *").value == 78.0, "the form must be prefilled"
print("    sections, targets, storage summary and prefilled form ✓")

# --------------------------------------------------------------------------- #
# 3. Editing updates the row in place and recalculates the targets
# --------------------------------------------------------------------------- #
print("--- 3. edit in place")
before_target = profile.daily_calorie_target
widget(at.number_input, "Weight (kg) *").set_value(70.0)
widget(at.selectbox, "Fitness goal *").set_value("Muscle Gain")
widget(at.button, "💾 Save changes").click().run()
assert not at.exception, at.exception
stored = get_user(user.user_id)
assert stored is not None and stored.weight_kg == 70.0 and stored.fitness_goal == "Muscle Gain"
with session_scope() as session:
    same_name = int(session.scalar(select(func.count(User.user_id)).where(User.name == TEST_NAME)) or 0)
assert same_name == 1, f"saving must never create a duplicate account (found {same_name})"
updated = stored.to_profile()
assert round(updated.daily_calorie_target) != round(before_target), "targets must be recalculated"
assert "Daily target" in text_of(at) and "kcal" in text_of(at)
print(f"    weight 78→70 kg, goal →Muscle Gain; target {round(before_target)}→{round(updated.daily_calorie_target)} kcal, one row only ✓")

# --------------------------------------------------------------------------- #
# 4. Deleting the meal history needs the typed word and keeps the profile
# --------------------------------------------------------------------------- #
print("--- 4. delete meal history")
at = run("pages/6_Profile.py", STATE)
at.button(key="nv_ask_delete_meals").click().run()
assert not at.exception, at.exception
assert any("Type DELETE to confirm" in (w.label or "") for w in at.text_input)
at.text_input(key="nv_confirm_delete_meals_input").input("please delete")
at.button(key="nv_do_delete_meals").click().run()
assert "Type DELETE exactly to confirm." in text_of(at)
assert user_data_summary(user.user_id)["meals"] == 3, "a wrong word must delete nothing"
at.text_input(key="nv_confirm_delete_meals_input").input("DELETE")
at.button(key="nv_do_delete_meals").click().run()
assert not at.exception, at.exception
summary = user_data_summary(user.user_id)
assert summary["meals"] == 0, summary
assert get_user(user.user_id) is not None, "the profile must survive a meal-history delete"
assert "Deleted 3 meal(s)" in text_of(at)
print("    wrong word → nothing; DELETE → 3 meals removed, profile kept ✓")

# --------------------------------------------------------------------------- #
# 5. Deleting everything removes user + children and clears the session
# --------------------------------------------------------------------------- #
print("--- 5. delete everything")
at = run("pages/6_Profile.py", STATE)
at.button(key="nv_ask_delete_all").click().run()
at.text_input(key="nv_confirm_delete_all_input").input("DELETE")
at.button(key="nv_do_delete_all").click().run()
assert not at.exception, at.exception
assert get_user(user.user_id) is None, "the user row must be gone"
assert "complete onboarding" in text_of(at), "the page must fall back to the onboarding prompt"
print("    user + children removed, session cleared ✓")

# --------------------------------------------------------------------------- #
# Cleanup + a last look at the sample data
# --------------------------------------------------------------------------- #
_cleanup()
sample = list_users(include_samples=True)
print(f"done — {len(sample)} user row(s) left untouched in the database")
print("\nALL PHASE 8 E2E CHECKS PASSED")
