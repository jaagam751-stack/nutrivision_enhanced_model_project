"""Create (or verify) the NutriVision AI database schema (Phase 2).

Usage
-----
Verify the connection and schema (read-only, safe to run any time):

    python database/init_db.py --check

Create the four tables (``users``, ``meals``, ``predictions``, ``recommendations``):

    python database/init_db.py

Create the tables in the local SQLite development fallback instead of Neon:

    python database/init_db.py --sqlite

Insert a clearly-labelled demo dataset for UI previews (sample data only):

    python database/init_db.py --sample-data

Drop and recreate every table (destroys all rows, asks for confirmation):

    python database/init_db.py --reset

Notes
-----
* ``create_all`` only creates tables that do not exist; it never alters existing
  ones. Changing a column later therefore needs a migration (documented in
  README). Keep this module in step with ``models/``.
* ``DATABASE_URL`` is read from the environment / ``.env``. ``--sqlite`` overrides
  it for the current process only, and prints a warning saying so.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Allow `python database/init_db.py` from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ACTIVITY_LEVELS, DIET_PREFERENCES, FITNESS_GOALS  # noqa: E402
from database.connection import (  # noqa: E402
    SQLITE_URL,
    DatabaseNotConfiguredError,
    check_database,
    describe_database,
    get_database_url,
    get_engine,
    reset_engine,
)
from models import Meal, Prediction, Recommendation, User  # noqa: E402,F401 (registers tables)
from models.base import Base  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nutrivision.init_db")


# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #
def create_schema(verbose: bool = True) -> bool:
    """Create any missing tables. Returns ``True`` on success."""
    try:
        engine = get_engine()
    except DatabaseNotConfiguredError as exc:
        logger.error("%s", exc)
        return False

    try:
        Base.metadata.create_all(engine)
        if verbose:
            logger.info("Schema ready. Tables in metadata: %s", ", ".join(sorted(Base.metadata.tables)))
        return True
    except Exception as exc:  # noqa: BLE001 - surface any driver/network error plainly
        logger.error("Schema creation failed: %s: %s", exc.__class__.__name__, exc)
        return False


def drop_schema() -> bool:
    """Drop every NutriVision table."""
    try:
        Base.metadata.drop_all(get_engine())
        logger.warning("All NutriVision tables dropped.")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Drop failed: %s: %s", exc.__class__.__name__, exc)
        return False


def print_status() -> bool:
    """Print a readable connection + schema report. Returns ``True`` if schema is complete."""
    status = check_database()
    print()
    print("NutriVision AI — database status")
    print("-" * 46)
    print(f"backend          : {status.label} ({status.backend})")
    if status.backend != "not configured":
        masked = describe_database(get_database_url())["location"] if not status.location.startswith("invalid") else status.location
        print(f"target           : {masked}")
    print(f"tables found     : {', '.join(status.tables) if status.tables else '—'}")
    print(f"tables missing   : {', '.join(status.missing_tables) if status.missing_tables else 'none'}")
    if status.error:
        print(f"error            : {status.error}")
    print("-" * 46)
    print("RESULT           : " + ("schema is complete ✅" if status.ok else "schema incomplete ⚠️"))
    print()
    return status.ok


def insert_sample_data() -> None:
    """Insert a clearly-labelled demo user with 84 days (12 weeks) of sample meals.

    This data is for UI previews only. Rows are flagged with
    ``users.is_sample = True`` so they stay strictly separated from real user
    records, and the function is idempotent (it deletes its own previous sample
    rows first).

    The window spans twelve weeks on purpose: the PNIE (Phase 6) needs at least
    ``PNIE_MIN_TRAINING_ROWS`` weekly windows before it will train a model, so a
    one-month demo could never show a real forecast. A few days are deliberately
    left empty, exactly like a real user who forgets to log one day.
    """
    from database.repository import delete_sample_data, save_meals_bulk, save_user

    deleted = delete_sample_data()
    if deleted:
        logger.info("Removed %s previous sample row group(s).", deleted)

    user = User.from_profile(_sample_profile())
    user.is_sample = True          # marks these rows for --sample-data cleanup
    user = save_user(user)
    logger.info("Sample user created with user_id=%s", user.user_id)

    rows = []
    base_day = datetime.now(timezone.utc).date() - timedelta(days=_SAMPLE_DAYS - 1)
    for day_index in range(_SAMPLE_DAYS):
        if day_index in _SAMPLE_SKIPPED_DAYS:
            continue                      # a day the user did not log
        template = _sample_day_template(day_index)
        scale = _sample_day_scale(day_index)
        step = (_SAMPLE_LAST_MEAL_HOUR - _SAMPLE_FIRST_MEAL_HOUR) / max(len(template) - 1, 1)
        for meal_offset, (food, grams) in enumerate(template):
            grams = max(20, int(round(grams * scale)))
            hour_offset = _SAMPLE_FIRST_MEAL_HOUR + meal_offset * step
            calories_per_100g = _SAMPLE_FOODS[food]
            factor = grams / 100.0
            rows.append(
                Meal(
                    user_id=user.user_id,
                    timestamp=datetime.combine(
                        base_day + timedelta(days=day_index),
                        datetime.min.time(),
                        tzinfo=timezone.utc,
                    )
                    + timedelta(hours=hour_offset),
                    food_name=food,
                    portion_grams=float(grams),
                    calories=round(calories_per_100g[0] * factor, 1),
                    protein=round(calories_per_100g[1] * factor, 1),
                    fat=round(calories_per_100g[2] * factor, 1),
                    carbohydrates=round(calories_per_100g[3] * factor, 1),
                    fiber=round(calories_per_100g[4] * factor, 1),
                    health_score=float(calories_per_100g[5]),
                )
            )
    saved = save_meals_bulk(rows)
    logger.info("Inserted %s sample meal rows for user_id=%s.", saved, user.user_id)


# --------------------------------------------------------------------------- #
# Sample data (UI previews only — excluded from tests, prefixed [SAMPLE])
# --------------------------------------------------------------------------- #
_SAMPLE_NAME = "Sample Demo User"

#: Twelve weeks of history, so the Phase 6 engine can actually train a model
#: (10 usable weekly windows with the default 14-day look-back / 7-day horizon).
_SAMPLE_DAYS: int = 84
#: Days the demo user "forgot" to log (gaps must behave like gaps, not 0 kcal).
_SAMPLE_SKIPPED_DAYS: frozenset[int] = frozenset({9, 23, 41, 55, 62, 70})
#: Meals are spread evenly between these hours so every row stays inside its own
#: calendar day (a fixed 5-hour step used to push the last meals into tomorrow).
_SAMPLE_FIRST_MEAL_HOUR: float = 7.5      # 07:30
_SAMPLE_LAST_MEAL_HOUR: float = 21.5      # 21:30


def _sample_profile():
    from utils.helpers import build_profile

    return build_profile(
        {
            "name": _SAMPLE_NAME,
            "age": 29,
            "gender": "Female",
            "height_cm": 163.0,
            "weight_kg": 64.5,
            "activity_level": list(ACTIVITY_LEVELS)[2],
            "diet_preference": DIET_PREFERENCES[0],
            "fitness_goal": FITNESS_GOALS[0],
        }
    )


# food -> (kcal, protein, fat, carbs, fiber, health_score) per 100 g (rounded)
_SAMPLE_FOODS: dict[str, tuple[float, float, float, float, float, float]] = {
    "Chapati (whole wheat roti)": (297.0, 9.6, 7.1, 49.0, 6.1, 72.0),
    "Dal (cooked toor dal)": (116.0, 7.0, 0.7, 20.0, 4.5, 78.0),
    "Steamed white rice": (130.0, 2.7, 0.3, 28.0, 0.4, 55.0),
    "Paneer butter masala": (230.0, 8.0, 18.0, 9.0, 1.5, 48.0),
    "Mixed vegetable sabzi": (95.0, 2.5, 5.0, 10.0, 3.5, 80.0),
    "Boiled egg": (155.0, 13.0, 10.6, 1.1, 0.0, 74.0),
    "Banana": (89.0, 1.1, 0.3, 22.8, 2.6, 68.0),
    "Curd (plain dahi)": (60.0, 3.1, 3.3, 4.7, 0.0, 76.0),
    "Poha": (150.0, 3.0, 4.0, 26.0, 1.2, 62.0),
    "Idli": (120.0, 4.5, 0.6, 25.0, 1.0, 70.0),
    "Masala dosa": (155.0, 3.9, 5.9, 21.6, 1.8, 58.0),
    "Palak paneer": (190.0, 7.5, 14.0, 7.0, 3.0, 66.0),
    "Gulab jamun": (340.0, 4.0, 12.0, 55.0, 0.4, 22.0),
    "Grilled chicken breast": (165.0, 31.0, 3.6, 0.0, 0.0, 88.0),
}


def _sample_day_scale(day_index: int) -> float:
    """Deterministic portion scaling that gives the demo history a real trend.

    No randomness anywhere: re-running the script produces identical data, so
    charts, tests and screenshots stay reproducible. The shape is:

    * a slow decline (a weight-loss journey) across the ten weeks,
    * a partial recovery in the last three weeks, so the trend reverses and the
      PNIE forecast is not simply a flat continuation,
    * a mild weekly rhythm, so individual weeks are not clones of each other.
    """
    progress = day_index / max(_SAMPLE_DAYS - 1, 1)
    drift = 1.10 - 0.20 * progress                    # 1.10 -> 0.90
    recovery = 0.12 * max(0.0, progress - 0.70) / 0.30  # +12 % in the last 30 %
    rhythm = (0.00, 0.02, -0.02, 0.03, -0.03, 0.07, 0.03)[day_index % 7]
    return drift + recovery + rhythm


def _sample_day_template(day_index: int) -> list[tuple[str, int]]:
    """Four deterministic daily menus (~1,700-1,900 kcal) for the demo user.

    The menus are sized for the sample profile (29 y, female, 163 cm, 64.5 kg,
    moderately active, weight loss) whose daily target is about 1,680 kcal, so
    the demo shows a realistic intake instead of a suspiciously tiny one. The
    fourth day adds a non-vegetarian option; the rotation (4 menus × 7 day
    rhythm) repeats every 28 days.
    """
    rotation = [
        # Day A — poha + dosa breakfast, chapati/dal/sabzi dinner
        [
            ("Poha", 250),
            ("Masala dosa", 180),
            ("Chapati (whole wheat roti)", 180),
            ("Dal (cooked toor dal)", 250),
            ("Mixed vegetable sabzi", 200),
            ("Curd (plain dahi)", 200),
            ("Banana", 120),
        ],
        # Day B — South Indian breakfast, rice + palak paneer + a sweet
        [
            ("Idli", 240),
            ("Boiled egg", 100),
            ("Steamed white rice", 250),
            ("Palak paneer", 200),
            ("Chapati (whole wheat roti)", 60),
            ("Mixed vegetable sabzi", 150),
            ("Curd (plain dahi)", 150),
            ("Banana", 110),
            ("Gulab jamun", 60),
        ],
        # Day C — dosa + poha day, paneer butter masala dinner
        [
            ("Masala dosa", 250),
            ("Poha", 200),
            ("Chapati (whole wheat roti)", 120),
            ("Paneer butter masala", 180),
            ("Mixed vegetable sabzi", 150),
            ("Dal (cooked toor dal)", 200),
        ],
        # Day D — the non-vegetarian day: grilled chicken instead of paneer
        [
            ("Boiled egg", 100),
            ("Chapati (whole wheat roti)", 120),
            ("Grilled chicken breast", 180),
            ("Steamed white rice", 200),
            ("Palak paneer", 150),
            ("Curd (plain dahi)", 150),
            ("Banana", 110),
            ("Gulab jamun", 50),
        ],
    ]
    return rotation[day_index % len(rotation)]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create or verify the NutriVision AI database schema.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="only verify connection and tables")
    parser.add_argument("--sqlite", action="store_true", help="use the local SQLite fallback")
    parser.add_argument("--sample-data", action="store_true", help="insert flagged sample/demo rows")
    parser.add_argument("--reset", action="store_true", help="drop all tables before creating them")
    parser.add_argument("--yes", action="store_true", help="skip the --reset confirmation prompt")
    args = parser.parse_args(argv)

    if args.sqlite:
        import os

        os.environ["DATABASE_URL"] = SQLITE_URL
        print(f"⚠️  Using the SQLite development fallback for this run: {SQLITE_URL}")

    if args.check:
        return 0 if print_status() else 1

    try:
        print_status()
    except DatabaseNotConfiguredError as exc:
        logger.error("%s", exc)
        return 1

    if args.reset:
        if not args.yes:
            answer = input("This DELETES all rows in every NutriVision table. Type 'yes' to continue: ")
            if answer.strip().lower() != "yes":
                print("Aborted.")
                return 1
        drop_schema()
        reset_engine()

    if not create_schema():
        return 1

    if args.sample_data:
        insert_sample_data()

    return 0 if print_status() else 1


if __name__ == "__main__":
    raise SystemExit(main())
