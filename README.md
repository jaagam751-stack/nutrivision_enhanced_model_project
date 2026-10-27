# 🥗 NutriVision AI

**AI-Powered Food Recognition & Predictive Nutrition Analysis**

A final-year / NSP project that turns a food photo into nutrition insight: recognise the food,
estimate calories and macronutrients, track meal history, forecast the next week's calorie trend
with XGBoost, and get personalised coaching from Gemini — all inside a polished Streamlit app.

> **Current status: Phases 1–8 complete — the project is finished.**
> The Streamlit shell, onboarding and validated profile handling are built, the profile is
> persisted in Neon PostgreSQL through SQLAlchemy (with the `meals`, `predictions` and
> `recommendations` tables ready for later phases), and the **Upload Meal page recognises food
> photos** with a pretrained Food-101 model — reporting the food name, a confidence score, the
> top-5 shortlist and an explicit "confirm this yourself" warning when the model is unsure.
> **Phase 4 adds the nutrition engine**: 134 Indian foods with per-100 g values, documented
> matching rules that ask you to confirm uncertain dishes, portion-size scaling, daily-target
> context from your profile, and a transparent rule-based Nutrition Health Score (0–100) with a
> point-by-point breakdown. **Phase 5 adds the explicit Save Meal flow** (two duplicate guards,
> nothing is ever stored automatically), the Plotly dashboard and the filterable meal history with
> CSV export. **Phase 6 adds the PNIE**: one XGBoost regressor per user forecasts next week's
> average daily calories from the saved meals, evaluated on a chronological hold-out, with
> descriptive-statistics fallbacks instead of fabricated numbers when history is short. **Phase 7
> adds the Gemini 2.5 Flash AI coach**: it shows the exact snapshot it will send (profile, meal
> averages, forecast), makes one request per button press, stores every answer in the
> `recommendations` table, and reports each failure — bad key, quota, timeout, network, blocked
> answer — with what to do about it instead of substituting canned advice. **Phase 8 finishes the
> project**: a *Profile & data* page (targets with their formulas, edit-in-place, typed-confirmation
> deletes), one hub and one footer across all seven screens, and a deployment guide for Streamlit
> Community Cloud (§15).

---

## 1. What is built so far

| Item | Status |
| --- | --- |
| Project skeleton, dark-themed Streamlit shell, custom CSS | ✅ Phase 1 |
| Onboarding form + full server-side validation | ✅ Phase 1 |
| Derived values: BMI (+Asian-Indian band), BMR, TDEE, goal-adjusted calorie target, protein & fibre targets | ✅ Phase 1 |
| Reusable UI components, session-state helpers, logging | ✅ Phase 1 |
| **SQLAlchemy ORM: `users`, `meals`, `predictions`, `recommendations`** | ✅ **Phase 2** |
| **Neon PostgreSQL connection handling, secrets loading, health check** | ✅ **Phase 2** |
| **Schema CLI (`init_db.py`): create / verify / reset / sample data** | ✅ **Phase 2** |
| **Repository layer (`repository.py`): all reads and writes, injection-safe, graceful degradation** | ✅ **Phase 2** |
| **Profile persistence, "continue as" restore, update-in-place, delete stored copy** | ✅ **Phase 2** |
| **Upload Meal page: JPG/JPEG/PNG validation, preview, Analyze Meal button** | ✅ **Phase 3** |
| **Pretrained food classification (MobileNetV2 / Food-101, 101 classes) with confidence + top-5** | ✅ **Phase 3** |
| **Model caching, sanity checking, honest failure handling (no fabricated predictions)** | ✅ **Phase 3** |
| **Indian food dataset: 134 foods, per-100 g values, diet type, serving sizes** | ✅ **Phase 4** |
| **Documented matching rules (exact → alias → model label → partial → fuzzy) with user confirmation** | ✅ **Phase 4** |
| **Portion scaling, daily-target context, transparent 0–100 Health Score with full breakdown** | ✅ **Phase 4** |
| **Explicit Save Meal button (never automatic) with session lock + database duplicate guard** | ✅ **Phase 5** |
| **Dashboard: today's totals vs targets, Plotly trend/macro/health-score/weekly charts, recent meals** | ✅ **Phase 5** |
| **Meal History: date + food filters, CSV export, confirmed single-meal deletion** | ✅ **Phase 5** |
| **PNIE: one XGBoost regressor predicting next week's average daily calories** | ✅ **Phase 6** |
| **Weekly-window feature engineering with a leakage test (features never see the future)** | ✅ **Phase 6** |
| **Chronological hold-out evaluation (MAE/RMSE) + baseline comparison, honest modes when history is short** | ✅ **Phase 6** |
| **Predictions page: forecast, error band, feature importances, descriptive trends, stored forecasts** | ✅ **Phase 6** |
| **Gemini 2.5 Flash coach with a strictly factual snapshot (no invented numbers)** | ✅ **Phase 7** |
| **Transparent UI: the exact prompt is readable, and only one API call happens per button press** | ✅ **Phase 7** |
| **Stored recommendations + dashboard card, with every failure classified and explained** | ✅ **Phase 7** |
| **Profile & data page: targets with their formulas, edit-in-place, typed-confirmation deletes** | ✅ **Phase 8** |
| **One hub, one footer, one consistent look across all seven screens** | ✅ **Phase 8** |
| **Deployment guide + checklist for Streamlit Community Cloud and Neon** | ✅ **Phase 8** |

**No feature is presented as working before it is implemented.** Meals are never written to the
database in Phase 2. Nothing is written until you press **Save Meal** (Phase 5) — see §7.

---

## 2. Project structure

```
NutriVision-AI/
├── app.py                      # Onboarding, home hub (six page links), database panel
├── config.py                   # Options, ranges, health maths constants, secrets access
├── requirements.txt
├── README.md
├── .env.example                # DATABASE_URL / GEMINI_API_KEY templates
├── .gitignore                  # Blocks .env, secrets, DB files, model weights
├── .streamlit/config.toml      # Dark theme + server settings
├── assets/style.css            # Custom dark theme
├── database/
│   ├── __init__.py             # Public data-access API
│   ├── connection.py           # Engine, session factory, health check, URL handling
│   ├── init_db.py              # CLI: create / check / reset schema, sample data
│   └── repository.py           # Every query in the project lives here
├── models/
│   ├── __init__.py             # ALL_MODELS registry
│   ├── base.py                 # DeclarativeBase + utcnow
│   ├── user.py                 # users table
│   ├── meal.py                 # meals table
│   ├── prediction.py           # predictions table
│   └── recommendation.py       # recommendations table
├── utils/
│   ├── helpers.py              # UserProfile, validation, health maths, session state
│   ├── analytics.py            # Phase 5: pure aggregation + Plotly chart builders
│   ├── data_access.py          # Phases 5-8: cached DB reads + one function per write action
│   ├── ui.py                   # Reusable Streamlit components
│   └── logging_config.py
├── tests/
│   ├── conftest.py             # shared fixtures (temp SQLite DB, user factory)
│   ├── test_profile.py         # 36 tests (Phase 1)
│   ├── test_database.py        # 29 tests (Phase 2, runs on temp SQLite)
│   ├── test_food_recognition.py# 31 tests (Phase 3, offline via a stub model)
│   ├── test_nutrition_engine.py# 33 tests (Phase 4: dataset, matching, scaling)
│   ├── test_health_score.py    # 32 tests (Phase 4: rubric behaviour)
│   ├── test_analytics.py       # 22 tests (Phase 5: aggregation, charts, duplicate guard)
│   ├── test_pnie.py            # 53 tests (Phase 6: features, leakage, model, engine, persistence)
│   ├── test_gemini_coach.py    # 41 tests (Phase 7: snapshot, prompt, failures, storage, guards)
│   ├── test_profile_data.py    # 23 tests (Phase 8: summary, persist-in-place, deletes)
│   ├── e2e_phase7_apptest.py   # page-level AppTest run (not collected by pytest, run it directly)
│   └── e2e_phase8_apptest.py   # seven screens + profile writes/deletes (run it directly)
├── food_recognition/           # Phase 3 — food image recognition
│   ├── __init__.py             #   public API (FoodPredictor, errors, helpers)
│   ├── preprocessing.py        #   upload validation + 224×224 ImageNet preprocessing
│   ├── model_loader.py         #   ONNX / PyTorch backends, caching, sanity check
│   ├── predictor.py            #   prediction, label formatting, confidence logic
│   ├── export_onnx.py          #   optional: checkpoint → verified ONNX export
│   └── food101_classes.json    #   the 101 class names in model output order
├── assets/models/              # shipped ONNX model (9.4 MB FP32, 2.5 MB INT8)
├── nutrition/                  # Phase 4 — 134-food Indian dataset + engine + health score
├── pnie/                       # Phase 6 — predictive engine
│   ├── __init__.py             #   public API (run_pnie, PNIEEngine, PNIEPredictor, ...)
│   ├── feature_engineering.py  #   daily series → weekly training rows (leakage-free)
│   ├── trend_analysis.py       #   descriptive statistics, slope, direction (no ML)
│   ├── predictor.py            #   the single XGBoost regressor + chronological evaluation
│   └── pnie_engine.py          #   orchestration, modes, guards, DB metadata
├── ai/                         # Phase 7 — Gemini 2.5 Flash coach
│   ├── __init__.py             #   public API + the ERROR_* vocabulary
│   └── gemini_coach.py         #   snapshot builder, SDK wrapper, failure classifier
└── pages/
    ├── 1_Dashboard.py          # Phase 5 (Phase 6 forecast card, Phase 7 coach card)
    ├── 2_Upload_Meal.py        # Phase 3–5
    ├── 3_Meal_History.py       # Phase 5
    ├── 4_Predictions.py        # Phase 6 — forecast, explanation, stored forecasts
    ├── 5_AI_Coach.py           # Phase 7 — snapshot, one-call coach, stored recommendations
    └── 6_Profile.py            # Phase 8 — targets, edit in place, storage summary, deletes
```

---

## 3. Quick start

### 3.1 Install

```bash
cd NutriVision-AI
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3.2 Create the database (free, ~2 minutes)

**Option A — Neon PostgreSQL (the project database, required for deployment)**

1. Sign up at **<https://neon.tech>** (free tier) and create a project.
2. Open **Dashboard → Connect** and copy the connection string, which looks like:
   `postgresql://USER:PASSWORD@ep-cool-name-12345.region.aws.neon.tech/neondb?sslmode=require`
3. Copy `.env.example` to `.env` and paste it in:

```bash
cp .env.example .env
```

```dotenv
DATABASE_URL=postgresql+psycopg2://USER:PASSWORD@ep-cool-name-12345.region.aws.neon.tech/neondb?sslmode=require
```

`postgres://` and `postgresql://` URLs are accepted as-is and rewritten to use psycopg2.

**Option B — SQLite development fallback (no account needed)**

For UI work before you have Neon. This is **opt-in and never silent**: the app shows the active
backend in the sidebar, and the fallback is unsuitable for Streamlit Community Cloud (ephemeral
filesystem).

```bash
python database/init_db.py --sqlite --sample-data   # creates nutrivision.db + demo rows
```

Keep `DATABASE_URL` unset while you do this, or set `DATABASE_URL=sqlite:///nutrivision.db`.
In the running app you can also press **Use SQLite for this session** on the setup panel, and
**↩️ Switch back to PostgreSQL** afterwards.

### 3.3 Create the tables

```bash
python database/init_db.py --check    # verify connection + which tables exist (read-only)
python database/init_db.py            # create the four tables in Neon
```

Other commands:

```bash
python database/init_db.py --sample-data   # insert flagged sample rows for previews
python database/init_db.py --reset --yes   # DROP and recreate every table (destroys data)
```

Optional demo rows: `--sample-data` creates one user named `Sample Demo User` with **84 days
(12 weeks) of meals** — long enough for the Phase 6 engine to train a real model — flagged with
`users.is_sample = True`. The data is deterministic (no randomness), includes six unlogged days and
a portion drift that reverses in the last three weeks, so trends and forecasts are meaningful. Sample rows are hidden from the "continue as" list and are
removed by re-running the command.

### 3.4 Run

```bash
streamlit run app.py
```

Open <http://localhost:8501>:

1. Fill the onboarding form and press **Create profile**.
2. The sidebar shows **Database: online (Neon PostgreSQL)**, and the home screen shows a
   *Saved profile* card with your `user_id` and live row counts.
3. Open a new browser tab/reload and use **Continue as → Load … data** to restore the profile from
   the database (Phase 2 has no authentication yet, see limitations).
4. Edit your profile and save again: the existing row is **updated in place** (verified: no
   duplicate users), and **Remove saved copy from the database** deletes it while keeping your
   in-session profile.

### 3.5 Tests

```bash
pytest -v          # 300 tests: validation, maths, schema, repository, recognition, nutrition,
                   #            analytics, PNIE (features, leakage, model, engine, persistence),
                   #            Gemini coach (snapshot, prompt, failures, storage, guards),
                   #            profile data (summary, edit-in-place, deletes)
```

The database tests use a temporary SQLite file, so they never touch your Neon data. The
recognition tests generate their images in memory and use a stub model, so they run offline. The
PNIE tests build synthetic histories (and exercise the real XGBoost model), so they need no database
either. The coach tests inject a fake SDK client and never touch the network, so **no API key is
needed to run the suite** — it finishes in about 6 seconds.

The pages themselves are covered by a script that drives the real Streamlit app through
`streamlit.testing.v1.AppTest`. It is **not** part of the pytest suite (it needs a database with a
profile and meals, and it writes one demo row which it deletes again):

```bash
python database/init_db.py --sqlite --sample-data   # if you have no data yet
python tests/e2e_phase7_apptest.py                  # AI Coach page + dashboard card, offline
python tests/e2e_phase8_apptest.py                  # seven screens + profile edits/deletes
```

Both scripts create and remove their own throwaway rows (the Phase 8 one never touches the sample
user), so they are safe to re-run against a demo database.

---

## 4. Database schema (Phase 2)

```
users                                  meals
------------------------------------   ------------------------------------------
user_id        PK, autoincrement       meal_id       PK, autoincrement
name           VARCHAR(60)             user_id       FK → users.user_id  ON DELETE CASCADE
age            INTEGER  (>0)           timestamp     timestamptz  (indexed)
gender         VARCHAR(20)             food_name     VARCHAR(120)
height_cm      FLOAT    (>0)           portion_grams FLOAT (>0)
weight_kg      FLOAT    (>0)           calories, protein, fat,
activity_level VARCHAR(80)             carbohydrates, fiber   FLOAT (>=0)
diet_preference VARCHAR(40)            health_score  FLOAT (0..100)
fitness_goal   VARCHAR(40)             INDEX (user_id, timestamp)
created_at     timestamptz
is_sample      BOOLEAN                 predictions
                                       ------------------------------------------
recommendations                        prediction_id PK
------------------------------------   user_id       FK → users  ON DELETE CASCADE
recommendation_id PK                   prediction_date timestamptz (indexed)
user_id        FK → users ON DELETE CASCADE   predicted_next_week_avg_calories FLOAT
timestamp      timestamptz (indexed)   trend_direction, trend_change_kcal
recommendation_text TEXT               model_name, model_version, training_samples, mae
source         VARCHAR(40)             feature_metadata JSON
prompt         TEXT
```

Notes

* `health_score` is constrained to 0–100; nutrition columns cannot be negative; `portion_grams > 0`.
* Relationships: `User.meals`, `User.predictions`, `User.recommendations` (with
  `delete-orphan` + `ON DELETE CASCADE`), and each child has a `user` back-reference loaded with
  `lazy="selectin"` so it stays usable after the query session closes.
* `Meal.calories_per_100g` is a convenience property for dashboard tooltips.
* `Predictions` stores the *future* forecast only. The recent-history baseline used for comparison
  is recomputed from `meals` in Phase 6 and stored in `feature_metadata`, so it never goes stale.
* **Schema changes:** `create_all()` only creates missing tables; it never alters existing ones.
  For a column change during development use `init_db.py --reset` (drops data), or add Alembic
  (listed as optional in `requirements.txt`) once you are past prototyping.

---

## 5. Food image recognition (Phase 3)

### 5.1 The model, and why this one

| | |
| --- | --- |
| Model | `AlexKoff88/mobilenet_v2_food101` — torchvision **MobileNetV2** fine-tuned on **Food-101** |
| Classes | 101 dishes (pizza, samosa, chicken curry, sushi, dosa-style items such as *spring rolls*, …) |
| Size | 19 MB checkpoint · **9.4 MB ONNX (FP32)** · **2.5 MB ONNX (INT8)** — both shipped in `assets/models/` |
| Speed | ~5–15 ms per image on 2 CPU cores; model loads in ~0.05 s from ONNX |
| Licence | Apache-2.0 |
| Reported accuracy | 76.3 % top-1 on the Food-101 test split (model card) |

Why it was chosen:

* **It is food-specific.** A generic ImageNet model only knows ~10 food classes and cannot tell a
  samosa from a spring roll; Food-101 is a real food taxonomy with Indian dishes included
  (`samosa`, `chicken_curry`, …).
* **It is small and CPU-only.** A ViT-base Food-101 model is ~340 MB and needs ≥1 GB RAM — too
  heavy for a 1 GB Streamlit Community Cloud container. MobileNetV2 leaves plenty of headroom.
* **It runs offline once cached**, and the ONNX export means the app needs only `onnxruntime`
  (~15 MB) — no multi-gigabyte PyTorch install.

> **A real finding, documented instead of hidden.** The pre-exported
> `mobilenet_v2_food101.onnx` file on the Hub is **broken**: it returns the *same all-zero logits
> for every input, including random noise*. It was detected during development, so
> `model_loader.py` now runs a **sanity check on every load** (a real inference on a deterministic
> synthetic image that verifies the output is a finite, non-degenerate probability distribution
> summing to 1). A broken artifact is rejected with an explicit error and the loader falls through
> to the next backend — it can never silently produce meaningless "predictions". The shipped ONNX
> files were exported from the working PyTorch checkpoint by `food_recognition/export_onnx.py` and
> verified against PyTorch (max probability difference ≈ 1e-7 for FP32).

### 5.2 How a request flows

```
upload (JPG/JPEG/PNG, ≤10 MB)
  → preprocessing.validate_upload()      extension + size + real decode + format sniff
  → preprocessing.load_image()           EXIF orientation applied, converted to RGB
  → preprocessing.preprocess()           256 px short side → centre-crop 224 → /255 → ImageNet stats → NCHW
  → model_loader.load_food_model()       cached per server (st.cache_resource), sanity-checked once
  → predictor.FoodPredictor.predict()    softmax → top-1 + top-5 + margin → confidence tier
  → Upload Meal page                     preview, result cards, ranking bars, uncertainty warning
```

Result caching: the prediction is stored in Streamlit session state keyed by a content hash of the
file, so normal Streamlit reruns (button presses, widget changes) never re-run inference. Pressing
**Analyze Meal** again re-runs it.

### 5.3 What the app guarantees about honesty

* No model → a real error with repair steps is shown; **no label is ever invented**.
* Low confidence (below 40 %) **or** a narrow margin (< 10 points between #1 and #2) sets
  `is_uncertain`, and the UI explicitly asks the user to confirm the food (Phase 4 will let them
  pick from the top-5 list before nutrition is looked up).
* Results show the backend, model, class count, input size, inference time, and the reminder that
  Food-101 does not contain Indian home-cooking staples such as dal, chapati, idli or paneer — a
  photo of those will land on the nearest Western dish.
* Uploaded images are processed in memory and **never written to disk** by the app.

### 5.4 Measured behaviour on real photos (honest numbers)

Four real web photos were used during development (kept out of the repository — they are
third-party images):

| Photo | Top-1 | Confidence | Notes |
| --- | --- | --- | --- |
| Plate of samosa (single object, clean shot) | **Samosa** | 66.5 % | Spring rolls 32.5 % second — correct behaviour |
| Plate of samosa (busy plate, garnish, sauce) | Chicken wings | 11.1 % | flagged **uncertain** (margin 0.4 pts) — the model correctly admits it is guessing |
| Margherita pizza (staged product shot) | Bruschetta | 47.9 % | `pizza` is #2 at 34.8 % |
| Margherita pizza (top-down, dark background) | Sushi | 41.5 % | wrong — a genuine miss; `pizza` appears at 5 % |

Takeaway: the model is useful but far from perfect on casual photos, exactly as a 76 % top-1
lightweight classifier should be. This is why the UI shows the full shortlist and asks for
confirmation instead of pretending the top label is a fact.

### 5.5 Commands

```bash
# Runs out of the box with the shipped ONNX model (no PyTorch needed):
pip install -r requirements.txt
streamlit run app.py        # → sidebar → Upload Meal

# Optional: re-create the ONNX export yourself (needs PyTorch once)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m food_recognition.export_onnx --quantize --out assets/models

# Backend control
FOOD_MODEL_BACKEND=torch streamlit run app.py     # force PyTorch
FOOD_MODEL_BACKEND=onnx  streamlit run app.py     # force ONNX Runtime
FOOD_MODEL_PREFER_INT8=1 streamlit run app.py     # use the 2.5 MB quantised model
```

The INT8 model agreed with FP32 on **all four** sample photos, but quantisation shifts
probabilities by up to ~0.025 and can flip the top-1 on near-uniform, ambiguous inputs — which is
why **FP32 is preferred by default** and the INT8 tolerance is handled explicitly in the exporter.

### 5.6 Phase 3 limitations

* **Single food per photo.** A full thali is classified as whichever single dish looks most
  dominant. Multi-item detection (e.g. an object detector plus per-item classification) is the
  documented future enhancement.
* **Food-101 ≠ Indian home cooking.** Missing classes (dal, chapati, idli, paneer dishes,
  poha, upma, biryani is only partial…) are mapped to the nearest Western class, so protein/calorie
  accuracy for Indian meals will depend on the user confirming the dish in Phase 4.
* **No portion estimation** yet — the user sets the portion size in Phase 4.
* Accuracy above is measured on 4 photos, not a validation set: it is *illustrative*, not a
  benchmark. The 76.3 % figure is the model card's Food-101 test-split number.
* Recognition only: nothing is saved to the database on this page (Phase 5 adds the explicit
  **Save Meal** button). Nutrition values are not calculated yet (Phase 4).

---

## 6. Nutrition engine & Health Score (Phase 4)

### 6.1 The dataset

`nutrition/food_nutrition.csv` — **134 foods**, values **per 100 g** of the edible portion,
covering what people actually eat in India: rice, roti/chapati, paratha, poha, idli, dosa, upma,
khichdi, biryani; dals (toor/moong/masoor/chana), rajma, chole, sambar, rasam, kadhi; paneer
dishes, curd, raita, lassi, milk, ghee; sabzis (aloo gobi, bhindi, palak, methi, lauki, cabbage);
chicken, mutton, fish, prawns and eggs; street food (samosa, pakora, vada, dhokla, bhel, pani puri,
pav bhaji, chole bhature, momos, tikki); Western items the recognition model can return (pizza,
pasta, burger, fries, sushi, cake, doughnut); fruits, nuts, sweets and beverages.

Each row carries: `aliases` (synonyms), `model_labels` (which Food-101 classes map to it),
`category`, `diet_type` (vegan / vegetarian / eggetarian / non_vegetarian), a typical
`serving_label` + `serving_grams`, the seven nutrient columns (calories, protein, fat, carbs,
fibre, sugar — sugar blank where unknown), `sodium_mg` (blank where unknown), `tags`
(fried / sweet / raw / legume / leafy / high_protein / natural_sugar / high_sodium …) and a
`note` describing what the values assume.

**Data provenance is stated, not implied.** Values are typical averages rounded from published
Indian and international food composition tables (IFCT 2017 / USDA FoodData Central style). Food
composition genuinely varies with recipe, oil, salt and cut, so the app calls every number an
estimate. Every row is validated by the test suite:

* no negative or missing core values, `fibre ≤ carbs`, `sugar ≤ carbs`, unique ids and names;
* **energy balance**: fibre-corrected Atwater (4 kcal/g protein and available carbohydrate,
  2 kcal/g fibre, 9 kcal/g fat) must match the stated calories within 15 %;
* diet types restricted to the four allowed values.

That energy-balance check caught five sloppy rows during development (tea/coffee values were
inconsistent); they were corrected rather than tolerated.

### 6.2 Food-name matching — the documented rules

The recognition model returns Food-101 class names (`samosa`, `chicken_curry`, `bruschetta`),
while the dataset uses human names ("Samosa (fried)"). Matching runs in this order, and the method
plus a score are always reported to the user:

| # | Method | Rule | Result |
| --- | --- | --- | --- |
| 1 | `exact` | Normalised names identical (lower-case, punctuation and bracketed text removed, whitespace collapsed, plurals singularised) | score 100, **no confirmation** |
| 2 | `alias` | Query equals one of the row's `aliases` (`phulka` → Chapati) | score 96, **no confirmation** |
| 3 | `model` | Query equals one of the row's `model_labels` — the curated Food-101 → CSV mapping, stored *in the CSV* so it is data, not code | score 92, **no confirmation** |
| 4 | `token` | Every query token appears in a name/alias, with at least one token ≥ 4 characters | score 80, **confirmation required** |
| 5 | `fuzzy` | `difflib` similarity ratio ≥ 0.72 against any name/alias | score = ratio (capped 89), **confirmation required** |
| — | `none` | Nothing scored | **no food returned** — the UI must ask the user to pick; values are never invented |

Confirmation is required when the match is not confident **or** the Phase 3 recognition was
uncertain (`is_uncertain`). The page then pre-selects the suggestion, explains *why* it is asking,
and offers a searchable list of all 134 foods plus diet-compatible filtering. When nothing matches,
**nothing is pre-selected at all** and no table or score is rendered until the user chooses — this
is verified by tests.

### 6.3 Portion scaling and daily context

`NutritionEngine.calculate(food, grams)` scales the per-100 g values **linearly** (documented; no
density or cooking-loss correction) inside the accepted range 1–2000 g, then adds:

* the macro energy split (protein/carb/fat shares of the portion's energy, using 4/4/9 kcal per g);
* the serving equivalence ("150 g ≈ 1.0 × 1 katori (150 g)");
* **daily-target context** from the Phase 1 profile: % of the daily calorie target, % of the
  protein target and % of the fibre target. Fat and carbohydrate have no defined daily target in
  this project, so they are shown as % of energy instead of a fabricated target.

### 6.4 The Nutrition Health Score

Rule-based, 0–100, **no machine learning**, and every point is displayed in the UI:

| Component | Max | Rule |
| --- | --- | --- |
| Protein density | 22 | `4·P / kcal`: 0 → 22 points from 0 % to 25 % of energy |
| Fibre density | 22 | fibre per 100 kcal: 0 → 22 points up to 2.5 g/100 kcal |
| Sugar load | 16 | `4·sugar / kcal`: full at ≤ 5 % of energy, 0 at ≥ 30 %. **Exemption:** foods tagged `natural_sugar` (whole fruit, plain dairy, coconut water) keep full marks, because the dataset cannot separate intrinsic from added sugar |
| Fat balance | 16 | 4 points at 0 % fat energy → full in the 15–30 % band → 0 at ≥ 60 %. **Guards:** exactly 0 g fat scores the low plateau; a low-energy food (≤ 200 kcal) with 0–5 g fat/100 g keeps full marks (the taper targets genuinely high-fat foods, not a spoonful of curd) |
| Sodium | 12 | 12 points at ≤ 120 mg/100 g, falling to 0 at ≥ 800 mg/100 g |

Food-type adjustments (capped at +8 / −12): `raw`/`fruit` +6, `vegetable` +5, `legume` +4,
`leafy` +4, `high_fibre` +3, `high_protein` +3, `wholegrain` +2, `refined` −2, `processed` −4,
`high_sugar` −4, `sweet` −5, `high_sodium` −6, `fried` −8.

Missing data is handled explicitly: if sodium is unknown the component is dropped and the score is
rescaled (with a visible note) so foods stay comparable; if sugar is unknown the component is
omitted.

Bands: **78–100 Excellent · 62–77.9 Good · 45–61.9 Fair · 32–44.9 Poor · 0–31.9 Very poor.**

Example scores from the shipped dataset (computed by `tests/test_health_score.py`):

| Food | Score | Band |
| --- | --- | --- |
| Moong sprouts (raw) | 95.3 | Excellent |
| Dal toor (cooked) | 89.4 | Excellent |
| Oats (cooked) | 80.0 | Excellent |
| Chapati (whole wheat) | 77.9 | Good |
| Apple | 76.0 | Good |
| Chicken breast (grilled) | 69.0 | Good |
| Curd (dahi) | 62.2 | Good |
| Egg (boiled) | 52.9 | Fair |
| Pizza (veg, cheese) | 51.1 | Fair |
| French fries | 37.8 | Poor |
| Samosa (fried) | 34.2 | Poor |
| Gulab jamun | 20.2 | Very poor |
| Cola | 8.0 | Very poor |

**Known biases, stated openly:** the score measures composition per calorie, so fibre-free animal
foods (eggs, fish) land mid-range even though they are nutritious; fat *quality* is not in the data
(only quantity); total sugar is used, not added sugar; and whole-grain/refined status is only
partially tagged. It is a transparent wellness heuristic — **not** a medical or dietary
prescription.

### 6.5 What the page shows

After the recognition step, the Upload Meal page renders: the dataset row in use (name, category,
match method, match score, note), a diet-compatibility check against the profile preference
(e.g. a non-vegetarian match for a vegetarian profile), portion presets + slider + exact gram entry,
a **results table** (per portion, per 100 g reference, daily context), summary cards, and the health
score with an expandable rule-by-rule breakdown. Nothing is saved — the explicit **Save Meal**
button is Phase 5.


---

## 7. Saving meals, Dashboard and History (Phase 5)

### 7.1 The Save Meal flow, and how duplicates are prevented

A meal is written to the database **only** when the user presses **Save Meal** on the Upload Meal
page. Analysis alone never writes anything. Two independent guards make accidental duplicates
practically impossible, because Streamlit re-runs the whole script on every interaction:

1. **Session lock** — after a successful save the button is replaced by a "saved as meal #N" card.
   A rerun re-renders that card instead of the button, so the same analysis cannot be saved twice.
2. **Database guard** — `repository.find_duplicate_meal()` looks for a meal with the same user,
   same food name, same portion (±0.5 g tolerance for float rounding) and the same calorie value
   written in the last `DUPLICATE_GUARD_SECONDS` (120 s). If found, the existing row is returned and
   the insert is skipped, so a double click racing a rerun cannot create a second row either.

Verified end to end during development: save → 1 row; two extra reruns → still 1 row; a second
`store_meal()` call for the same dish/portion → *"already saved a moment ago as meal #1"*, DB still
1 row.

Additional honesty rules in the same section:

* no database → the section explains that the meal **cannot be stored yet**, with the fix command,
  and keeps the analysis on screen (nothing is lost, nothing is faked);
* profile without a database row → saving is blocked with instructions, because meals reference
  `users.user_id` by foreign key;
* the card shows exactly what will be stored (food, portion, energy, macros, fibre, score, owner);
* the photo itself is never uploaded or stored — only the computed values.

After saving, the user can jump to the Dashboard or press **Save another meal** (which clears the
analysis, the match and the portion for a fresh capture).

### 7.2 Dashboard (`pages/1_Dashboard.py`)

* **Today** — energy with % of the daily target and a progress bar, plus a remaining/over-target
  message; protein and fibre against their targets; average health score for the day; a
  macronutrient donut of where today's energy came from.
* **Trends** — Plotly charts built by `utils/analytics.py`:
  daily energy bars with the target reference line, health-score line (fixed 0–100 axis),
  weekly average energy bars coloured by distance from target (green ≤10 %, amber ≤25 %, red beyond),
  and stacked daily macros. Weeks with no logged meals are dropped rather than drawn as a
  0-kcal week, and days with no meals are drawn as real zeros so gaps are visible.
* **Recent meals** — the last six saved rows as a dataframe.
* **Latest AI recommendation** — an honest card stating the coach is Phase 7 and that no advice is
  shown until it exists (it renders stored rows instead as soon as Phase 7 writes them).
* Every chart has an annotated empty state, so the page is usable from day one.

### 7.3 Meal History (`pages/3_Meal_History.py`)

* Date range presets (7/14/30/90 days, all time) or a **custom** `date_input` range.
* Food-name search, a row limit, totals for exactly the filtered set (meals, energy, average per
  meal, average health score).
* The filtered table, a **CSV download of exactly those rows**, and a calorie chart for the range.
* **Delete a meal** with a two-step confirmation (this cannot be undone) — useful when a wrong dish
  was saved; the delete is scoped by `meal_id`.

### 7.4 Design note: where the data flow lives

| Module | Responsibility |
| --- | --- |
| `utils/analytics.py` | Pure functions: totals/averages, dataframes, weekly aggregation, five Plotly figure builders with a shared dark theme and empty-state annotations. No DB, no Streamlit — fully unit tested. |
| `utils/data_access.py` | The Streamlit-facing layer: cached database status, range resolution, history/today queries, and `store_meal()` — the single write path in the whole app. |

This split keeps the pages thin and means the write path can be audited in one place.


---

## 8. PNIE — Predictive Nutrition Intelligence Engine (Phase 6)

### 8.1 What it predicts

Exactly **one** number with machine learning: *the average daily calories of the next seven days.*
One XGBoost regressor is trained per user from that user's own saved meals. Everything else the
Predictions page shows — averages, spread, week-over-week change, direction, the weekly table — is
plain arithmetic over the same rows. There is no second model and no hidden logic.

### 8.2 How a training row is built (`pnie/feature_engineering.py`)

| Step | Rule |
| --- | --- |
| Daily series | Saved meals are bucketed per calendar day (UTC) by `repository.get_daily_totals`, then re-indexed onto a continuous calendar grid. Days without meals are marked `logged = False` and are **never** treated as 0 kcal days — they are missing data, exactly like the Phase 5 charts. |
| Origins | One training row every `PNIE_ORIGIN_STRIDE_DAYS` (7) days — one row per week. |
| Features | Computed from the `PNIE_LOOKBACK_DAYS` (14) days **strictly before** the origin. |
| Target | Average calories over the **logged** days in the 7 days starting at the origin (the forecast week). |
| Discarding | A window is dropped when its target week has fewer than `PNIE_MIN_TARGET_LOGGED_DAYS` (3) logged days, or when the look-back holds too few logged days to describe. Both counts are reported in the UI instead of being hidden. |

The 20 model inputs — 3/7/14-day calorie means, 7-day spread, 14-day slope, 7-vs-14-day momentum,
three lags, protein/fat/carbohydrate/fibre/health-score means, macro energy shares, meals per logged
day, logged ratios and the origin weekday — are declared once in `FEATURE_COLUMNS`, and training and
prediction both call the same `feature_vector()`, so the two can never drift apart. A unit test
perturbs every day at and after an origin and asserts the feature vector does not move: that is the
leakage guarantee, and it is tested rather than promised.

### 8.3 The model (`pnie/predictor.py`)

* `XGBRegressor` with deliberately small trees (`max_depth=3`, 300 rounds, `learning_rate=0.06`,
  `random_state=42`): with 8–40 weekly windows a deeper model would memorise noise.
* **Chronological split** — the oldest rows train, the newest `PNIE_TEST_FRACTION` (25 %) are held
  out. A random split would let the model train on the future and then "predict" it.
* The **hold-out MAE/RMSE** is the error shown to the user; the model used for the live forecast is
  then **refit on all rows** (standard practice for a small dataset — stated on the page too).
* The **baseline** (always predicting the mean) is reported next to it, so it is obvious when the
  model is *not* worth much: on smooth demo data the baseline can win, and the page says so instead
  of hiding the comparison.
* Training takes milliseconds and happens on demand, so no model file is committed or downloaded.

### 8.4 Honesty rules (`pnie/pnie_engine.py`)

The engine returns exactly one mode and never fills a gap with an invented number:

| Mode | When | What you see |
| --- | --- | --- |
| `no_data` | nothing saved yet | a prompt to save meals |
| `insufficient_history` | < 14 days or < 7 logged days | descriptive statistics + how many more days are needed |
| `descriptive_only` | statistics possible, < 8 weekly windows | statistics + the shortfall ("about N more days of logging") |
| `awaiting_recent_data` | the last 14 days hold too few logged days | a request to log more — a forecast would describe days never recorded |
| `model_unavailable` | training failed (e.g. XGBoost missing) | the real error message |
| `model` | enough history | the forecast, its range, its measured error and its explanation |

Two guards are applied to every forecast **and disclosed in the UI**: an absolute clamp to
`PNIE_PREDICTION_KCAL_BOUNDS` (600–6,000 kcal/day) and a relative guard that pulls a prediction back
when it strays more than `PNIE_MAX_DEVIATION` (60 %) from the recent average. History older than
`PNIE_STALE_AFTER_DAYS` (7) is flagged so a stale pattern is never presented as current.

### 8.5 The Predictions page (`pages/4_Predictions.py`)

* Four KPI cards: the forecast (with its hold-out range), the change against the last recorded week,
  the difference from the profile's calorie target, and the typical error plus a "better / not better
  than the baseline" verdict.
* **History and forecast chart**: one point per *logged* day (gaps stay gaps), a dotted 7-logged-day
  average, and the forecast as a dashed line across its window with the error as a shaded band and a
  dotted divider where the records end.
* **Why this number**: the fitted model's own feature importances, the chronological split, MAE,
  RMSE, baseline and training error, the number of discarded windows, and every guard that fired.
* **Your trends so far**: average, spread, min/max, last vs previous week, slope per week, macro
  energy split, highest/lowest logged day and log consistency — the honest answer whenever the model
  cannot run, and a sanity check when it can.
* **Stored forecasts**: the forecast is computed on every page load, but a `predictions` row is
  written **only** when you press *Save this forecast to my records* — the same explicit-action rule
  as Save Meal. A forecast identical (within 0.5 kcal) to one saved in the last
  `PNIE_DUPLICATE_GUARD_SECONDS` (300 s) is reported instead of duplicated, and the table below
  lists previous forecasts with their model, version, window count and MAE. The Dashboard reads the
  newest stored row for its *Next week (PNIE)* card.
* A methodology expander states the leakage rules, thresholds and limits, and the page ends with the
  non-medical-advice notice.

### 8.6 Try it in two minutes

```bash
python database/init_db.py --sample-data     # 12 weeks of flagged demo meals
streamlit run app.py                         # "Continue as" → load the demo user, then open Predictions
```

The demo history is deterministic (no randomness) and contains six deliberately unlogged days, so
the gaps, the logged-ratio features, the weekly aggregation and the forecast are all visible.

### 8.7 Module map

| File | Responsibility |
| --- | --- |
| `pnie/feature_engineering.py` | Daily series, the 20-feature vector, weekly training rows, the live row. Pure — no DB, no Streamlit. |
| `pnie/trend_analysis.py` | Descriptive statistics, least-squares slope, direction labels. No ML at all. |
| `pnie/predictor.py` | The XGBoost regressor: chronological split, hold-out error, refit, absolute clamp. |
| `pnie/pnie_engine.py` | Orchestration, the six modes, guards, JSON-safe metadata for the database. |
| `utils/data_access.py` | *(extended)* `pnie_daily_totals`, `run_pnie_for_user` (cached by data content), `store_prediction`, `stored_predictions`, `latest_stored_prediction`. |
| `utils/analytics.py` | *(extended)* `prediction_forecast_figure`, `feature_importance_figure`. |

### 8.8 Phase 6 dependencies

`xgboost>=2.0` (installed 3.4.1 here) — the only new package. It is CPU-only, needs no GPU and no
downloaded weights: the model is trained from your own rows in milliseconds.


---

## 9. Gemini 2.5 Flash AI coach (Phase 7)

### 9.1 What it does

The coach turns the app's *own* numbers into one short, practical message. It is the only component
that talks to the internet, and it does so under three rules:

1. **One call per button press.** Nothing is requested on page load, so browsing the app or letting a
   rerun happen never spends API quota.
2. **A factual snapshot only.** The prompt is built from the profile, the meal averages and the PNIE
   forecast — see 9.2. The model is instructed to use *only* those numbers, and to say what is
   missing instead of guessing.
3. **Store what was generated.** Every successful answer is written to the `recommendations` table
   with its model name and the question asked, so AI text is always traceable and never mixed up
   with the rule-based nutrition engine.

### 9.2 What the coach sees — and what it never sees

| Sent in the prompt | Not sent |
| --- | --- |
| Age, sex, height, weight, BMI and its band, activity level, diet preference, fitness goal | **Your name** (deliberately excluded from the prompt) |
| The app's calculated daily targets (kcal, protein, fibre) | Your e-mail or any account data |
| History summary: days covered, days logged, meals saved, coverage % | **Your food photos** — they never leave your machine |
| Averages per logged day (kcal, protein, fat, carbs, fibre), average health score, macro energy split, last vs previous week, slope per week | Your database URL or credentials |
| Up to 8 recent meals (name, grams, kcal, health score) and the most frequent foods | Any other user's data |
| The PNIE forecast (value, range, direction, typical error, window) | |
| Your question, if you typed one | |

The page renders the whole snapshot as a card **and** lets you expand the exact prompt before you
press the button, so nothing is hidden. The privacy note on the page repeats this list.

### 9.3 Failure handling

Every failure returns a specific `error_kind`, a plain-language message and advice — never a
substitute recommendation, and never a silent failure:

| `error_kind` | Cause | What the page shows |
| --- | --- | --- |
| `not_configured` | no `GEMINI_API_KEY` | setup card with the two places to put the key; the button stays disabled |
| `sdk_missing` | `google-genai` not installed | the install command |
| `invalid_key` | the API rejected the key (verified against the live endpoint: HTTP 400 `API_KEY_INVALID`) | "check the key in Google AI Studio" |
| `permission_denied` | key not allowed to use the model | enable the API for that project |
| `model_not_found` | wrong model id | check `GEMINI_MODEL_NAME` |
| `quota_exceeded` | free-tier quota / rate limit (HTTP 429) | "wait a few minutes and try again" |
| `server_error` | Google 5xx | "usually temporary" |
| `timeout` / `network_error` | no answer within `GEMINI_REQUEST_TIMEOUT_MS` / connection failed | retry guidance |
| `blocked` | safety filters stopped the answer | what happened; nothing invented in its place |
| `empty_response` | model returned no text | nothing is shown rather than filler |

The API key is read from Streamlit secrets or the environment, is never logged, and is scrubbed out
of any error text before it reaches the UI (unit-tested).

### 9.4 The pages

* **AI Coach (`pages/5_AI_Coach.py`)** — status badge (ready / key missing), the setup card when
  needed, the "what the coach will see" snapshot + expandable prompt, an optional question box
  (400 chars), one primary button, the answer card with model/latency/tokens, the latest stored
  recommendation (with a stale warning after `COACH_STALE_AFTER_DAYS` = 7 days), the full history,
  the "what is sent to Google" note and the non-medical-advice disclaimer.
* **Dashboard** — the *Latest AI recommendation* card now shows the newest stored answer with its
  timestamp and source, or an honest "no recommendation yet" card pointing at the coach page. No
  canned tips ever appear there.

A second press with the **same question** within `COACH_DUPLICATE_GUARD_SECONDS` (120 s) reuses the
stored answer instead of spending another call — the same "explicit action" pattern as Save Meal
(Phase 5) and Save this forecast (Phase 6).

### 9.5 Setup and quota

```bash
cp .env.example .env      # then set GEMINI_API_KEY=...  (free key: https://aistudio.google.com/app/apikey)
streamlit run app.py      # sidebar → AI Coach
```

On Streamlit Community Cloud, add the same key under *App → Settings → Secrets*. The coach uses
`gemini-2.5-flash` with `temperature=0.6`, `top_p=0.95`, `max_output_tokens=1200` and a 45 s timeout;
the free tier is enough for normal use (one request per press). Without a key the app is fully
functional — only this page asks for one.

### 9.6 Module map

| File | Responsibility |
| --- | --- |
| `ai/gemini_coach.py` | `build_context()` (snapshot from profile + meals + PNIE outcome), `NutritionContext.to_prompt()`, `GeminiCoach` (SDK wrapper), `_classify_error()` (failure mapping), `_redact()` (key scrubbing), `SYSTEM_INSTRUCTION`, `ERROR_*` + `ERROR_ADVICE`. Defensive SDK import — the app runs without it. |
| `ai/__init__.py` | The public surface used by pages and tests (coach, results, error vocabulary). |
| `utils/data_access.py` | *(extended)* `coach_status()`, `coach_context()`, `request_recommendation()` (the only writer), `stored_recommendations()`, `coach_is_stale()`. |
| `pages/5_AI_Coach.py` | The whole UI described above. |
| `tests/test_gemini_coach.py` | 41 tests with an injected fake client: snapshot accuracy, prompt contents (including "the name is never sent"), request configuration, all twelve failure kinds, key scrubbing, storage, duplicate guard, stale marking — no network, no key needed. |

### 9.7 What the coach deliberately does not do

* It never computes nutrition values: every figure in its message must come from the snapshot.
* It does not run automatically, on a schedule, or on page load.
* It does not diagnose, prescribe, or promise weight change — the system instruction forbids it, and
  the page repeats that the text is AI-generated, estimate-based, general wellness information.
* It cannot see your photos, and it will not "analyse" an image: recognition stays local (Phase 3).


---

## 10. How the database layer is organised

| File | Responsibility |
| --- | --- |
| `database/connection.py` | Reads `DATABASE_URL` (Streamlit secrets → environment → `.env`), normalises provider URLs, builds one cached engine (`pool_pre_ping` keeps Neon's idle-suspend wakeups painless), exposes `get_session_factory()`, and provides `check_database()` which reports backend, target and missing tables **without** raising. |
| `database/repository.py` | Every query in the project. All statements use bound parameters, so user input is never interpolated into SQL. `session_scope()` commits on success and rolls back on error; the `@db_session` decorator closes the session for you. If a table is missing (you opened a page before running `init_db.py`), reads return an empty result and log a diagnostic instead of crashing the UI. |
| `database/init_db.py` | CLI to create, verify, reset and seed the schema. Also the reference for how to run database code outside Streamlit. |
| `models/*` | ORM classes with type hints (`Mapped[...]`), check constraints, indexes and relationships. `models/base.py` holds `Base` and `utcnow()` (timezone-aware, works on PostgreSQL `timestamptz` and SQLite). |

Example use in a future page:

```python
from database import get_daily_totals, list_recent_meals, save_meal
from models import Meal

totals = get_daily_totals(user_id, start=..., end=...)   # ready for Plotly
recent = list_recent_meals(user_id, limit=5)
save_meal(Meal(user_id=user_id, food_name=..., portion_grams=..., calories=...))
```

---

## 11. Secrets

* **Local:** `cp .env.example .env`, then fill in `DATABASE_URL` and `GEMINI_API_KEY` (the coach key
  is optional — without it every feature except the AI Coach page still works).
* **Streamlit Community Cloud:** *App → Settings → Secrets* (TOML):

```toml
DATABASE_URL = "postgresql+psycopg2://user:password@host/dbname?sslmode=require"
GEMINI_API_KEY = "your-key-here"
```

Read them only through `config.get_secret("DATABASE_URL")` / `config.get_secret("GEMINI_API_KEY")`
(the coach additionally accepts an explicit `api_key=` argument, which is how the tests inject a fake
client). Values are never logged, and the coach scrubs its key out of any API error before displaying
it.
`.gitignore` blocks `.env`, `.streamlit/secrets.toml`, `*.db`, model weights and caches.

---

## 12. Limitations (cumulative)

**Database (Phase 2)**

* **No authentication.** A returning user is matched by exact name (case-insensitive) and offered
  in the sidebar. This is a demo convenience, not a login system.
* **No migrations.** Column changes during development need `init_db.py --reset` (which deletes
  rows) or Alembic.
* Meal timestamps are stored UTC-aware and aggregated by UTC day; per-user timezones are not yet
  applied.
* Neon's free tier suspends when idle, so the first query after a pause takes ~1 s.
* The SQLite fallback is development-only (single-writer, file ignored by Git).

**Recognition (Phase 3)**

* Single food per photo; no multi-item detection on mixed plates.
* Food-101 covers 101 specific dishes and **no Indian staples** (dal, chapati, idli, poha, paneer
  curries). Unrecognised dishes fall back to the nearest Western class, which is why the user is
  asked to confirm uncertain labels.
* Confidence is the model's own softmax output — it is **not calibrated**, so a 66 % score is a
  ranking hint, not a guarantee.
* No portion-size estimation; that is user input in Phase 4.
* Accuracy numbers in §5.4 come from four photos and are illustrative only.

**Nutrition engine (Phase 4)**

* Values are **averages from composition tables**, not measurements of your actual plate: recipes,
  oil, salt and portion sizes vary enormously. Two cooks, two very different numbers.
* Portion scaling is linear per 100 g; there is no density, water-loss or cooking-method correction.
* Fat *type* (saturated/trans) and added-vs-natural sugar cannot be distinguished from this dataset.
* 134 foods is a curated subset — regional dishes, restaurant preparations and brand products are
  not covered; unknown dishes require the user to choose the closest row.
* Food-name matching is string-based (no embeddings), so unusual spellings can require a manual pick.
* The health score is a transparent heuristic with the known biases listed in §6.4 (fibre-free
  animal foods and fruit are scored less favourably than a dietitian might rank them).

**Saving & dashboard (Phase 5)**

* "Today" follows the **UTC calendar day** (the same convention the daily aggregation uses), not the
  user's local day. A meal logged at 02:00 IST counts toward the previous UTC day.
* The duplicate guard is a *same food + same portion + same calories within 120 s* heuristic: it is
  deliberately conservative, so two genuinely identical meals logged minutes apart would be treated
  as one. It reports what it did, and there is no silent data loss beyond that window.
* Deleting a meal is permanent and there is no bulk delete or edit (portion corrections mean deleting
  and re-saving).
* Charts cover at most one year in the history view for very long custom ranges (the table still
  shows every row).
* The dashboard has no PDF/image export and no goal-weight projection.
* Meals can only be created through the Upload Meal page — there is no manual "add without a photo"
  entry (a possible future improvement).

**PNIE / forecasting (Phase 6)**

* **The model cannot know about the future.** It learns your recent pattern and continues it; a
  festival week, illness, travel or a new diet will not be anticipated.
* Tree models cannot extrapolate beyond the targets they were trained on, so a forecast is a
  *continuation* of the observed range, not a projection of a steep new trend. The relative guard
  (±60 % of the recent average) and the 600–6,000 kcal clamp are sanity limits, not accuracy.
* One regression target only (average daily calories). Protein, micronutrients and body weight are
  **not** forecast — adding them would mean more models, which is out of scope here.
* The reported error is a **hold-out average** from 1–N held-out weeks; with the minimum history it
  can be based on very few weeks and is therefore noisy. The page always states how many.
* Accuracy depends entirely on logging consistency: gaps are treated as missing data, and long gaps
  reduce the usable windows (the page reports how many were discarded and why).
* The training window is capped at the last 365 days; older meals are excluded (and that is stated).
* The engine is deterministic — the same history always yields the same forecast — which also means
  it does not "improve" between retrains unless your data changes.
* Predictions are stored only when you press the save button; there is no scheduled/automatic
  forecast, no comparison of multiple models, and no hyper-parameter search (a fixed, small,
  documented configuration is used so results stay reproducible).

**AI coach (Phase 7)**

* The message is **generated text**: it can be generic, repetitive or wrong, and it only knows the
  snapshot. Treat it as a starting point, not a plan.
* Quality depends on your history: with few logged meals the coach gets little to work with (it is
  told to say so) and with none it can only give profile-based general advice.
* Answers are **not** reviewed by a human and are not medical advice; there is no dietician in the
  loop, no allergy checking and no drug/food interaction checking.
* The coach does not see your food photos, so it cannot comment on a specific unpublished meal.
* Usage depends on Google's free-tier quota and on your internet connection; every failure is
  reported with its cause instead of being papered over.
* Stored recommendations are snapshots of their moment: after logging new meals they may be out of
  date (the UI flags them after 7 days, but it does not rewrite them).
* The prompt is truncated at `GEMINI_MAX_PROMPT_CHARS` (8,000) — irrelevant for normal histories,
  but a very long log would send fewer recent meals.
* No streaming, no conversation memory: each press is a single request with no chat history.

**Deployment (Phase 8)**

* The free Neon tier **suspends idle databases**; the first query after a pause can take a second or
  two, and a deploy that has been idle overnight may look slow on the first page load.
* Streamlit Community Cloud gives each app a small container: the bundled INT8 ONNX model is used
  when available, and recognition is deliberately limited to one image at a time.
* Deleting rows does not shrink a PostgreSQL database file — reclaim space with Neon's *Vacuum*.
* The app still has **no authentication**: a deployed link is usable by anyone who has it, and the
  person who opens it sees whichever profile is in their session.

**Not implemented yet (by design, and labelled as such in the UI)**

* Nothing — Phases 1–8 are all delivered. Ideas that were deliberately left out are listed at the
  end of section 15.6 (authentication, photo storage, multi-dish recognition, offline install).
* Nothing writes to `meals` outside the explicit Save Meal flow and `init_db.py --sample-data`;
  nothing writes to `predictions` outside the explicit save button on the Predictions page; nothing
  writes to `recommendations` outside the coach button.

## 13. Profile page & final polish (Phase 8)

### 13.1 `pages/6_Profile.py`

| Block | What it shows |
| --- | --- |
| Header | BMI (Asian-Indian bands), maintenance calories, daily target with its deficit/surplus, protein and fibre targets |
| Your profile | every stored field, straight from the `users` row, with the creation timestamp |
| How the targets are calculated | BMR (Mifflin-St Jeor), the activity multiplier, the goal adjustment, the protein rule (`kg × goal factor`) and the fibre rule (14 g / 1000 kcal) — every number can be reproduced by hand |
| Edit your profile | the same validated form as onboarding, prefilled; saving updates the existing row **in place**, recalculates all targets and reports the change (`Daily target 1,286 → 1,765 kcal (+479 kcal)`) |
| What is stored | live counts read from the database (meals, logged days, forecasts, recommendations, first/last meal), the backend and its location, and an explicit list of what is **never** stored: photos, the API key, anything you did not save |
| Delete data | *Delete meal history* (profile, forecasts and recommendations survive) and *Delete everything* (profile + all children), both behind a typed `DELETE` confirmation |

Deletes follow the same "explicit action" rule as Save Meal: nothing is removed by navigation, and the
confirmation message reports the exact row counts that were deleted. Meal-history deletion also
clears the cached PNIE run, so the forecast page immediately reflects the new history.

### 13.2 The polish pass

* **Home is a hub, not a second dashboard.** `app.py` now shows the four headline numbers, a note
  about who can see the profile, and **page links to all six screens** with one-line descriptions.
  The duplicated profile/energy cards were removed (they live on *Profile & data* now, in more
  detail).
* **One footer everywhere.** `ui.page_footer()` prints `NutriVision AI · build 0.8.0 · Phase 8 …` plus
  the "estimates, not medical advice" line at the bottom of **all seven screens, in both the
  with-profile and without-profile states**.
* **The stale sidebar caption is gone.** It used to say the profile lives only in session state "until
  Phase 2"; it now states the real `user_id`, what is written and when.
* **New components.** `ui.key_value_grid()` (storage summary) and `ui.page_footer()` in
  `utils/ui.py`, plus `.nv-kv` and `.nv-danger` styles in `assets/style.css` — the destructive
  actions are visually separated from ordinary buttons.
* **Consistent structure.** Every page keeps `st.set_page_config(...)` with the shared name and icon,
  the same hero → sections → notices → footer rhythm.

### 13.3 Module map

| File | Responsibility |
| --- | --- |
| `database/repository.py` | *(extended)* `user_data_summary()` (counts, span, `exists`), `delete_meals_for_user()`, `delete_user()` (explicit child-first deletes so the behaviour is backend-independent and the caller can report real counts) |
| `utils/data_access.py` | *(extended)* `ProfileOverview`, `profile_overview()`, `persist_profile()` (create / update-in-place / match-by-name), `delete_meal_history()`, `delete_profile_and_data()` |
| `utils/ui.py` | *(extended)* `page_footer()`, `key_value_grid()` |
| `assets/style.css` | *(extended)* `.nv-kv`, `.nv-danger` |
| `pages/6_Profile.py` | The page itself |
| `app.py` | Hub landing, page catalogue, sidebar storage caption, roadmap row, footer |
| `tests/test_profile_data.py` | 23 tests: summary counts/span, `exists`, overview formatting, create/update-in-place/no-duplicate, error reporting, both deletes, cascade without FK enforcement |

### 13.4 What is protected by tests

* Saving twice never creates a duplicate account (matched by `user_id`, then by name).
* A database failure returns `(None, message)` instead of raising, so the session profile is never lost.
* `delete_meal_history` removes meals **only** — forecasts and recommendations stay.
* `delete_user` removes the user plus meals, predictions and recommendations, leaves other users
  untouched, and reports the real counts.
* The user's counts/summary are read-only: opening the Profile page writes nothing.

## 14. Roadmap

| Phase | Scope | Status |
| --- | --- | --- |
| 1 | Streamlit UI, onboarding, validation, session profile | ✅ |
| 2 | Neon PostgreSQL + SQLAlchemy models + repository layer | ✅ |
| 3 | Food image recognition with confidence display | ✅ |
| 4 | Nutrition engine (Indian food CSV, matching, health score) | ✅ |
| 5 | Explicit Save Meal flow, dashboard with Plotly charts, history filters | ✅ |
| 6 | PNIE: XGBoost next-week calorie prediction, chronological validation | ✅ |
| 7 | Gemini 2.5 Flash nutrition coach with graceful failure handling | ✅ |
| 8 | Profile page, consistent polish, deployment guide | ✅ |

---

## 15. Deploying to Streamlit Community Cloud

The whole app is one Streamlit process plus a PostgreSQL database, so deployment is: push the repo →
point Streamlit at `app.py` → paste two secrets. No Docker, no reverse proxy, no build step.

### 15.1 Before you push — the checklist

```bash
cd NutriVision-AI
pytest -q                       # 300 tests must pass (no key, no database needed)
git init && git add . && git commit -m "NutriVision AI — final"
git status --short              # make sure .env and *.db are NOT listed
```

| Check | Why it matters |
| --- | --- |
| `.env` is **not** committed (it is in `.gitignore`) | it holds the Neon password and the Gemini key |
| `nutrivision.db` is **not** committed | the SQLite fallback is local-only |
| `assets/models/*.onnx` **is** committed (9.0 MB + 2.5 MB) | `.gitignore` ignores `*.onnx` but explicitly re-includes this folder — the deployed app has no internet-free model download of its own¹ |
| `assets/style.css`, `nutrition/food_nutrition.csv`, `food_recognition/food101_classes.json` committed | the theme, the food table and the class names are all required at runtime |
| GitHub repo is **private** if you do not want the project public | Streamlit Cloud can deploy private repos |

¹ *If the weights are missing, `food_recognition/model_loader.py` downloads them once into
`.cache/` and reports exactly what it is doing — but shipping them makes the first load instant.*

```bash
git remote add origin https://github.com/<you>/NutriVision-AI.git
git push -u origin main
```

### 15.2 Create the app

1. <https://share.streamlit.io> → **New app** → *Deploy a public app from GitHub*.
2. Repository `<you>/NutriVision-AI`, branch `main`, **Main file path `app.py`**.
3. **Advanced settings → Python version 3.11 or 3.12** (the pinned requirements support both).
4. **Secrets** (TOML) — this is the only place credentials live in production:

```toml
DATABASE_URL = "postgresql+psycopg2://USER:PASSWORD@ep-xxx.region.aws.neon.tech/neondb?sslmode=require"
GEMINI_API_KEY = "your-key-here"       # optional: without it only the AI Coach page is limited
```

5. **Deploy**. The first build installs ~80 packages (onnxruntime, pandas, plotly, xgboost) and takes
   a few minutes.

### 15.3 Prepare the database (once, from your laptop)

```bash
python database/init_db.py --check      # confirms the Neon URL works and lists existing tables
python database/init_db.py             # creates users/meals/predictions/recommendations
python database/init_db.py --sample-data   # optional: 84 days of demo meals (flagged is_sample)
```

Do **not** run `--reset` against a database that already holds real rows.

### 15.4 Smoke-test the deployed app

| Step | Expected |
| --- | --- |
| Open the app URL | dark theme, "Create your nutrition profile" |
| Complete onboarding | "Database: online (Neon PostgreSQL)" in the sidebar, `user_id` shown |
| Upload Meal → pick a food photo | Food-101 prediction with confidence, nutrition values, health score |
| Press **Save Meal**, then open Meal History | the row is there (and *not* there if you never pressed the button) |
| Predictions → Generate → **Save this forecast** | forecast with its MAE; the row appears under stored forecasts |
| AI Coach → **Get my recommendation** | one call, a short message, "stored as recommendation #N" (or a classified error with advice if the key is missing) |
| Profile & data | counts match what you saved; the deletes work |
| Reload the tab, use *Continue as* | the profile is restored from the database |

### 15.5 Updating and housekeeping

* `git push` → Streamlit Cloud redeploys automatically (it reruns the app on the next visit).
* Change a secret in *Settings → Secrets* and the app restarts by itself.
* Logs: *Manage app → Logs* — they contain the same `INFO`/`WARNING` lines as a local run, and never
  contain the API key or the database password (checked by tests).
* Free-tier characteristics to expect: the app sleeps after long inactivity (first visit wakes it),
  and Neon idles its database (first query after that can take a second).

### 15.6 Ideas that were deliberately left out

Authentication (the app matches profiles by name instead), permanent photo storage, multi-dish
recognition in one photo, barcode scanning, offline packaging, and a mobile layout. Each one is a
feature decision, not an unfinished piece: the README documents what exists and what it costs.

---

## 16. Disclaimer

NutriVision AI provides **general wellness information only**. It does not diagnose, treat or
prevent disease and is not a substitute for a qualified dietitian or physician. Nutrition values
are estimates from published per-100 g food composition data, calorie predictions are statistical
forecasts that can be wrong, and coaching messages are AI-generated text that may be inaccurate or
generic. Nothing in this app replaces professional medical or dietetic advice.
