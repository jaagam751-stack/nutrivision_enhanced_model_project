# 🥗 NutriVision AI

### AI-Powered Food Recognition & Predictive Nutrition Analysis

NutriVision AI is a final-year / NSP project that turns a food photo
into practical nutrition insights. It combines food-image
classification, a curated Indian-food nutrition dataset, meal tracking,
calorie-trend forecasting, and optional AI-generated wellness coaching
in a Streamlit application.

> **Project status:** Phases 1--8 are marked complete in the project
> documentation. Model outputs and nutrition values are estimates, and
> the app clearly identifies important limitations.

## Table of contents

-   [Highlights](#-highlights)
-   [How it works](#-how-it-works)
-   [Technology stack](#-technology-stack)
-   [Project structure](#-project-structure)
-   [Getting started](#-getting-started)
-   [Configuration and secrets](#-configuration-and-secrets)
-   [Database setup](#-database-setup)
-   [Run the application](#-run-the-application)
-   [Run tests](#-run-tests)
-   [Food recognition](#-food-recognition)
-   [Nutrition analysis](#-nutrition-analysis)
-   [Predictive Nutrition Intelligence
    Engine](#-predictive-nutrition-intelligence-engine)
-   [Gemini AI Coach](#-gemini-ai-coach)
-   [Deployment](#-deployment)
-   [Limitations](#-limitations)
-   [Privacy and security](#-privacy-and-security)
-   [Disclaimer](#-disclaimer)

## ✨ Highlights

-   **Food image recognition:** Classifies an uploaded image with a
    pretrained MobileNetV2 model trained on Food-101, displaying
    confidence and a top-five shortlist.
-   **Indian-food nutrition data:** Uses a curated dataset of 134 foods
    with estimated nutrition values per 100 g.
-   **Portion-aware calculations:** Scales nutrition estimates to the
    selected portion and compares them with profile-based daily targets.
-   **Nutrition Health Score:** Calculates a transparent, rule-based
    score from 0--100 and shows its component breakdown.
-   **Explicit meal saving:** Nothing is stored as a meal unless the
    user presses **Save Meal**.
-   **Dashboard and history:** Displays daily totals, trends,
    macronutrient charts, filters, and CSV export.
-   **Calorie trend forecasting:** Uses XGBoost to estimate the next
    week's average daily calories when enough logged history is
    available.
-   **Optional Gemini coaching:** Sends a limited, visible nutrition
    snapshot to Gemini when the user explicitly requests a
    recommendation.
-   **Profile management:** Supports profile review, updates, and
    explicit data deletion.
-   **Testing and deployment guidance:** Includes unit tests and
    deployment instructions for Streamlit Community Cloud.

## 🔄 How it works

``` text
Food photo
   ↓
Image validation and preprocessing
   ↓
MobileNetV2 / Food-101 classification
   ↓
Top prediction + confidence + top-five shortlist
   ↓
Food-name matching against nutrition dataset
   ↓
Portion-based nutrition estimates + Health Score
   ↓
User explicitly saves meal
   ↓
Database → Dashboard / Meal History / PNIE forecast
                         ↓
             Optional Gemini AI Coach
```

The food classifier predicts a single dominant dish; it is not a
multi-item detector. The app asks users to confirm uncertain food
matches rather than presenting every prediction as certain.

## 🧰 Technology stack

  -----------------------------------------------------------------------
  Area                                Technologies
  ----------------------------------- -----------------------------------
  User interface                      Python, Streamlit, custom CSS

  Food recognition                    MobileNetV2, Food-101, ONNX Runtime
                                      (optional PyTorch backend)

  Data processing                     Pandas, NumPy

  Nutrition and charts                Curated CSV dataset, Plotly

  Forecasting                         XGBoost

  Database                            SQLAlchemy, Neon PostgreSQL; SQLite
                                      for local development

  AI coaching                         Google Gemini API (optional)

  Testing                             pytest, Streamlit AppTest

  Configuration                       Environment variables, `.env`,
                                      Streamlit secrets
  -----------------------------------------------------------------------

## 📁 Project structure

``` text
NutriVision-AI/
├── app.py
├── config.py
├── requirements.txt
├── requirements-training.txt
├── README.md
├── RUN_GUIDE.md
├── .env.example
├── .gitignore
├── .streamlit/
│   └── config.toml
├── ai/
│   └── gemini_coach.py
├── assets/
│   ├── style.css
│   └── models/
│       ├── mobilenet_v2_food101.onnx
│       └── mobilenet_v2_food101_int8.onnx
├── database/
│   ├── connection.py
│   ├── init_db.py
│   └── repository.py
├── food_recognition/
│   ├── export_onnx.py
│   ├── food101_classes.json
│   ├── model_loader.py
│   ├── predictor.py
│   └── preprocessing.py
├── models/
│   ├── base.py
│   ├── meal.py
│   ├── prediction.py
│   ├── recommendation.py
│   └── user.py
├── nutrition/
│   ├── food_nutrition.csv
│   ├── health_score.py
│   └── nutrition_engine.py
├── pages/
│   ├── 1_Dashboard.py
│   ├── 2_Upload_Meal.py
│   ├── 3_Meal_History.py
│   ├── 4_Predictions.py
│   ├── 5_AI_Coach.py
│   └── 6_Profile.py
├── pnie/
│   ├── feature_engineering.py
│   ├── pnie_engine.py
│   ├── predictor.py
│   └── trend_analysis.py
├── tests/
└── utils/
```

## 🚀 Getting started

### Prerequisites

-   Python 3.10--3.12 is recommended; use the version supported by the
    installed dependencies.
-   Git
-   A Google Gemini API key only if you want to use AI coaching.
-   A Neon PostgreSQL database for persistent hosted use. SQLite is
    available for local development.

### 1. Clone the repository

``` bash
git clone https://github.com/jaagam751-stack/nutrivision_enhanced_model_project.git
cd nutrivision_enhanced_model_project/NutriVision-AI
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you are only preparing datasets or training/exporting models, consult
`requirements-training.txt` for the additional packages. Avoid
installing the training stack unless you need it.

## 🔐 Configuration and secrets

Copy the example environment file:

**Windows PowerShell**

``` powershell
Copy-Item .env.example .env
```

**macOS / Linux**

``` bash
cp .env.example .env
```

Edit `.env` and set the values you need:

``` dotenv
DATABASE_URL=postgresql+psycopg2://USER:PASSWORD@HOST/DATABASE?sslmode=require
GEMINI_API_KEY=your-gemini-api-key
```

-   `DATABASE_URL` is required for a persistent PostgreSQL deployment.
-   `GEMINI_API_KEY` is optional; without it, the rest of the app can
    still be used, while AI coaching is unavailable.
-   Never commit `.env`, database credentials, API keys, or Streamlit
    secrets to GitHub.

Create a Gemini key through [Google AI
Studio](https://aistudio.google.com/app/apikey). The app should read
secrets through its configuration layer rather than hard-coding them in
source files.

## 🗄️ Database setup

### Option A: Neon PostgreSQL (recommended for deployment)

1.  Create a project at [Neon](https://neon.tech).
2.  Copy its connection string from the Neon dashboard.
3.  Put the connection string in `DATABASE_URL` in `.env`.
4.  From the `NutriVision-AI` directory, verify the connection and
    create the tables:

``` bash
python database/init_db.py --check
python database/init_db.py
```

The application uses SQLAlchemy to manage the `users`, `meals`,
`predictions`, and `recommendations` tables.

### Option B: SQLite for local development

SQLite can be used for local development and demos without a hosted
database:

``` bash
python database/init_db.py --sqlite --sample-data
```

Keep in mind that a local SQLite file is not suitable for persistent
production data on Streamlit Community Cloud.

### Optional sample data

``` bash
python database/init_db.py --sample-data
```

This creates clearly flagged demonstration data for exploring dashboard
and forecasting features. Do not use reset commands against a database
containing data you need.

> **Warning:** `python database/init_db.py --reset --yes` drops and
> recreates tables and can destroy stored data. Use it only on a
> disposable development database.

## ▶️ Run the application

Run from the `NutriVision-AI` directory:

``` bash
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

Typical workflow:

1.  Create or load a profile.
2.  Open **Upload Meal** and upload a JPG, JPEG, or PNG image.
3.  Review the prediction, shortlist, nutrition match, portion, and
    estimated nutrition values.
4.  Press **Save Meal** only when you want to store the meal.
5.  Explore **Dashboard** and **Meal History**.
6.  Open **Predictions** after enough meal history has been recorded.
7.  Configure a Gemini API key and open **AI Coach** if you want an
    AI-generated recommendation.
8.  Use **Profile & data** to review profile values and manage stored
    records.

## 🧪 Run tests

Run the unit test suite from the project directory:

``` bash
pytest -v
```

The project documentation describes tests for profile validation,
database/repository behaviour, food recognition, nutrition calculations,
health scoring, analytics, PNIE, Gemini coach error handling, and
profile data operations. Check the current test output for the actual
pass count and failures; counts can change as the project evolves.

Some Streamlit end-to-end scripts are run separately:

``` bash
python tests/e2e_phase7_apptest.py
python tests/e2e_phase8_apptest.py
```

These scripts may expect a configured development database and suitable
sample data. Use a disposable local database when testing flows that
write or delete records.

## 🍽️ Food recognition

The app uses a MobileNetV2 classifier trained on Food-101. It returns a
predicted class, confidence information, and a top-five shortlist. ONNX
Runtime is the lightweight inference path; a PyTorch backend may be
available for model development.

Important points:

-   The model classifies one dominant food item per image; it does not
    detect every item on a mixed plate.
-   Food-101 does not cover every Indian home-cooked dish. Dal, chapati,
    idli, poha, and paneer dishes may be mapped to a visually similar
    class.
-   Confidence is the model's score, not a guarantee of correctness.
-   Confirm uncertain predictions and nutrition matches before saving.
-   Uploaded photos are processed in memory according to the project
    design and are not intended to be stored as meal records.

Optional model export tools and backend settings are documented in
`food_recognition/export_onnx.py` and `RUN_GUIDE.md`. Use those
instructions if you need to rebuild or troubleshoot model artifacts.

## 🥦 Nutrition analysis

The curated `nutrition/food_nutrition.csv` dataset documents 134 foods
with typical nutrient values per 100 g. The nutrition engine matches the
predicted food label to a dataset entry, scales values to the selected
portion, and provides context against profile-based targets.

The Nutrition Health Score is a transparent rule-based heuristic from
0--100, not a clinical measure. Nutrition values vary by ingredients,
recipe, cooking method, oil, salt, and serving size. The app should be
treated as an estimation tool, not a substitute for weighing or
laboratory analysis.

## 📈 Predictive Nutrition Intelligence Engine (PNIE)

PNIE uses an XGBoost regressor to estimate the user's average daily
calorie intake for the next seven days from their saved meal history.

-   Features are built from historical windows, with safeguards against
    using future data.
-   Evaluation uses a chronological hold-out rather than a random split.
-   MAE/RMSE and a simple baseline are used to contextualise model
    performance.
-   When history is too short or too sparse, the app should show
    descriptive trends or explain why a forecast is unavailable instead
    of fabricating a prediction.
-   Forecasts are stored only when the user explicitly selects the save
    action.

A forecast describes a pattern in recorded meals; it cannot predict
future events or guarantee future calorie intake.

## 🤖 Gemini AI Coach

The optional AI Coach uses Gemini to generate a short wellness-oriented
message from a snapshot of the user's profile, saved-meal summaries, and
available forecast information.

-   The request is made only when the user presses the relevant button.
-   The app is designed to display the information included in the
    request.
-   The user's name, food photos, database credentials, and other users'
    records should not be included in the coach prompt.
-   Responses can be generic or inaccurate and are not medical advice.
-   Quota, network, configuration, and API errors may prevent a
    response.

Set `GEMINI_API_KEY` in `.env` locally or in Streamlit Community Cloud
secrets for deployment. Keep API keys private.

## ☁️ Deployment

The app can be deployed to [Streamlit Community
Cloud](https://share.streamlit.io).

1.  Push the repository to GitHub.
2.  Create a new app in Streamlit Community Cloud and select this
    repository.
3.  Set the app's **Main file path** to `NutriVision-AI/app.py` if
    deploying from the repository root, or `app.py` if the app root is
    set to `NutriVision-AI`.
4.  Select a Python version compatible with the dependencies (Python
    3.11 or 3.12 is recommended by the supplied project notes).
5.  Add the required secrets under the app's settings.

Example Streamlit secrets (TOML):

``` toml
DATABASE_URL = "postgresql+psycopg2://USER:PASSWORD@HOST/DATABASE?sslmode=require"
GEMINI_API_KEY = "your-gemini-api-key"
```

Before deploying, verify that `.env`, local database files, personal
data, and credentials are excluded from Git. Test the deployed app with
a non-sensitive demo profile before using real information.

## ⚠️ Limitations

-   **No full authentication system:** profile selection is a project
    convenience, not secure user authentication. Do not treat a public
    deployment as a private health-data service.
-   **Single-item recognition:** mixed meals and multiple foods in one
    photo are not independently detected.
-   **Dataset coverage:** the recognition classes and nutrition dataset
    cannot represent every regional dish or recipe.
-   **Estimated values:** calories and nutrients are approximate and
    depend on recipe and portion accuracy.
-   **Forecast uncertainty:** PNIE depends on consistent historical
    logging and may be unavailable or unreliable with limited data.
-   **AI-generated text:** Gemini can produce incomplete, generic, or
    inaccurate suggestions.
-   **Time handling:** the project documentation notes that meal-day
    aggregation uses UTC rather than a per-user local timezone.
-   **Database operations:** schema creation does not automatically
    migrate existing columns; use migrations or a carefully managed
    development reset when schemas change.

## 🔒 Privacy and security

-   Keep `.env`, API keys, database passwords, and
    `.streamlit/secrets.toml` out of version control.
-   Review `.gitignore` before pushing model artifacts, local databases,
    or generated data.
-   The app design does not require storing the original food photo as a
    meal record; meal data is saved only after explicit user action.
-   Do not use the application to store sensitive personal information
    in an unauthenticated public deployment.
-   If a secret has ever been committed to Git, removing it in a later
    commit is not enough: revoke or rotate it.

## 📌 Disclaimer

NutriVision AI provides general wellness information for educational and
project-demonstration purposes. It does not diagnose, treat, cure, or
prevent disease and is not a substitute for a qualified physician or
dietitian. Nutrition figures are estimates, forecasts can be wrong, and
AI-generated text may be inaccurate. Use professional advice for
personal medical or dietary decisions.

------------------------------------------------------------------------

**Built as a final-year / NSP project.**
