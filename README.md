<div align="center">

# 🥗 NutriVision AI

### AI-Powered Food Recognition & Predictive Nutrition Analysis

Snap a photo of your meal and get instant nutrition insights, calorie trends, and AI-powered wellness coaching.

[![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/MobileNetV2-Food--101-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/Forecasting-XGBoost-189FDD)](https://xgboost.ai/)
[![Gemini](https://img.shields.io/badge/AI%20Coach-Gemini-8E75B2?logo=googlegemini&logoColor=white)](https://ai.google.dev/)

<br>

### 🎬 [▶️ Watch the Demo Video](https://www.kapwing.com/videos/6ae05b5f02f50f708c1f6cc6)

<sub>See NutriVision AI in action: food recognition, nutrition analysis, and forecasting.</sub>

</div>

---

## 📖 Overview

NutriVision AI is a final-year / NSP project that turns a food photo into practical nutrition insights. It combines food-image classification, a curated Indian-food nutrition dataset, meal tracking, calorie-trend forecasting, and optional AI-generated wellness coaching, all inside a Streamlit application.

> **Project status:** Phases 1 to 8 are complete. Model outputs and nutrition values are estimates, and the app clearly identifies its limitations.

## ✨ Highlights

| Feature | What it does |
|---|---|
| 📸 **Food Recognition** | Classifies a photo with a MobileNetV2 model trained on Food-101, showing confidence and a top-five shortlist |
| 🍛 **Indian Food Dataset** | Estimated nutrition for **134 foods** per 100 g, scaled to your portion |
| 🏆 **Health Score** | Transparent, rule-based score from **0 to 100** with a component breakdown |
| 💾 **Explicit Saving** | Nothing is stored until you press **Save Meal** |
| 📊 **Dashboard & History** | Daily totals, trends, macronutrient charts, filters, and CSV export |
| 🔮 **Calorie Forecasting** | XGBoost predicts your next week's average daily calories from logged history |
| 🤖 **Gemini AI Coach** | Optional wellness tips, sent only when you request them |
| 👤 **Profile Management** | Review, update, or explicitly delete your stored data |

## 🔄 How It Works

```text
Food photo
   ↓
Image validation & preprocessing
   ↓
MobileNetV2 / Food-101 classification
   ↓
Top prediction + confidence + top-five shortlist
   ↓
Food-name matching against nutrition dataset
   ↓
Portion-based estimates + Health Score
   ↓
You press "Save Meal"
   ↓
Database → Dashboard / Meal History / PNIE Forecast
                              ↓
                  Optional Gemini AI Coach
```

> The classifier predicts a single dominant dish, not every item on a plate. Uncertain matches are shown for you to confirm.

## 🧰 Tech Stack

| Area | Technologies |
|---|---|
| 🖥️ **Interface** | Python, Streamlit, custom CSS |
| 🧠 **Food Recognition** | MobileNetV2, Food-101, ONNX Runtime (optional PyTorch backend) |
| 📈 **Data & Charts** | Pandas, NumPy, Plotly, curated CSV dataset |
| 🔮 **Forecasting** | XGBoost |
| 🗄️ **Database** | SQLAlchemy, Neon PostgreSQL (SQLite for local development) |
| 🤖 **AI Coaching** | Google Gemini API (optional) |
| 🧪 **Testing** | pytest, Streamlit AppTest |
| ⚙️ **Configuration** | Environment variables, `.env`, Streamlit secrets |

## 📁 Project Structure

```text
NutriVision-AI/
├── app.py                  # Streamlit entry point
├── config.py               # Configuration layer
├── ai/                     # Gemini AI coach
├── assets/                 # CSS and ONNX model files
├── database/               # Connection, setup, and repository layer
├── food_recognition/       # Model loading, preprocessing, prediction
├── models/                 # Data models (user, meal, prediction, recommendation)
├── nutrition/              # Nutrition dataset, engine, and health score
├── pages/                  # Dashboard, Upload, History, Predictions, Coach, Profile
├── pnie/                   # Predictive Nutrition Intelligence Engine
├── tests/                  # Unit and end-to-end tests
└── utils/                  # Helpers
```

## 🚀 Getting Started

### Prerequisites

- **Python 3.10 to 3.12** (use the version supported by your installed dependencies)
- **Git**
- A **Google Gemini API key** (only for AI coaching)
- A **Neon PostgreSQL** database for persistent hosted use (SQLite works locally)

### 1️⃣ Clone the repository

```bash
git clone https://github.com/jaagam751-stack/nutrivision_enhanced_model_project.git
cd nutrivision_enhanced_model_project/NutriVision-AI
```

### 2️⃣ Create and activate a virtual environment

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Only need to prepare datasets or train models? See `requirements-training.txt`. Skip the training stack otherwise.

## 🔐 Configuration

Copy the example environment file:

```bash
# Windows PowerShell: Copy-Item .env.example .env
cp .env.example .env
```

Set the values you need in `.env`:

```dotenv
DATABASE_URL=postgresql+psycopg2://USER:PASSWORD@HOST/DATABASE?sslmode=require
GEMINI_API_KEY=your-gemini-api-key
```

| Variable | Required? | Purpose |
|---|---|---|
| `DATABASE_URL` | For persistent PostgreSQL | Neon/PostgreSQL connection string |
| `GEMINI_API_KEY` | Optional | Enables AI Coach. Get one from [Google AI Studio](https://aistudio.google.com/app/apikey) |

> 🔒 **Never commit** `.env`, database credentials, API keys, or `.streamlit/secrets.toml` to GitHub.

## 🗄️ Database Setup

### Option A: Neon PostgreSQL (recommended for deployment)

1. Create a project at [Neon](https://neon.tech).
2. Copy the connection string from the Neon dashboard.
3. Put it in `DATABASE_URL` inside `.env`.
4. Verify the connection and create the tables:

```bash
python database/init_db.py --check
python database/init_db.py
```

### Option B: SQLite for local development

```bash
python database/init_db.py --sqlite --sample-data
```

> A local SQLite file is not suitable for persistent production data on Streamlit Community Cloud.

### Optional sample data

```bash
python database/init_db.py --sample-data
```

This creates clearly flagged demo data for exploring the dashboard and forecasts.

> ⚠️ **Warning:** `python database/init_db.py --reset --yes` drops and recreates tables and can destroy stored data. Use it only on a disposable development database.

## ▶️ Run the Application

From the `NutriVision-AI` directory:

```bash
streamlit run app.py
```

Open the local URL Streamlit prints, usually **http://localhost:8501**.

### 🧭 Typical Workflow

1. **Create or load** a profile.
2. Open **Upload Meal** and upload a JPG, JPEG, or PNG.
3. Review the prediction, shortlist, nutrition match, portion, and estimates.
4. Press **Save Meal** only when you want to store it.
5. Explore the **Dashboard** and **Meal History**.
6. Open **Predictions** once enough history exists.
7. Add a Gemini key and open **AI Coach** for a recommendation.
8. Use **Profile & Data** to review or manage your records.

## 🧪 Run Tests

```bash
pytest -v
```

Tests cover profile validation, database operations, food recognition, nutrition calculations, health scoring, analytics, PNIE, Gemini error handling, and profile data operations.

End-to-end Streamlit tests run separately:

```bash
python tests/e2e_phase7_apptest.py
python tests/e2e_phase8_apptest.py
```

> These may write or delete records, so run them against a disposable local database.

## 🍽️ Food Recognition

- Uses a **MobileNetV2** classifier trained on **Food-101**
- Returns a predicted class, confidence score, and top-five shortlist
- **ONNX Runtime** is the lightweight inference path

**Keep in mind:**

- Classifies **one dominant food item** per image, not every item on a mixed plate
- Food-101 doesn't cover every Indian home-cooked dish. Dal, chapati, idli, poha, and paneer dishes may map to a visually similar class
- Confidence is the model's score, not a guarantee
- Uploaded photos are processed in memory and are not stored as meal records

## 🥦 Nutrition Analysis

The `nutrition/food_nutrition.csv` dataset documents **134 foods** with typical nutrient values per 100 g. The engine matches the predicted label to a dataset entry, scales values to your portion, and compares them with profile-based daily targets.

The **Nutrition Health Score** is a rule-based heuristic from 0 to 100, not a clinical measure. Actual values vary with ingredients, recipe, cooking method, oil, salt, and serving size. Treat the app as an estimation tool, not a substitute for weighing or laboratory analysis.

## 📈 Predictive Nutrition Intelligence Engine (PNIE)

PNIE uses an **XGBoost** regressor to estimate average daily calorie intake for the **next seven days** from your saved meal history.

- Features come from historical windows, with safeguards against future data leakage
- Evaluation uses a **chronological hold-out**, not a random split
- MAE and RMSE are reported alongside a simple baseline
- With too little or sparse history, the app shows descriptive trends or explains why a forecast is unavailable
- Forecasts are stored only when you explicitly save them

> A forecast describes a pattern in your recorded meals. It cannot predict future events or guarantee future intake.

## 🤖 Gemini AI Coach

The optional coach generates a short wellness-oriented message from a snapshot of your profile, saved-meal summaries, and forecast information.

- Requests are made **only when you press the button**
- The exact information sent is displayed in the app
- Your name, food photos, database credentials, and other users' records are **not** included in the prompt
- Responses can be generic or inaccurate and are **not medical advice**

## ☁️ Deployment

Deploy to [Streamlit Community Cloud](https://share.streamlit.io):

1. Push the repository to GitHub.
2. Create a new app and select this repository.
3. Set the **Main file path**:
   - `NutriVision-AI/app.py` if deploying from the repository root
   - `app.py` if the app root is set to `NutriVision-AI`
4. Choose **Python 3.11 or 3.12**.
5. Add secrets in the app settings:

```toml
DATABASE_URL = "postgresql+psycopg2://USER:PASSWORD@HOST/DATABASE?sslmode=require"
GEMINI_API_KEY = "your-gemini-api-key"
```

Before deploying, confirm `.env`, local databases, personal data, and credentials are excluded from Git. Test with a non-sensitive demo profile first.

## ⚠️ Limitations

| Area | Limitation |
|---|---|
| 🔑 **Authentication** | Profile selection is a convenience, not secure login. Don't treat a public deployment as a private health-data service |
| 🍱 **Recognition** | Mixed meals and multi-item photos aren't independently detected |
| 🌏 **Coverage** | Classes and dataset can't represent every regional dish or recipe |
| 📏 **Accuracy** | Calories and nutrients are approximate |
| 🔮 **Forecasts** | Depend on consistent logging and may be unreliable with limited data |
| 🤖 **AI text** | Gemini output may be incomplete, generic, or inaccurate |
| 🕒 **Time** | Meal-day aggregation uses UTC, not per-user local time |
| 🗃️ **Schema** | Table creation doesn't auto-migrate existing columns |

## 🔒 Privacy & Security

- Keep `.env`, API keys, database passwords, and `.streamlit/secrets.toml` out of version control
- Review `.gitignore` before pushing model artifacts, local databases, or generated data
- Meal data is saved only after an explicit user action
- Don't store sensitive personal information in an unauthenticated public deployment
- If a secret was ever committed, **rotate or revoke it**. Deleting it in a later commit isn't enough

## 📌 Disclaimer

NutriVision AI provides general wellness information for educational and project-demonstration purposes. It does not diagnose, treat, cure, or prevent disease, and it is not a substitute for a qualified physician or dietitian. Nutrition figures are estimates, forecasts can be wrong, and AI-generated text may be inaccurate. Consult a professional for personal medical or dietary decisions.

---

<div align="center">

**🎬 [Watch the Demo](https://www.kapwing.com/videos/6ae05b5f02f50f708c1f6cc6)** · Built as a final-year / NSP project

⭐ If you found this project useful, consider giving it a star!

</div>
