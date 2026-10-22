# LinkedIn Job Trend Analysis & Prediction

## Overview
LinkedInJobTrendAI is a Flask-based analytics dashboard for uploaded job-posting CSV datasets.
It combines data analysis, NLP, machine learning, deep learning, and optional LLM-based explanation
in one end-to-end workflow:

Upload CSV → Analyze data → Extract skills → Classify roles → Build trends → Forecast skill volume → Explain results

The project is intentionally kept simple enough for a fresher-level Accenture interview.

## Problem Statement
Raw job-posting datasets are difficult to understand row by row. A student or recruiter may want to know:
- which roles appear most often,
- which skills are most common,
- how those skills are changing over time,
- how job descriptions relate to broad job-role categories,
- and what the recent historical pattern suggests for the next few months.

## Objective
Build a clean Python + Flask application where a user uploads a compatible CSV dataset and the application:
1. analyzes the dataset structure and quality,
2. cleans job-description text,
3. extracts important technical skills,
4. classifies descriptions into broad job roles,
5. builds monthly skill trends and engineered features,
6. forecasts future mention volume for one selected skill,
7. and optionally generates AI-written explanations of the computed results.

## Key Features
- CSV upload with safe filenames and size checks
- Exploratory data analysis with Pandas + Matplotlib
- NLP preprocessing and dictionary-based skill extraction
- Scikit-learn job-role classification (TF-IDF + Logistic Regression)
- Monthly skill-trend analysis and feature engineering
- TensorFlow/Keras LSTM forecasting with naive baseline comparison
- Optional LLM-generated explanation layer
- Consolidated Flask dashboard for all project phases

## Technology Stack
Only technologies actually used in the codebase are listed here:

- **Backend:** Flask
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib
- **NLP:** Python regex + tokenization + stopword removal + light rule-based lemmatization
- **Machine Learning:** Scikit-learn (`TfidfVectorizer`, `LogisticRegression`, evaluation metrics)
- **Deep Learning:** TensorFlow / Keras (`LSTM(32)`, `Dense(1)`)
- **Forecast Scaling:** Scikit-learn `MinMaxScaler`
- **Model Persistence:** Joblib (Phase 3 classifier)
- **LLM Integration:** `requests` + OpenAI-compatible Chat Completions API
- **Frontend:** HTML5, CSS, Jinja2 templates

## System Architecture
```text
         USER
           │
           ▼
    HTML5 + CSS UI
           │
           ▼
        Flask
           │
   ┌───────┼────────┐
   ▼       ▼        ▼
Pandas    NLP    Scikit-learn
   │       │        │
   └───────┼────────┘
           ▼
    Trend Analysis
           │
           ▼
    TensorFlow LSTM
           │
           ▼
      LLM Insights
           │
           ▼
    Results Dashboard
```

Important distinction:
- **Pandas / Scikit-learn / TensorFlow** compute the numerical results.
- **The LLM does not compute the numbers.** It only explains already computed results.

## Project Structure
```text
LinkedInJobTrendAI/
│
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── sample_jobs.csv
├── test_migration.py
├── generate_sample_data.py
│
├── uploads/
├── models/
├── data/
│   ├── raw/
│   └── processed/
│       └── skill_trends.csv
│
├── src/
│   ├── __init__.py
│   ├── data_analysis.py
│   ├── nlp_processing.py
│   ├── skill_extraction.py
│   ├── job_classifier.py
│   ├── trend_analysis.py
│   ├── forecasting.py
│   └── llm_service.py
│
├── templates/
│   ├── index.html
│   └── results.html
│
└── static/
    ├── style.css
    └── charts/
```

## How the Application Works
1. User opens the Flask home page.
2. User chooses or drags a `.csv` file into the upload area.
3. Flask receives the file with `request.files['dataset']`.
4. The file is validated (extension, empty file, malformed CSV, safe filename).
5. Pandas reads the CSV.
6. The application runs Phase 1 → Phase 6 logic in sequence.
7. Flask renders one consolidated dashboard using Jinja2.
8. The user can also:
   - choose a skill and run a forecast,
   - paste a job description and predict a broad job role,
   - click a button to generate AI-written insights.

## Dataset Format
The application expects an uploaded **CSV** file.

Example:
```csv
job_title,company,location,job_description,date_posted,employment_type,experience_level,salary
Data Scientist,Example Company,Bangalore,"Python, SQL, Machine Learning",2026-01-15,Full-time,Mid,1200000
```

Supported/expected fields:
- **job_title** — useful for Phase 3 role classification labels
- **company** — used in EDA
- **location** — used in EDA
- **job_description** — useful for Phase 2 NLP and Phase 4/5 skill-based analysis
- **date_posted** — useful for Phase 4 trends and Phase 5 forecasting
- **employment_type** — optional EDA distribution
- **experience_level** — optional EDA distribution
- **salary** — optional numerical statistics

Important:
- Not every column is strictly required.
- The dashboard degrades gracefully when a column is missing.
- `sample_jobs.csv` is **synthetic development data** and is **not scraped LinkedIn data**.

## Phase 1 — Data Analysis
Implemented in `src/data_analysis.py`.

What it does:
- row/column counts,
- missing-value analysis,
- duplicate detection,
- descriptive numerical statistics,
- top job titles, companies, and locations,
- employment-type and experience-level distributions,
- monthly posting trend chart when `date_posted` exists.

Libraries used:
- Pandas
- NumPy
- Matplotlib

## Phase 2 — NLP & Skill Extraction
Implemented in `src/nlp_processing.py` and `src/skill_extraction.py`.

Text preprocessing steps:
- lowercase conversion,
- URL removal,
- punctuation cleanup,
- whitespace normalization,
- tokenization,
- stopword removal,
- light rule-based lemmatization.

Skill extraction approach:
- fixed dictionary of ~28 skills,
- simple aliases such as `ml → Machine Learning`, `reactjs → React`, `aws → AWS`,
- count each skill at most once per job description.

Why dictionary-based extraction was chosen:
- simple,
- transparent,
- easy to modify,
- easy to explain in an interview.

## Phase 3 — Job Role Classification
Implemented in `src/job_classifier.py`.

Goal:
Predict one broad role from a job description.

Target categories:
- Software Engineer
- Data Scientist
- Data Analyst
- AI/ML Engineer
- DevOps/Cloud Engineer
- Cybersecurity Engineer

Approach:
- map `job_title` to one of the six categories using simple regex rules,
- clean `job_description`,
- convert text to TF-IDF features,
- train `LogisticRegression`,
- evaluate with test accuracy, precision, recall, F1-score, and confusion matrix.

Why TF-IDF:
> TF-IDF converts text into numerical features based on how important words are within the document and across the dataset.

Why Logistic Regression:
> A classification algorithm that learns relationships between the TF-IDF features and the job-role categories.

## Phase 4 — Skill Trend Analysis
Implemented in `src/trend_analysis.py`.

Purpose:
Turn skill mentions into time-based analytical features.

Computed metrics:
- **Skill Frequency** = number of postings mentioning the skill
- **Skill Share (%)** = (Skill Job Count / Total Job Postings) × 100
- **Growth Rate (%)** = ((Recent − Previous) / Previous) × 100

Trend classification rules:
- Growth > +10% → **Growing**
- Between −10% and +10% → **Stable**
- Growth < −10% → **Declining**

Monthly aggregation is used by default because it is easy to explain and directly feeds Phase 5.

Role–skill matrix:
- percentage of postings in each broad role mentioning each top skill.

Phase 4 output file:
- `data/processed/skill_trends.csv`

## Phase 5 — LSTM Forecasting
Implemented in `src/forecasting.py`.

Purpose:
Forecast the future **job_count** for one selected skill using Phase 4's monthly time-series data.

Target variable:
- `job_count`

Why monthly aggregation:
- reduces noise,
- keeps the sequence simple,
- matches the Phase 4 engineered feature file,
- easy to explain.

Lookback window:
- `LOOKBACK = 3`
- meaning: the LSTM uses the previous 3 months to estimate the next month.

Train / test methodology:
- chronological 80/20 split,
- **no shuffling** (time series cannot be randomly shuffled),
- `MinMaxScaler` fitted on the training portion only to avoid leakage.

LSTM architecture:
- `LSTM(32)`
- `Dense(1)`
- Adam optimizer
- MSE loss
- modest epochs with optional EarlyStopping

Evaluation metrics:
- **MAE** — Average absolute difference between actual and predicted values.
- **RMSE** — Similar to MAE, but penalizes larger errors more strongly.

Baseline:
- naive forecast = previous month's job count.
- both baseline and LSTM are evaluated on the same held-out test months.

Future forecast process:
- recursive 3-month prediction,
- inverse-scale to original counts,
- clip at zero,
- round for display.

## Phase 6 — LLM Insights
Implemented in `src/llm_service.py`.

Purpose:
Add an interpretation layer after all numerical analysis is complete.

The LLM receives:
- dataset summary,
- top skills,
- observed growth/decline,
- role–skill matrix summary,
- classifier metrics,
- forecast summary.

The LLM does **not** receive the raw CSV.

Why not send the whole dataset:
- unnecessary cost,
- slower,
- exposes more raw text than needed,
- increases hallucination risk.

What the LLM produces:
- Overall Summary
- Key Trends
- Forecast Interpretation
- Role–Skill Insight
- Limitations

Important rule:
- The LLM is **not** the source of the numerical results.
- It only explains results already computed by the application.

## Phase 7 — Final Application
This final phase focuses on:
- polished home page,
- consistent dashboard sections,
- summary cards,
- better upload UX,
- graceful empty/error states,
- improved navigation,
- improved documentation,
- final interview readiness.

No new ML or DL model is added in Phase 7.

## Installation
```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration
Optional `.env` setup for Phase 6 AI insights:
```bash
cp .env.example .env
```

Example `.env` values:
```env
LLM_ENABLED=false
LLM_API_KEY=your_api_key_here
LLM_MODEL=gpt-4o-mini
LLM_API_URL=https://api.openai.com/v1/chat/completions
LLM_TIMEOUT=30
```

Notes:
- Set `LLM_ENABLED=true` to enable AI insights.
- Keep `.env` out of version control.
- The API key is never entered in the browser.

## Running the Project
```bash
python app.py
```
Then open:
```text
http://127.0.0.1:5000
```

## Testing
Automated tests are in:
```bash
python test_migration.py
```

The tests cover:
- Phase 1: CSV loading, missing values, duplicates, EDA
- Phase 2: cleaning, stopwords, skill extraction, repeated skill handling
- Phase 3: title mapping, train/test split, classifier metrics, prediction
- Phase 4: frequency, share, growth, trend labels, role-skill matrix, engineered features
- Phase 5: sequence creation, chronological split, scaling, baseline, MAE, RMSE, insufficient-history handling
- Phase 6: disabled LLM, missing key, mocked success, invalid JSON, API failure

Important:
- Phase 6 tests do **not** call a real API.
- Phase 5's full TensorFlow test runs only when TensorFlow is installed in the environment.

## Example Workflow
1. Open the Flask home page.
2. Upload `sample_jobs.csv` or your own compatible CSV.
3. Click **Analyze Dataset**.
4. Review:
   - Dataset Overview
   - Data Quality
   - Exploratory Analysis
   - NLP & Skills
   - Job Role Classification
   - Skill Trend Analysis
5. Choose a skill and click **Forecast**.
6. Paste a new job description and click **Predict Job Role**.
7. Optionally click **Generate AI Insights**.

## Limitations
- `sample_jobs.csv` is synthetic and spans only a small date range, so it mainly demonstrates the insufficient-history path for forecasting.
- Dictionary-based skill extraction only detects listed skills and aliases.
- Title mapping is rule-based and may skip unusual titles.
- LSTM forecasting uses only the monthly `job_count` series; it does not model external market factors.
- LLM wording may vary and must be treated as explanation, not ground truth.
- The project analyzes uploaded datasets only and does **not** scrape LinkedIn live.

## Future Improvements
These are ideas after the current 7-phase scope and are **not implemented**:
- larger real-world datasets spanning longer periods,
- better title normalization,
- more robust skill taxonomy,
- better chart export/reporting,
- deployment hardening.

## Interview Questions & Answers

### 1. Explain the project in 60 seconds.
It is a Flask web application where I upload a job-posting CSV, analyze the data with Pandas, clean job-description text, extract skills, classify job roles using TF-IDF plus Logistic Regression, build monthly skill trends, forecast one skill's future posting volume using a small TensorFlow LSTM, and optionally generate AI-written insights that explain the computed results.

### 2. What problem does it solve?
It turns a raw job-posting dataset into understandable insights about roles, skills, trends, and short-term estimated posting volume.

### 3. Why Pandas?
Because Pandas makes CSV reading, missing-value handling, grouping, statistics, and time-based aggregation simple.

### 4. Why NumPy?
NumPy supports numerical operations used under Pandas, Scikit-learn, and TensorFlow.

### 5. Why NLP?
Because important information like required skills is written inside text, not just in structured columns.

### 6. How does skill extraction work?
The application cleans the description text and matches words/phrases against a small skill dictionary with aliases.

### 7. Why TF-IDF?
It converts text into numeric features so a machine-learning model can learn which words are associated with each job role.

### 8. Why Logistic Regression?
It is a fast, understandable baseline for text classification and easy to explain in an interview.

### 9. How did you evaluate the classifier?
Using an 80/20 train/test split and reporting accuracy, precision, recall, F1-score, and a confusion matrix.

### 10. What is feature engineering?
It is turning raw data into useful numerical inputs, such as monthly skill counts, share, previous-month count, and growth rate.

### 11. Why monthly aggregation?
It reduces noise and creates a simple time series that is easy to explain and use for forecasting.

### 12. What is an LSTM?
LSTM is a recurrent neural network designed to learn patterns from sequential data such as monthly job-count history.

### 13. Why compare LSTM with a naive baseline?
Because a more complex model should prove that it improves on a very simple forecast like "next month = last month".

### 14. What are MAE and RMSE?
MAE is average absolute error. RMSE is similar but penalizes larger mistakes more strongly.

### 15. Why does the LLM not perform the forecasting?
Because forecasting is a numerical modeling task handled by TensorFlow; the LLM only explains the results.

### 16. How is the API key protected?
It is stored server-side in environment variables / `.env`, never entered in the browser and never rendered into HTML.

### 17. What happens if the LLM API fails?
The dashboard still works; only the insights section shows a friendly error message.

### 18. What are the project's main limitations?
It depends on the uploaded dataset, uses dictionary-based skills, uses a simple role mapping, and gives model estimates rather than guaranteed market predictions.

### 19. What is prompt engineering?
It is writing instructions that guide the LLM to explain only the supplied structured results in a predictable JSON format.

### 20. What is the difference between an ML prediction and an LLM explanation?
An ML prediction is a computed numerical/class result from Scikit-learn or TensorFlow; an LLM explanation is text describing those results.
