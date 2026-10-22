# Job Posting Analytics

A Python and Flask web application that turns job-posting CSV files into insights about job roles, technical skills, and monthly skill trends. It combines exploratory data analysis, NLP, machine learning, optional time-series forecasting, and optional LLM-generated explanations in one dashboard.

> **Note:** The application analyzes datasets supplied by the user. It does not scrape live LinkedIn data. The included `sample_jobs.csv` is synthetic sample data, not real LinkedIn data.

## Features

- **CSV upload and data quality:** Upload a job-posting dataset and review row counts, columns, missing values, and duplicates.
- **Exploratory data analysis:** Explore job titles, companies, locations, and other available fields with summary statistics and charts.
- **NLP and skill extraction:** Clean job-description text and identify skills using a dictionary of skills and aliases.
- **Job-role classification:** Use TF-IDF text features and Logistic Regression to classify descriptions into six broad role categories.
- **Skill trend analysis:** Calculate monthly skill counts, share, growth, and a role–skill summary.
- **Optional skill-volume forecasting:** Use a TensorFlow/Keras LSTM to estimate future monthly posting counts for a selected skill when enough history is available. The workflow also compares the model with a simple previous-month baseline.
- **Optional AI insights:** Use an LLM API to explain computed results in natural language. The LLM is an explanation layer; the application's data-analysis and ML code computes the numerical results.

## Technology Stack

| Area | Tools |
|---|---|
| Backend | Python, Flask |
| Data analysis | Pandas, NumPy |
| Visualization | Matplotlib, HTML, CSS |
| NLP | Text preprocessing, tokenization, stopword removal, rule-based skill matching |
| Machine learning | Scikit-learn, TF-IDF, Logistic Regression |
| Forecasting | TensorFlow / Keras LSTM, MinMaxScaler |
| AI explanations | OpenAI-compatible Chat Completions API via `requests` |
| Frontend | HTML5, CSS, Jinja2 templates |

## How It Works

```text
Upload CSV
   ↓
Validate and analyze data
   ↓
Clean job-description text and extract skills
   ↓
Classify broad job roles
   ↓
Calculate monthly skill trends
   ↓
Optionally forecast a selected skill's posting volume
   ↓
Optionally generate an LLM explanation
   ↓
View results in the Flask dashboard
```

## Project Structure

```text
job-posting-analytics/
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── sample_jobs.csv
├── generate_sample_data.py
├── test_migration.py
├── src/
│   ├── data_analysis.py
│   ├── nlp_processing.py
│   ├── skill_extraction.py
│   ├── job_classifier.py
│   ├── trend_analysis.py
│   ├── forecasting.py
│   └── llm_service.py
├── templates/
│   ├── index.html
│   └── results.html
├── static/
│   ├── style.css
│   └── charts/
├── data/
│   ├── raw/
│   └── processed/
├── models/
└── uploads/
```

Generated files and empty folders may appear after the application runs; they are not all required to be present in a fresh clone.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Shashank07k/job-posting-analytics.git
cd job-posting-analytics
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Use dependency versions compatible with your installed Python version. If installation fails, review the package versions in `requirements.txt`, especially the TensorFlow and NumPy versions.

### 4. Configure optional LLM insights

LLM insights are optional. To configure them, copy the example environment file.

**Windows PowerShell**

```powershell
Copy-Item .env.example .env
```

Open `.env` and set the values required by the LLM provider configured in `src/llm_service.py`. The example configuration uses these variable names:

```dotenv
LLM_ENABLED=false
LLM_API_KEY=your_api_key_here
LLM_MODEL=your_model_name
LLM_API_URL=https://api.example.com/v1/chat/completions
LLM_TIMEOUT=30
```

Set `LLM_ENABLED=true` only after configuring a valid endpoint, model, and API key. The URL above is a placeholder; use the endpoint required by your provider.

**Never commit your real `.env` file or API keys.** Keep secrets server-side and use `.env.example` only for placeholder values.

### 5. Run the application

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

## Using the Dashboard

1. Open the application in your browser.
2. Upload `sample_jobs.csv` or another compatible CSV file.
3. Select **Analyze Dataset**.
4. Review data quality, exploratory analysis, extracted skills, role classification, and skill trends.
5. Select a skill to request a forecast, provided the dataset has enough monthly history.
6. Enter a job description to try role classification.
7. Generate AI insights if the LLM feature is configured and enabled.

## Dataset Format

Upload a CSV file. These columns are useful for the corresponding features:

| Column | Used for |
|---|---|
| `job_title` | Role-label mapping and job-title analysis |
| `job_description` | NLP, skill extraction, and role classification |
| `date_posted` | Monthly trends and forecasting |
| `company` | Company-level analysis |
| `location` | Location-level analysis |
| `employment_type` | Employment-type distribution, when present |
| `experience_level` | Experience-level distribution, when present |
| `salary` | Numerical summaries, when present |

Some fields may be optional for individual analyses, but `job_description`, `job_title`, and `date_posted` are important for the full workflow. Dates should be valid and consistent enough to support monthly grouping.

The included `sample_jobs.csv` is synthetic demonstration data. For meaningful trend or forecast analysis, use a larger, suitable dataset covering enough months. Forecast output is an estimate based on historical posting counts, not a guarantee of future hiring demand.

## Machine Learning Details

### Job-role classification

The application maps job titles to broad labels using rules, converts cleaned descriptions into TF-IDF features, and trains a Logistic Regression classifier. The six target categories are:

- Software Engineer
- Data Scientist
- Data Analyst
- AI/ML Engineer
- DevOps/Cloud Engineer
- Cybersecurity Engineer

The implementation reports classification metrics such as accuracy, precision, recall, F1-score, and a confusion matrix where available.

### Skill trend analysis

For skills detected in job descriptions, the application calculates monthly posting counts and trend-related metrics. Results depend on the available dates and on the skills included in the application's dictionary.

### Forecasting

The LSTM forecasts the monthly job-posting count for one selected skill using a short history window. The implementation uses a chronological split and compares results with a simple previous-month baseline. Forecasts can be unreliable when data is limited, irregular, or not representative; they should be treated as estimates rather than predictions of guaranteed demand.

## Testing

Run the project's test script from the repository root:

```bash
python test_migration.py
```

The test script covers several data-processing, NLP, classification, trend, forecasting, and LLM-service paths. Some TensorFlow-specific behavior depends on TensorFlow being installed. The LLM tests use test/mocked responses rather than requiring a live API call. Test results may vary with the local environment and installed dependency versions.

## Limitations

- Skill extraction uses a predefined dictionary and may miss skills that are not listed or recognized by its aliases.
- Job-title mapping is rule-based, so unusual titles may be assigned incorrectly or not mapped.
- Forecasting uses historical posting counts and does not model external market factors.
- Results depend on the quality, date coverage, and representativeness of the uploaded dataset.
- LLM-generated text can be inaccurate; use it as an explanation of calculated results, not as a source of numerical truth.
- The project processes uploaded data and does not scrape live LinkedIn listings.

## Possible Future Improvements

- Add more skills and improve job-title normalization.
- Evaluate against larger, real-world datasets with suitable usage rights.
- Add exportable reports and additional chart options.
- Improve model evaluation and handle more types of irregular time-series data.
- Deploy with production-grade upload limits, logging, and security controls.

## Project Summary

**Job Posting Analytics** demonstrates how a Python web application can combine data analysis, NLP, machine learning, time-series modeling, and optional LLM explanations to explore job-posting datasets through a single dashboard.
