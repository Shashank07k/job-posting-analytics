"""Flask entry point for LinkedInJobTrendAI."""

import logging
import os
import uuid
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename

from src.data_analysis import perform_eda, generate_bar_chart
from src.skill_extraction import analyze_skills_in_series
from src.job_classifier import train_job_classifier, predict_job_role
from src.trend_analysis import analyze_skill_trends, render_selected_skill
from src.forecasting import get_forecastable_skills, run_skill_forecast
from src.llm_service import build_analysis_summary, generate_insights, is_llm_enabled

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'linkedin-job-trend-ai-fresher-secret')

# Tracebacks go to the terminal; users only see friendly messages.
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
CHARTS_FOLDER = os.path.join(BASE_DIR, 'static', 'charts')
SAMPLE_DATA_PATH = os.path.join(BASE_DIR, 'sample_jobs.csv')

ALLOWED_EXTENSIONS = {'csv'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHARTS_FOLDER, exist_ok=True)

# Single-user, in-memory state for the most recent analysis (no database).
LATEST_ANALYSIS = {}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def build_summary_cards(state):
    """Read already-computed results for the dashboard cards; never recalculates."""
    state = state or {}
    eda = state.get('eda') or {}
    ml = state.get('ml') or {}
    trend = state.get('trend') or {}
    forecast_state = state.get('forecast') or {}
    forecast_result = forecast_state.get('result') or {}

    top_skill = 'N/A'
    total_skills = 0
    if trend.get('available') and trend.get('frequency_table'):
        total_skills = trend.get('tracked_skills') or len(trend['frequency_table'])
        top_skill = trend['frequency_table'][0]['skill']
    elif (state.get('nlp') or {}).get('available') and state['nlp'].get('top_skills'):
        total_skills = state['nlp'].get('unique_skills_found') or len(state['nlp']['top_skills'])
        top_skill = state['nlp']['top_skills'][0]['name']

    top_role = 'N/A'
    if eda.get('top_titles'):
        top_role = str(eda['top_titles'][0][0])

    if ml.get('status') == 'trained':
        classifier_status = f"{ml['metrics']['accuracy'] * 100:.0f}% test accuracy"
    else:
        classifier_status = 'Unavailable'

    if forecast_result.get('status') in ('ok', 'ok_constant'):
        forecast_status = f"Ready ({forecast_result.get('skill')})"
    elif forecast_state.get('skills'):
        forecast_status = 'Available - select a skill'
    else:
        forecast_status = 'Needs monthly dates'

    return {
        'total_jobs': int(eda.get('total_rows') or 0),
        'total_columns': int(eda.get('total_cols') or 0),
        'total_skills': int(total_skills or 0),
        'top_skill': top_skill,
        'top_role': top_role,
        'classifier_status': classifier_status,
        'forecast_status': forecast_status,
    }


def render_dashboard(prediction=None, user_input_desc=None):
    return render_template(
        'results.html',
        eda=LATEST_ANALYSIS.get('eda'),
        nlp=LATEST_ANALYSIS.get('nlp'),
        nlp_chart=LATEST_ANALYSIS.get('nlp_chart'),
        ml=LATEST_ANALYSIS.get('ml'),
        trend=LATEST_ANALYSIS.get('trend'),
        forecast=LATEST_ANALYSIS.get('forecast'),
        insights=LATEST_ANALYSIS.get('insights'),
        cards=build_summary_cards(LATEST_ANALYSIS),
        prediction=prediction,
        user_input_desc=user_input_desc,
    )


@app.route('/', methods=['GET'])
def index():
    error = request.args.get('error')
    return render_template('index.html', error=error)


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'dataset' not in request.files:
        flash('No file part in the request.')
        return redirect(url_for('index', error='No file part found in request.'))

    file = request.files['dataset']

    if file.filename == '':
        return redirect(url_for('index', error='No file selected. Please choose a CSV dataset.'))

    if not allowed_file(file.filename):
        return redirect(url_for('index', error='Unsupported file type. Please upload a .csv file only.'))

    # Prefix with a UUID so user-supplied names can never collide or traverse paths.
    original_name = secure_filename(file.filename) or 'dataset.csv'
    safe_filename = f"{uuid.uuid4().hex}_{original_name}"
    saved_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)

    try:
        file.save(saved_path)

        if os.path.getsize(saved_path) == 0:
            os.remove(saved_path)
            return redirect(url_for('index', error='The uploaded CSV file is empty (0 bytes).'))

        try:
            df = pd.read_csv(saved_path, keep_default_na=True)
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(saved_path, encoding='latin1', keep_default_na=True)
            except Exception as e:
                return redirect(url_for('index', error=f'Encoding error: Could not decode CSV. Please upload UTF-8 text ({str(e)}).'))
        except pd.errors.EmptyDataError:
            return redirect(url_for('index', error='The uploaded CSV has no rows or data.'))
        except pd.errors.ParserError as e:
            return redirect(url_for('index', error=f'Malformed CSV error: {str(e)}'))

        if len(df) == 0:
            return redirect(url_for('index', error='Dataset contains headers but 0 data rows.'))

        eda_results = perform_eda(df, filename=original_name)

        nlp_results = None
        nlp_chart_path = None
        desc_col = None
        for col in df.columns:
            if col.strip().lower() in {'job_description', 'description', 'job_desc'}:
                desc_col = col
                break

        if desc_col:
            nlp_results = analyze_skills_in_series(df[desc_col])
            if nlp_results['top_skills']:
                skill_chart_items = [(s['name'], s['count']) for s in nlp_results['top_skills'][:10]]
                nlp_chart_path = generate_bar_chart(
                    skill_chart_items,
                    'top_skills.png',
                    'Top In-Demand Technical Skills (Phase 2)',
                    xlabel='Postings Mentioning Skill',
                    color='#28785e'
                )

        ml_results = train_job_classifier(df)

        trend_results = analyze_skill_trends(df, filename=original_name)

        # The LSTM trains on demand in /forecast so uploads stay fast.
        try:
            forecast_skills = get_forecastable_skills()
        except Exception:
            forecast_skills = []
        forecast_state = {'skills': forecast_skills, 'result': None, 'selected': None}

        # The LLM only runs when the user explicitly requests insights.
        insights_state = {'result': None, 'enabled': is_llm_enabled()}

        global LATEST_ANALYSIS
        LATEST_ANALYSIS = {
            'eda': eda_results,
            'nlp': nlp_results,
            'nlp_chart': nlp_chart_path,
            'ml': ml_results,
            'trend': trend_results,
            'forecast': forecast_state,
            'insights': insights_state
        }

        return render_dashboard()

    except Exception:
        app.logger.exception('Unexpected error while analyzing uploaded dataset.')
        return redirect(url_for(
            'index',
            error='Unable to analyze this dataset. Please check that the CSV contains valid job-posting data.'
        ))

    finally:
        if os.path.exists(saved_path):
            try:
                os.remove(saved_path)
            except Exception:
                pass


@app.route('/predict', methods=['POST'])
def predict():
    user_text = request.form.get('job_description', '').strip()

    prediction_result = predict_job_role(user_text)

    global LATEST_ANALYSIS
    if not LATEST_ANALYSIS:
        flash('Please upload a dataset first to view analysis and train the model.')
        return redirect(url_for('index', error='Please upload a dataset first to train the classifier.'))

    return render_dashboard(prediction=prediction_result, user_input_desc=user_text)


@app.route('/skill-trend', methods=['GET'])
def skill_trend():
    """Regenerate only the selected skill's detail; trends are not recomputed."""
    global LATEST_ANALYSIS
    if not LATEST_ANALYSIS:
        return redirect(url_for('index', error='Please upload a dataset first to view skill trends.'))

    selected = request.args.get('skill', '').strip() or None
    trend_results = LATEST_ANALYSIS.get('trend')

    if trend_results is None or not trend_results.get('available') or not trend_results.get('time_series_available'):
        return render_dashboard()

    trend_results, _ = render_selected_skill(trend_results, selected)
    LATEST_ANALYSIS['trend'] = trend_results

    return render_dashboard()


@app.route('/forecast', methods=['GET'])
def forecast():
    global LATEST_ANALYSIS
    if not LATEST_ANALYSIS:
        return redirect(url_for('index', error='Please upload a dataset first to run a forecast.'))

    selected = request.args.get('skill', '').strip() or None
    forecast_state = LATEST_ANALYSIS.get('forecast') or {'skills': [], 'result': None, 'selected': None}

    # skill_trends.csv may have changed since the upload.
    try:
        forecast_state['skills'] = get_forecastable_skills()
    except Exception:
        forecast_state['skills'] = forecast_state.get('skills', [])

    forecast_state['selected'] = selected
    forecast_state['result'] = run_skill_forecast(selected)
    LATEST_ANALYSIS['forecast'] = forecast_state

    return render_dashboard()


@app.route('/insights', methods=['POST'])
def insights():
    """Send structured results (never the CSV) to the LLM; the API key stays server-side."""
    global LATEST_ANALYSIS
    if not LATEST_ANALYSIS:
        return redirect(url_for('index', error='Please upload a dataset first to generate AI insights.'))

    summary = build_analysis_summary(LATEST_ANALYSIS)
    result = generate_insights(summary)

    insights_state = {'result': result, 'enabled': is_llm_enabled()}
    LATEST_ANALYSIS['insights'] = insights_state

    return render_dashboard()


@app.route('/download-sample', methods=['GET'])
def download_sample():
    if os.path.exists(SAMPLE_DATA_PATH):
        return send_from_directory(BASE_DIR, 'sample_jobs.csv', as_attachment=True)
    return redirect(url_for('index', error='Sample dataset not found.'))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
