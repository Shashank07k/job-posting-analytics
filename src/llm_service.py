"""
LLM Insight Layer (Phase 6)

Purpose:
Turn the STRUCTURED results already computed by Phases 1-5 into a short,
human-readable explanation. The LLM is an INTERPRETATION layer only.

    Dataset -> EDA -> NLP/Skills -> ML Classification -> Trends -> LSTM Forecast
        -> small structured summary (dict)  -> LLM -> readable insights

The LLM does NOT classify roles, extract skills, forecast, or calculate any
number. Scikit-learn / Pandas / TensorFlow produce the numbers; the LLM only
explains them, and the prompt forbids it from inventing new ones.

What this module does:
1. is_llm_enabled()          -> reads LLM_ENABLED / LLM_API_KEY from the environment
2. build_analysis_summary()  -> compact dict built from LATEST_ANALYSIS (never the raw CSV)
3. build_insight_prompt()    -> system + user prompt asking for strict JSON output
4. call_llm_api()            -> one plain HTTPS request via `requests` (no SDK, no LangChain)
5. validate_llm_response()   -> checks required fields, normalises types
6. generate_insights()       -> the single entry point used by app.py; never raises

Provider: any OpenAI-compatible Chat Completions endpoint (default: OpenAI).
Configured only through environment variables:
    LLM_ENABLED  = true|false      (default false -> feature off, app still works)
    LLM_API_KEY  = <secret>        (never hardcoded, never sent to the browser)
    LLM_MODEL    = gpt-4o-mini     (default)
    LLM_API_URL  = https://api.openai.com/v1/chat/completions (default; override for
                   other OpenAI-compatible providers)
    LLM_TIMEOUT  = 30              (seconds)

No RAG, no vector database, no agents, no streaming, no chat memory.
"""

import json
import os

import requests

DEFAULT_MODEL = 'gpt-4o-mini'
DEFAULT_API_URL = 'https://api.openai.com/v1/chat/completions'
DEFAULT_TIMEOUT = 30
MAX_TOKENS = 700
TEMPERATURE = 0.2  # low temperature = stick to the supplied facts

REQUIRED_FIELDS = ('summary', 'key_trends', 'forecast_interpretation',
                   'role_skill_insight', 'limitations')

UNAVAILABLE_MESSAGE = 'AI insights unavailable. Configure an LLM API key to enable this feature.'


# ---------------------------------------------------------------------------
# 1. Configuration (environment variables only)
# ---------------------------------------------------------------------------
def _load_dotenv_if_present():
    """
    Tiny .env loader (KEY=VALUE lines) so a developer can keep the API key in
    a git-ignored file. Real environment variables always win over .env values.
    No extra dependency needed.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(base_dir, '.env')
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, 'r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        pass  # a broken .env must never break the app


_load_dotenv_if_present()


def get_llm_config():
    """Reads LLM settings from the environment. The key is never logged or returned to templates."""
    enabled_raw = os.environ.get('LLM_ENABLED', 'false').strip().lower()
    try:
        timeout = int(os.environ.get('LLM_TIMEOUT', DEFAULT_TIMEOUT))
    except ValueError:
        timeout = DEFAULT_TIMEOUT
    return {
        'enabled': enabled_raw in {'1', 'true', 'yes', 'on'},
        'api_key': os.environ.get('LLM_API_KEY', '').strip(),
        'model': os.environ.get('LLM_MODEL', DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        'api_url': os.environ.get('LLM_API_URL', DEFAULT_API_URL).strip() or DEFAULT_API_URL,
        'timeout': timeout,
    }


def is_llm_enabled(config=None):
    """True only when the feature flag is on AND a key exists."""
    config = config or get_llm_config()
    return bool(config['enabled'] and config['api_key'])


# ---------------------------------------------------------------------------
# 2. Structured summary (small, already-computed facts; never the raw CSV)
# ---------------------------------------------------------------------------
def build_analysis_summary(state):
    """
    Builds the compact dict sent to the LLM from the app's LATEST_ANALYSIS.

    Every value here was calculated earlier by Pandas / Scikit-learn / TensorFlow.
    Missing sections are simply left out (or marked unavailable) - nothing is invented.
    """
    state = state or {}
    eda = state.get('eda') or {}
    ml = state.get('ml') or {}
    trend = state.get('trend') or {}
    forecast_state = state.get('forecast') or {}
    forecast = forecast_state.get('result') or {}

    summary = {
        'dataset': {
            'filename': eda.get('filename'),
            'total_jobs': int(eda.get('total_rows') or 0),
            'duplicate_rows': int(eda.get('duplicate_count') or 0),
            'data_completeness_percent': eda.get('data_completeness'),
        },
        'top_job_titles': [str(title) for title, _ in (eda.get('top_titles') or [])[:5]],
        'top_skills': [],
        'growing_skills': [],
        'declining_skills': [],
        'trend_period': None,
        'role_skill_matrix': None,
        'classifier': None,
        'forecast': None,
    }

    # --- Phase 4: observed frequency / share / growth --------------------------
    if trend.get('available'):
        summary['top_skills'] = [
            {'skill': row['skill'], 'job_count': int(row['job_count']), 'share': float(row['share'])}
            for row in (trend.get('frequency_table') or [])[:8]
        ]
        if trend.get('time_series_available') and trend.get('growth_table'):
            summary['trend_period'] = {
                'previous_month': trend.get('previous_month'),
                'recent_month': trend.get('recent_month'),
                'note': 'Growth compares the two most recent months only.',
            }
            growth = [row for row in trend['growth_table'] if row.get('growth') is not None]
            growing = sorted([r for r in growth if r['trend'] == 'Growing'],
                             key=lambda r: (r['recent_count'], r['growth']), reverse=True)
            declining = sorted([r for r in growth if r['trend'] == 'Declining'],
                               key=lambda r: (r['previous_count'], -r['growth']), reverse=True)
            summary['growing_skills'] = [
                {'skill': r['skill'], 'growth_percent': float(r['growth']),
                 'previous_count': int(r['previous_count']), 'recent_count': int(r['recent_count'])}
                for r in growing[:5]]
            summary['declining_skills'] = [
                {'skill': r['skill'], 'growth_percent': float(r['growth']),
                 'previous_count': int(r['previous_count']), 'recent_count': int(r['recent_count'])}
                for r in declining[:3]]
        role_skill = trend.get('role_skill') or {}
        if role_skill.get('available'):
            rows = {}
            for role_info, values in zip(role_skill['roles'], role_skill['matrix']):
                rows[role_info['role']] = {skill: float(pct) for skill, pct in zip(role_skill['skills'], values)}
            summary['role_skill_matrix'] = {
                'meaning': 'percent of postings in that role mentioning the skill',
                'rows': rows,
            }

    # --- Phase 3: classifier evaluation ----------------------------------------
    if ml.get('status') == 'trained':
        summary['classifier'] = {
            'roles': list(ml.get('classes') or []),
            'test_accuracy': float(ml['metrics']['accuracy']),
            'test_f1_weighted': float(ml['metrics']['f1_score']),
            'test_records': int(ml.get('test_records') or 0),
        }

    # --- Phase 5: LSTM forecast (only if the user actually ran one) -------------
    if forecast.get('status') in ('ok', 'ok_constant') and forecast.get('future'):
        future = forecast['future']
        history = forecast.get('history') or []
        summary['forecast'] = {
            'skill': forecast.get('skill'),
            'model': 'LSTM' if forecast.get('status') == 'ok' else 'constant series (LSTM skipped)',
            'last_observed_month': history[-1]['month'] if history else None,
            'last_observed_count': history[-1]['actual'] if history else None,
            'next_month': {'month': future[0]['month'], 'forecast': future[0]['forecast']},
            'following_month': {'month': future[1]['month'], 'forecast': future[1]['forecast']} if len(future) > 1 else None,
            'third_month': {'month': future[2]['month'], 'forecast': future[2]['forecast']} if len(future) > 2 else None,
            'test_mae_lstm': (forecast.get('lstm') or {}).get('mae'),
            'test_mae_naive': (forecast.get('naive') or {}).get('mae'),
            'better_on_test': forecast.get('winner'),
        }
    else:
        summary['forecast'] = {'available': False,
                               'note': 'No LSTM forecast has been run for this dataset yet.'}

    return summary


# ---------------------------------------------------------------------------
# 3. Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    'You are a careful data analyst writing a short explanation of results that were '
    'already computed by a job-market analysis application. '
    'Rules you must follow: '
    '(1) Use ONLY the numbers and names supplied in the JSON. Never invent, estimate, or round '
    'to numbers that are not present. '
    '(2) Clearly separate OBSERVED historical facts from the FORECAST, which is a model estimate. '
    '(3) Never claim certainty about the future. Say "the LSTM forecast estimates" not "will increase". '
    '(4) Say "highest observed frequency in the analyzed dataset", never "most in-demand skill". '
    '(5) If a section of the data is missing or marked unavailable, say so briefly instead of guessing. '
    '(6) Keep every field concise and plain-English, suitable for a non-expert reader. '
    '(7) Respond with a single JSON object and nothing else.'
)


def build_insight_prompt(summary):
    """Returns (system_prompt, user_prompt). The user prompt embeds the structured summary."""
    user_prompt = (
        'Here are the structured results from the analysis (JSON):\n\n'
        f'{json.dumps(summary, indent=2, default=str)}\n\n'
        'Write the explanation as JSON with exactly these keys:\n'
        '{\n'
        '  "summary": "2-3 sentence overall summary of what was OBSERVED in the dataset",\n'
        '  "key_trends": ["2 to 5 short bullet strings about the most important observed skill trends"],\n'
        '  "forecast_interpretation": "explain the LSTM forecast as a model estimate, mention the '
        'baseline comparison if present, or state that no forecast was run",\n'
        '  "role_skill_insight": "how skills relate to broad job roles using the role_skill_matrix, '
        'or state it is unavailable",\n'
        '  "limitations": "1-3 sentences: dataset is the only source, observed frequency is not '
        'guaranteed demand, forecasts are model estimates, posting data can be biased"\n'
        '}\n'
        'Return only the JSON object.'
    )
    return SYSTEM_PROMPT, user_prompt


# ---------------------------------------------------------------------------
# 4. One plain HTTP call to the provider
# ---------------------------------------------------------------------------
def call_llm_api(system_prompt, user_prompt, config):
    """
    Sends ONE request to an OpenAI-compatible Chat Completions endpoint.
    Returns (content_text, error_message). The API key only ever travels in the
    Authorization header from Flask to the provider - never to the browser or logs.
    """
    headers = {
        'Authorization': f"Bearer {config['api_key']}",
        'Content-Type': 'application/json',
    }
    payload = {
        'model': config['model'],
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
        'temperature': TEMPERATURE,
        'max_tokens': MAX_TOKENS,
        'response_format': {'type': 'json_object'},  # JSON mode where supported
    }

    try:
        response = requests.post(config['api_url'], headers=headers, json=payload,
                                 timeout=config['timeout'])
    except requests.exceptions.Timeout:
        return None, f"The LLM provider did not respond within {config['timeout']} seconds. Please try again."
    except requests.exceptions.ConnectionError:
        return None, 'Could not connect to the LLM provider. Check your internet connection or LLM_API_URL.'
    except requests.exceptions.RequestException as error:
        return None, f'LLM request failed: {type(error).__name__}.'

    if response.status_code == 401:
        return None, 'The LLM provider rejected the API key (HTTP 401). Check LLM_API_KEY.'
    if response.status_code == 429:
        return None, 'The LLM provider is rate-limiting requests (HTTP 429). Wait a moment and try again.'
    if 400 <= response.status_code < 500:
        return None, f'The LLM provider returned a client error (HTTP {response.status_code}). Check LLM_MODEL and LLM_API_URL.'
    if response.status_code >= 500:
        return None, f'The LLM provider had a server error (HTTP {response.status_code}). Try again later.'

    try:
        body = response.json()
        content = body['choices'][0]['message']['content']
    except (ValueError, KeyError, IndexError, TypeError):
        return None, 'The LLM provider returned an unexpected response format.'

    if not content or not str(content).strip():
        return None, 'The LLM provider returned an empty response.'
    return str(content), None


# ---------------------------------------------------------------------------
# 5. Validation (simple, no fancy parsing)
# ---------------------------------------------------------------------------
def _strip_code_fence(text):
    """Some models wrap JSON in ```json ... ``` even in JSON mode. Remove it if present."""
    text = text.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1] if '\n' in text else text[3:]
        if text.rstrip().endswith('```'):
            text = text.rstrip()[:-3]
    return text.strip()


def validate_llm_response(raw_text):
    """
    Parses the model output as JSON and checks the five required keys.
    Returns (insights_dict, error_message). Never raises.
    """
    if raw_text is None:
        return None, 'No response text to validate.'
    try:
        data = json.loads(_strip_code_fence(raw_text))
    except (ValueError, TypeError):
        return None, 'The LLM response was not valid JSON, so it was not displayed.'
    if not isinstance(data, dict):
        return None, 'The LLM response was not a JSON object.'

    missing = [field for field in REQUIRED_FIELDS if field not in data or data[field] in (None, '', [])]
    if missing:
        return None, f"The LLM response was incomplete (missing: {', '.join(missing)})."

    key_trends = data['key_trends']
    if isinstance(key_trends, str):
        key_trends = [key_trends]
    if not isinstance(key_trends, list):
        return None, 'The LLM response field "key_trends" was not a list.'
    key_trends = [str(item).strip() for item in key_trends if str(item).strip()][:5]
    if not key_trends:
        return None, 'The LLM response contained no key trends.'

    return {
        'summary': str(data['summary']).strip(),
        'key_trends': key_trends,
        'forecast_interpretation': str(data['forecast_interpretation']).strip(),
        'role_skill_insight': str(data['role_skill_insight']).strip(),
        'limitations': str(data['limitations']).strip(),
    }, None


# ---------------------------------------------------------------------------
# 6. Entry point used by Flask (never raises)
# ---------------------------------------------------------------------------
def generate_insights(analysis_summary, config=None, api_caller=None):
    """
    Builds the prompt, calls the provider, validates the output.

    `api_caller` lets tests inject a fake provider (no real network call).
    Always returns a dict:
        {'status': 'ok', 'insights': {...}, 'model': ..., 'summary_sent': {...}}
        {'status': 'disabled' | 'error', 'message': '...', 'summary_sent': {...}}
    """
    config = config or get_llm_config()
    caller = api_caller or call_llm_api

    if not analysis_summary or not analysis_summary.get('dataset', {}).get('total_jobs'):
        return {'status': 'error', 'message': 'No analysis results are available yet. Upload a dataset first.',
                'summary_sent': analysis_summary}

    if not config['enabled'] or not config['api_key']:
        return {'status': 'disabled', 'message': UNAVAILABLE_MESSAGE, 'summary_sent': analysis_summary}

    try:
        system_prompt, user_prompt = build_insight_prompt(analysis_summary)
        raw_text, api_error = caller(system_prompt, user_prompt, config)
        if api_error:
            return {'status': 'error', 'message': api_error, 'summary_sent': analysis_summary}
        insights, validation_error = validate_llm_response(raw_text)
        if validation_error:
            return {'status': 'error', 'message': validation_error, 'summary_sent': analysis_summary}
        return {'status': 'ok', 'insights': insights, 'model': config['model'],
                'summary_sent': analysis_summary}
    except Exception as error:  # last-resort guard: the LLM must never break the dashboard
        return {'status': 'error',
                'message': f'AI insight generation stopped safely ({type(error).__name__}).',
                'summary_sent': analysis_summary}
