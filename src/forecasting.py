"""
LSTM forecasting of monthly skill job counts.

TensorFlow is imported lazily in build_lstm_model() so the rest of the app
and the tests still work when it is not installed.
"""

import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

LOOKBACK = 3
MIN_MONTHS = 8
EPOCHS = 30
BATCH_SIZE = 8
LSTM_UNITS = 32
PATIENCE = 5
FUTURE_MONTHS = 3
TEST_FRACTION = 0.2
RANDOM_SEED = 42

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_CSV = os.path.join(BASE_DIR, 'data', 'processed', 'skill_trends.csv')
CHARTS_DIR = os.path.join(BASE_DIR, 'static', 'charts')


def load_trends_table(csv_path=None):
    """Return (dataframe, error_message); error_message is None on success."""
    path = csv_path or PROCESSED_CSV
    if not os.path.exists(path):
        return None, ('Forecast unavailable: data/processed/skill_trends.csv was not found. '
                      'Upload a dataset with a valid date_posted column first so Phase 4 can build it.')
    try:
        df = pd.read_csv(path)
    except Exception as error:
        return None, f'Forecast unavailable: skill_trends.csv could not be read ({error}).'

    for column in ('month', 'skill', 'job_count'):
        if column not in df.columns:
            return None, (f'Forecast unavailable: skill_trends.csv is missing the '
                          f'required "{column}" column.')

    df = df.copy()
    df['month_parsed'] = pd.to_datetime(df['month'], errors='coerce')
    df['job_count'] = pd.to_numeric(df['job_count'], errors='coerce')
    invalid_rows = int(df['month_parsed'].isna().sum() + df['job_count'].isna().sum())
    df = df.dropna(subset=['month_parsed', 'job_count'])
    if df.empty:
        return None, 'Forecast unavailable: skill_trends.csv has no valid month/job_count rows.'
    df = df.sort_values(['skill', 'month_parsed']).reset_index(drop=True)
    df['_invalid_rows'] = invalid_rows
    return df, None


def get_skill_monthly_series(skill, csv_path=None):
    if not skill or not str(skill).strip():
        return {'available': False, 'message': 'Select a skill to forecast.'}
    table, error = load_trends_table(csv_path)
    if error:
        return {'available': False, 'message': error}

    name = str(skill).strip()
    rows = table[table['skill'] == name]
    if rows.empty:
        rows = table[table['skill'].str.lower() == name.lower()]
    if rows.empty:
        return {'available': False,
                'message': f'Forecast unavailable: skill "{name}" was not found in skill_trends.csv.'}

    rows = rows.sort_values('month_parsed')
    months = rows['month_parsed'].dt.strftime('%Y-%m').tolist()
    values = [float(value) for value in rows['job_count'].tolist()]
    return {'available': True, 'skill': rows['skill'].iloc[0],
            'months': months, 'values': values, 'message': None}


def get_forecastable_skills(csv_path=None, min_months=MIN_MONTHS):
    """Skills below min_months are still listed, flagged, so users see why they can't be forecast."""
    table, error = load_trends_table(csv_path)
    if error:
        return []
    summary = (table.groupby('skill')
               .agg(months=('month_parsed', 'nunique'), total=('job_count', 'sum'))
               .reset_index()
               .sort_values('total', ascending=False))
    return [{'skill': row['skill'], 'months': int(row['months']),
             'total': int(row['total']), 'sufficient': int(row['months']) >= min_months}
            for _, row in summary.iterrows()]


def create_sequences(values, lookback=LOOKBACK):
    """Sliding windows: X shaped (samples, lookback, 1), y shaped (samples,)."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) <= lookback:
        return np.zeros((0, lookback, 1)), np.zeros((0,))
    windows = [values[i:i + lookback] for i in range(len(values) - lookback)]
    targets = [values[i + lookback] for i in range(len(values) - lookback)]
    return np.array(windows).reshape(-1, lookback, 1), np.array(targets)


def chronological_month_split(n_months, test_fraction=TEST_FRACTION):
    """Chronological split; shuffling would leak future months into training."""
    test_n = max(1, int(n_months * test_fraction))
    return n_months - test_n, test_n


def prepare_forecasting_data(skill, csv_path=None):
    series = get_skill_monthly_series(skill, csv_path)
    if not series['available']:
        return {'available': False, 'message': series['message']}

    months, values = series['months'], series['values']
    n_months = len(months)
    if n_months < MIN_MONTHS:
        return {'available': False,
                'message': (f'Forecast unavailable: "{series["skill"]}" has only {n_months} months '
                            f'of historical data. At least {MIN_MONTHS} months are recommended '
                            f'for reliable forecasting. Upload a dataset spanning more months.')}

    train_n, test_n = chronological_month_split(n_months)
    raw = np.array(values, dtype=float)

    # Fit the scaler on training months only to avoid leaking test statistics.
    scaler = MinMaxScaler()
    scaler.fit(raw[:train_n].reshape(-1, 1))
    scaled = scaler.transform(raw.reshape(-1, 1)).reshape(-1)

    X_all, y_all = create_sequences(scaled, LOOKBACK)
    # The last test_n windows are the ones whose targets are the test months.
    X_train, y_train = X_all[:-test_n], y_all[:-test_n]
    X_test, y_test = X_all[-test_n:], y_all[-test_n:]
    if len(X_train) < 2 or len(X_test) < 1:
        return {'available': False,
                'message': ('Forecast unavailable: not enough training sequences could be built '
                            'from this skill\'s history.')}

    test_actual_raw = raw[-test_n:].tolist()
    return {
        'available': True, 'message': None, 'skill': series['skill'],
        'months': months, 'values': values, 'n_months': n_months,
        'train_months': train_n, 'test_months': test_n,
        'test_month_labels': months[-test_n:], 'test_actual_raw': test_actual_raw,
        'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test,
        'scaler': scaler, 'scaled_full': scaled,
        'is_constant': bool(np.all(raw == raw[0])),
    }


def build_lstm_model(lookback=LOOKBACK):
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
    except ImportError as error:
        raise ImportError(
            'TensorFlow is not installed in this environment. '
            'Run "pip install -r requirements.txt" to enable LSTM forecasting.') from error

    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    model = Sequential([
        LSTM(LSTM_UNITS, input_shape=(lookback, 1)),
        Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def train_forecast_model(X_train, y_train):
    model = build_lstm_model(X_train.shape[1])
    fit_kwargs = {'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'verbose': 0}
    # Too few windows make a validation split meaningless.
    if len(X_train) >= 6:
        from tensorflow.keras.callbacks import EarlyStopping
        fit_kwargs['validation_split'] = 0.2
        fit_kwargs['callbacks'] = [EarlyStopping(monitor='val_loss', patience=PATIENCE,
                                                 restore_best_weights=True)]
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def evaluate_forecast_model(y_true_raw, y_pred_raw):
    actual = np.asarray(y_true_raw, dtype=float)
    predicted = np.asarray(y_pred_raw, dtype=float)
    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    return {'mae': round(mae, 2), 'rmse': round(rmse, 2)}


def naive_baseline_forecast(raw_values, test_n):
    """Predict each test month as the previous month's actual value."""
    raw = np.asarray(raw_values, dtype=float)
    actual = raw[-test_n:]
    predicted = raw[len(raw) - test_n - 1:len(raw) - 1]
    return evaluate_forecast_model(actual, predicted)


def forecast_future_values(model, scaler, scaled_full_series, n_months=FUTURE_MONTHS):
    window = list(np.asarray(scaled_full_series, dtype=float).reshape(-1)[-LOOKBACK:])
    scaled_predictions = []
    for _ in range(n_months):
        next_scaled = float(model.predict(np.array(window).reshape(1, LOOKBACK, 1), verbose=0)[0][0])
        scaled_predictions.append(next_scaled)
        # Feed each prediction back into the window for recursive forecasting.
        window = window[1:] + [next_scaled]
    raw = scaler.inverse_transform(np.array(scaled_predictions).reshape(-1, 1)).reshape(-1)
    return [int(max(0, round(float(value)))) for value in raw]


def next_month_labels(last_month_label, n_months=FUTURE_MONTHS):
    last = pd.Period(last_month_label, freq='M')
    return [str(last + offset) for offset in range(1, n_months + 1)]


def generate_forecast_chart(months, values, test_months, lstm_test_preds, future_months,
                            future_values, skill, filename):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    filepath = os.path.join(CHARTS_DIR, filename)

    history_x = list(range(len(months)))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    ax.plot(history_x, values, color='#28785e', linewidth=2.2, marker='o',
            markersize=4.5, label='Actual job count')

    if test_months and lstm_test_preds is not None:
        test_x = [months.index(month) for month in test_months]
        ax.plot(test_x, list(lstm_test_preds), color='#c98b30', linewidth=0,
                marker='s', markersize=7, label='LSTM prediction (held-out test)')

    if future_months and future_values is not None:
        future_x = [len(months) - 1] + [len(months) + i for i in range(len(future_months))]
        future_y = [values[-1]] + list(future_values)
        ax.plot(future_x, future_y, color='#3f7fbf', linewidth=2.2, linestyle='--',
                marker='D', markersize=5, label='Future forecast (next 3 months)')

    all_labels = list(months) + list(future_months or [])
    step = max(1, len(all_labels) // 12)
    tick_positions = list(range(0, len(all_labels), step))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([all_labels[i] for i in tick_positions], rotation=35, ha='right')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#e5eae6')
    ax.spines['bottom'].set_color('#e5eae6')
    ax.tick_params(colors='#4c5b50', labelsize=9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#e5eae6')
    ax.set_xlabel('Month', fontsize=9, color='#738177')
    ax.set_ylabel('Job Count (postings mentioning skill)', fontsize=9, color='#738177')
    ax.legend(fontsize=9, frameon=False, loc='upper left')
    ax.set_title(f'AI Trend Forecast: {skill} (predicted mention volume, not guaranteed demand)',
                 fontsize=11, fontweight='bold', color='#24352e', pad=12, loc='left')
    plt.tight_layout()
    plt.savefig(filepath, dpi=160)
    plt.close(fig)
    return f'charts/{filename}'


# ---------------------------------------------------------------------------
# 7. One entry point used by app.py (never raises; Flask must not crash)
# ---------------------------------------------------------------------------
def run_skill_forecast(skill, csv_path=None):
    """
    Trains the LSTM for ONE skill and returns a template-friendly result dict.

    Always returns a dict with status 'ok' or 'unavailable'. The naive baseline
    is included whenever test months exist, even if the LSTM itself failed, so
    the dashboard can still show something honest and useful.
    """
    result = {
        'status': 'unavailable',
        'message': 'Forecast could not run for this skill.',
        'skill': str(skill).strip() if skill else None,
        'config': {'lookback': LOOKBACK, 'epochs': EPOCHS, 'batch_size': BATCH_SIZE,
                   'lstm_units': LSTM_UNITS, 'test_fraction': TEST_FRACTION,
                   'min_months': MIN_MONTHS, 'future_months': FUTURE_MONTHS},
    }

    try:
        prepared = prepare_forecasting_data(skill, csv_path)
        if not prepared['available']:
            result['message'] = prepared['message']
            return result

        values = prepared['values']
        months = prepared['months']
        test_n = prepared['test_months']
        test_actual = prepared['test_actual_raw']
        test_labels = prepared['test_month_labels']

        result.update({
            'n_months': prepared['n_months'],
            'train_months': prepared['train_months'],
            'test_months': test_n,
            'test_month_labels': test_labels,
            'history': [{'month': month, 'actual': int(value)} for month, value in zip(months, values)],
        })

        # Naive baseline first: it never needs TensorFlow.
        result['naive'] = naive_baseline_forecast(values, test_n)

        if prepared['is_constant']:
            result['message'] = ('This skill has a constant job count every month, so a neural network '
                                 'has no pattern to learn. The naive baseline below is already perfect, '
                                 'and the future forecast simply repeats the last observed value.')
            future_values = [int(values[-1])] * FUTURE_MONTHS
            future_labels = next_month_labels(months[-1])
            safe = re.sub(r'[^a-z0-9]+', '_', prepared['skill'].lower()).strip('_')
            result['future'] = [{'month': m, 'forecast': v}
                                for m, v in zip(future_labels, future_values)]
            result['chart'] = generate_forecast_chart(
                months, values, [], None, future_labels, future_values,
                prepared['skill'], f'forecast_{safe}.png')
            result['status'] = 'ok_constant'
            result['lstm'] = None
            return result

        try:
            model = train_forecast_model(prepared['X_train'], prepared['y_train'])
        except ImportError as error:
            result['message'] = str(error)
            return result
        except Exception as error:
            result['message'] = (f'Model training failed safely ({str(error)[:160]}). '
                                 f'The naive baseline below was still computed on the test months.')
            return result

        # LSTM test predictions -> back to the original job-count scale.
        scaled_preds = model.predict(prepared['X_test'], verbose=0).reshape(-1)
        lstm_test_raw = prepared['scaler'].inverse_transform(scaled_preds.reshape(-1, 1)).reshape(-1)
        lstm_test_raw = [max(0.0, float(value)) for value in lstm_test_raw]
        result['lstm'] = evaluate_forecast_model(test_actual, lstm_test_raw)
        result['test_predictions'] = [
            {'month': month, 'actual': int(actual), 'lstm_pred': round(float(pred), 1)}
            for month, actual, pred in zip(test_labels, test_actual, lstm_test_raw)]

        # Honest verdict: the LSTM does NOT always win on small/noisy data.
        if result['lstm']['mae'] < result['naive']['mae']:
            result['verdict'] = ('LSTM outperformed the naive baseline on the held-out test period, '
                                 'suggesting the extra model complexity added value here.')
            result['winner'] = 'LSTM'
        elif result['lstm']['mae'] > result['naive']['mae']:
            result['verdict'] = ('The naive baseline performed better on this small/noisy series. '
                                 'This is reported honestly: added complexity did not help here.')
            result['winner'] = 'Naive baseline'
        else:
            result['verdict'] = 'LSTM and the naive baseline tied on the held-out test period.'
            result['winner'] = 'Tie'

        # Recursive future forecast from the latest window.
        future_values = forecast_future_values(model, prepared['scaler'], prepared['scaled_full'])
        future_labels = next_month_labels(months[-1])
        result['future'] = [{'month': m, 'forecast': v}
                            for m, v in zip(future_labels, future_values)]

        safe = re.sub(r'[^a-z0-9]+', '_', prepared['skill'].lower()).strip('_')
        result['chart'] = generate_forecast_chart(
            months, values, test_labels, lstm_test_raw, future_labels, future_values,
            prepared['skill'], f'forecast_{safe}.png')
        result['status'] = 'ok'
        result['message'] = None
        return result

    except Exception as error:
        # Last-resort guard: Phase 5 must never crash the Flask app.
        result['message'] = f'Forecast stopped safely: {str(error)[:200]}'
        return result
