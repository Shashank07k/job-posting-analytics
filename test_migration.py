"""
Automated Verification Suite for LinkedInJobTrendAI

Phase 1-3 tests:
1. NLP cleaning, tokenization, stopword removal
2. Skill extraction dictionary matching and aliases
3. Job title rule mapping
4. Full EDA on sample_jobs.csv
5. Model training, evaluation metrics, prediction
6. Graceful degradation (missing columns, empty text)

Phase 4 tests:
7. Skill frequency counted once per job description (repeated mentions)
8. Monthly skill time series with zero-filled gaps
9. Growth rate including division-by-zero (previous period = 0)
10. Trend classification thresholds (Growing / Stable / Declining)
11. Role-skill matrix percentages
12. Engineered feature table columns + CSV written to data/processed/
13. Missing date_posted handling (frequency still available)
14. Invalid dates handling
15. Very small dataset does not crash

Phase 5 tests:
16. Sliding-window sequences with lookback=3 (pure NumPy, no TensorFlow)
17. Chronological 80/20 split is never shuffled
18. MAE/RMSE match hand-computed values
19. Naive baseline predicts the previous month's count
20. Recursive 3-month forecast with a stub model (no TensorFlow)
21. Insufficient history, missing file, and missing job_count column
22. Constant series and invalid month rows handled without TensorFlow
23. Full LSTM run when TensorFlow is installed, graceful skip otherwise
24. Zero job counts handled in metrics and data preparation

Phase 6 tests (all use a fake API caller - NO real network call is ever made):
25. Missing API key -> disabled, friendly message
26. LLM_ENABLED=false -> disabled even when a key exists
27. Successful mock JSON response -> parsed into the five insight fields
28. Invalid / non-JSON response -> clean error, no crash
29. API failure (timeout / 429 / 500 style) -> clean error, no crash
30. Missing forecast info -> summary marks forecast unavailable, insights still work
31. Normal complete analysis -> summary contains only real computed values, no raw CSV
"""

import os
import io
import numpy as np
import pandas as pd
from src.data_analysis import perform_eda
from src.nlp_processing import clean_text, tokenize_text, remove_stopwords, preprocess_text
from src.skill_extraction import extract_skills_from_text, analyze_skills_in_series
from src.job_classifier import map_job_title, train_job_classifier, predict_job_role
from src.trend_analysis import (
    analyze_skill_trends, classify_trend, extract_skill_records,
    calculate_skill_frequency, build_skill_time_series, calculate_skill_growth,
    build_forecasting_features, save_forecasting_features
)
from src.forecasting import (
    create_sequences, chronological_month_split, evaluate_forecast_model,
    naive_baseline_forecast, forecast_future_values, prepare_forecasting_data,
    run_skill_forecast, LOOKBACK, MIN_MONTHS
)
from src.llm_service import (
    build_analysis_summary, build_insight_prompt, validate_llm_response,
    generate_insights, UNAVAILABLE_MESSAGE
)

FEATURES_PATH = os.path.join('data', 'processed', 'skill_trends.csv')

def run_tests():
    print("=" * 60)
    print("Starting LinkedInJobTrendAI Python Verification Tests...")
    print("=" * 60)

    # Test 1: NLP Preprocessing
    raw_sample = "Looking for a PYTHON Developer!!! Visit https://example.com/apply"
    cleaned = clean_text(raw_sample)
    assert cleaned == "looking for a python developer", f"Unexpected clean: {cleaned}"
    tokens = tokenize_text(cleaned)
    assert len(tokens) == 5
    filtered = remove_stopwords(tokens)
    assert "looking" in filtered and "python" in filtered and "developer" in filtered
    print(" [PASS] NLP Preprocessing (clean, tokenize, stopwords)")

    # Test 2: Skill Extraction
    skills_extracted = extract_skills_from_text("Experience with Python, Pandas, NumPy, Machine Learning and AWS required.")
    assert "Python" in skills_extracted
    assert "Pandas" in skills_extracted
    assert "NumPy" in skills_extracted
    assert "Machine Learning" in skills_extracted
    assert "AWS" in skills_extracted
    print(" [PASS] Skill Extraction (dictionary matching & aliases)")

    # Test 3: Title Mapping
    assert map_job_title("Senior Data Scientist") == "Data Scientist"
    assert map_job_title("Machine Learning Engineer") == "AI/ML Engineer"
    assert map_job_title("Backend Software Developer") == "Software Engineer"
    assert map_job_title("Cloud Engineer") == "DevOps/Cloud Engineer"
    assert map_job_title("Cybersecurity Analyst") == "Cybersecurity Engineer"
    assert map_job_title("Product Manager") is None
    print(" [PASS] Job Title Mapping Rules (6 broad roles)")

    # Test 4: End-to-end with sample_jobs.csv
    sample_path = "sample_jobs.csv"
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
        eda_res = perform_eda(df, "sample_jobs.csv")
        assert eda_res['total_rows'] > 0
        assert eda_res['total_cols'] == 8
        print(f" [PASS] Phase 1 EDA: {eda_res['total_rows']} rows, {eda_res['total_cols']} columns")

        nlp_res = analyze_skills_in_series(df['job_description'])
        assert nlp_res['available'] is True
        assert len(nlp_res['top_skills']) > 0
        print(f" [PASS] Phase 2 Skills: {nlp_res['unique_skills_found']} unique skills found across postings")

        ml_res = train_job_classifier(df)
        assert ml_res['status'] == 'trained'
        assert ml_res['metrics']['accuracy'] > 0.5
        print(f" [PASS] Phase 3 ML Classifier: Accuracy = {ml_res['metrics']['accuracy']*100:.1f}%, F1 = {ml_res['metrics']['f1_score']*100:.1f}%")

        # Test prediction
        pred = predict_job_role("Develop scalable microservices using Java, Spring Boot, and Docker.")
        assert pred['status'] == 'success'
        print(f" [PASS] Phase 3 Predict: Role='{pred['predicted_role']}' (Confidence: {pred['confidence']}%)")

    # Test 5: Graceful degradation without job_description
    df_no_desc = pd.DataFrame({'job_title': ['Software Engineer', 'Data Scientist'], 'company': ['A', 'B']})
    ml_no_desc = train_job_classifier(df_no_desc)
    assert ml_no_desc['status'] == 'unavailable'
    print(" [PASS] Graceful error handling: Missing job_description")

    # Test 6: Graceful degradation without job_title
    df_no_title = pd.DataFrame({'job_description': ['Python developer', 'Java developer'], 'company': ['A', 'B']})
    ml_no_title = train_job_classifier(df_no_title)
    assert ml_no_title['status'] == 'unavailable'
    print(" [PASS] Graceful error handling: Missing job_title")

    # Test 7: Empty prediction text
    pred_empty = predict_job_role("   ")
    assert pred_empty['status'] == 'error'
    print(" [PASS] Empty prediction text validation")

    print("-" * 60)
    print("PHASE 4: SKILL TREND ANALYSIS + FEATURE ENGINEERING")
    print("-" * 60)

    # Test 8: Repeated skill mentions counted once per description
    repeat_df = pd.DataFrame({
        'job_title': ['Data Analyst', 'Data Analyst'],
        'job_description': ['Python Python Python AWS', 'Python SQL'],
        'date_posted': ['2026-01-10', '2026-02-11']
    })
    records = extract_skill_records(repeat_df, 'job_description', 'date_posted')
    freq = calculate_skill_frequency(records, 2)
    python_row = next(row for row in freq if row['skill'] == 'Python')
    aws_row = next(row for row in freq if row['skill'] == 'AWS')
    assert python_row['job_count'] == 2, "Python mentioned in 2 postings -> count 2, not 4"
    assert aws_row['job_count'] == 1
    assert python_row['share'] == 100.0  # both postings mention Python
    print(" [PASS] Skill frequency counted once per description (repeats ignored)")

    # Test 9: Monthly series with a zero-filled gap month
    gap_df = pd.DataFrame({
        'job_title': ['Software Engineer', 'Software Engineer', 'Software Engineer'],
        'job_description': ['Python Java Docker', 'Python Java', 'Python Java'],
        'date_posted': ['2026-01-05', '2026-01-20', '2026-03-02']  # February deliberately absent
    })
    gap_records = extract_skill_records(gap_df, 'job_description', 'date_posted')
    monthly_totals = {'2026-01': 2, '2026-02': 0, '2026-03': 1}
    series, months, totals = build_skill_time_series(gap_records, monthly_totals)
    assert months == ['2026-01', '2026-02', '2026-03'], f"Unexpected months: {months}"
    assert int(series.loc['2026-02', 'Python']) == 0, "Missing month must be zero-filled, not skipped"
    print(" [PASS] Monthly time series with zero-filled gap (Jan -> Feb(0) -> Mar)")

    # Test 10: Growth rate + division-by-zero safety
    growth_table, prev_month, recent_month, _ = calculate_skill_growth(series)
    python_growth = next(row for row in growth_table if row['skill'] == 'Python')
    assert python_growth['previous_count'] == 0 and python_growth['growth'] is None
    assert python_growth['trend'] == 'New', "previous_count = 0 must not raise ZeroDivisionError"
    zero_df = pd.DataFrame({'Python': [0]}, index=['2026-01'])
    zero_table, _, _, _ = calculate_skill_growth(zero_df)
    assert zero_table[0]['growth'] is None and zero_table[0]['trend'] == 'Insufficient data'
    assert classify_trend(25.0) == 'Growing'
    assert classify_trend(4.0) == 'Stable'
    assert classify_trend(-30.0) == 'Declining'
    print(" [PASS] Growth rate with safe zero division + trend thresholds (+/-10%)")

    # Test 11: Engineered feature table + CSV saved for Phase 5
    features = build_forecasting_features(series, totals)
    expected_columns = ['month', 'skill', 'job_count', 'skill_share',
                        'previous_month_count', 'growth_rate', 'trend']
    assert list(features.columns) == expected_columns, features.columns.tolist()
    saved_path = save_forecasting_features(features)
    assert saved_path and os.path.exists(FEATURES_PATH), "skill_trends.csv was not written"
    reloaded = pd.read_csv(FEATURES_PATH)
    assert len(reloaded) == len(features)
    print(f" [PASS] Engineered features saved to {saved_path} ({len(features)} rows, 4 core features)")

    # Test 12: Full Phase 4 run on the sample dataset (frequency, growth, roles, charts)
    if os.path.exists(sample_path):
        sample_df = pd.read_csv(sample_path)
        trend = analyze_skill_trends(sample_df, 'sample_jobs.csv')
        assert trend['available'] is True
        assert trend['tracked_skills'] > 0
        assert trend['time_series_available'] is True
        assert trend['frequency_table'][0]['share'] <= 100.0
        assert trend['role_skill']['available'] is True
        assert len(trend['role_skill']['matrix']) >= 2
        assert trend['skill_detail'] is not None
        assert trend['charts'].get('top_frequency') and trend['charts'].get('skill_trend')
        max_share = max(row['share'] for row in trend['frequency_table'])
        assert max_share <= 100.0, "Share must never exceed 100%"
        print(f" [PASS] Phase 4 on sample: {trend['tracked_skills']} tracked skills, "
              f"growing={trend['trend_counts'].get('Growing', 0)}, stable={trend['trend_counts'].get('Stable', 0)}, "
              f"declining={trend['trend_counts'].get('Declining', 0)}")
        print(f" [PASS] Role-skill matrix built from {trend['role_source']} "
              f"({len(trend['role_skill']['roles'])} roles x {len(trend['role_skill']['skills'])} skills)")

    # Test 13: Missing date_posted -> no crash, frequency still available
    no_date_df = pd.DataFrame({
        'job_title': ['Data Scientist', 'Data Scientist'],
        'job_description': ['Python Pandas Machine Learning', 'Python TensorFlow deep learning']
    })
    no_date_trend = analyze_skill_trends(no_date_df, 'no_dates.csv')
    assert no_date_trend['available'] is True
    assert no_date_trend['time_series_available'] is False
    assert len(no_date_trend['frequency_table']) > 0
    assert no_date_trend['month_warning'] is not None
    print(" [PASS] Missing date_posted handled: frequency shown, monthly trend explained")

    # Test 14: Invalid dates -> excluded from trend only, clear warning
    bad_date_df = pd.DataFrame({
        'job_title': ['Data Analyst', 'Data Analyst', 'Data Analyst'],
        'job_description': ['Python SQL', 'SQL Excel', 'Python Pandas'],
        'date_posted': ['2026-01-10', 'not-a-date', '2026-02-11']
    })
    bad_date_trend = analyze_skill_trends(bad_date_df, 'bad_dates.csv')
    assert bad_date_trend['available'] is True
    assert bad_date_trend['month_warning'] is not None
    assert len(bad_date_trend['frequency_table']) == 4  # Python, SQL, Pandas, Excel
    print(" [PASS] Invalid dates handled with warning, no crash")

    # Test 15: Tiny dataset (2 rows) does not crash
    tiny_df = pd.DataFrame({
        'job_title': ['Software Engineer', 'Cybersecurity Analyst'],
        'job_description': ['Python basics', 'network security'],
        'date_posted': ['2026-01-01', '2026-01-02']
    })
    tiny_trend = analyze_skill_trends(tiny_df, 'tiny.csv')
    assert tiny_trend['available'] is True
    assert tiny_trend['emerging']['available'] is True
    assert tiny_trend['emerging']['items'] == []  # no volume, so no emerging skills
    print(" [PASS] Small dataset handled: analysis runs, emerging list stays empty instead of misleading")

    # Test 16: Missing job_description -> Phase 4 also degrades gracefully
    no_desc_trend = analyze_skill_trends(pd.DataFrame({'job_title': ['Data Analyst']}), 'no_desc.csv')
    assert no_desc_trend['available'] is False and 'job_description' in no_desc_trend['message']
    print(" [PASS] Missing job_description handled: clear Phase 4 message")

    print("-" * 60)
    print("PHASE 5: TENSORFLOW LSTM FORECASTING")
    print("-" * 60)

    # Test 17: Sliding windows with lookback=3 (pure NumPy, no TensorFlow)
    X_seq, y_seq = create_sequences([120, 140, 155, 170, 180], lookback=3)
    assert X_seq.shape == (2, 3, 1), X_seq.shape
    assert y_seq.tolist() == [170.0, 180.0]
    assert X_seq[0].reshape(-1).tolist() == [120.0, 140.0, 155.0]
    assert X_seq[1].reshape(-1).tolist() == [140.0, 155.0, 170.0]
    X_empty, y_empty = create_sequences([1, 2], lookback=3)
    assert len(X_empty) == 0 and len(y_empty) == 0
    assert LOOKBACK == 3
    print(" [PASS] Sliding windows: [120,140,155]->170 with lookback=3")

    # Test 18: Chronological 80/20 split is never shuffled
    assert chronological_month_split(10) == (8, 2)
    assert chronological_month_split(8) == (7, 1)
    assert chronological_month_split(12) == (10, 2)
    print(" [PASS] Chronological split: first months train, last months test")

    # Test 19: MAE/RMSE match hand-computed values
    import math
    metrics = evaluate_forecast_model([10, 20, 30], [12, 18, 33])
    assert metrics['mae'] == round((2 + 2 + 3) / 3, 2)  # 2.33
    assert metrics['rmse'] == round(math.sqrt((4 + 4 + 9) / 3), 2)  # 2.38
    print(" [PASS] MAE/RMSE match hand-computed values")

    # Test 20: Naive baseline predicts the previous month's count
    naive = naive_baseline_forecast([10, 12, 15, 14, 18], 2)
    assert naive == evaluate_forecast_model([14, 18], [15, 14])  # test actual vs shifted actual
    print(" [PASS] Naive baseline: next month = last month's count")

    # Test 21: Recursive 3-month forecast with a stub model (no TensorFlow)
    from sklearn.preprocessing import MinMaxScaler

    class StubModel:
        """Fake model: predicts last window value + 0.1 (scaled space)."""
        def predict(self, X, verbose=0):
            return np.array([[X[0, -1, 0] + 0.1]])

    scaler = MinMaxScaler().fit(np.array([[0.0], [10.0]]))
    future = forecast_future_values(StubModel(), scaler, [0.0, 0.5, 1.0], n_months=3)
    assert future == [11, 12, 13], future  # 1.1->11, 1.2->12, 1.3->13 after inverse scale

    class NegativeStub:
        def predict(self, X, verbose=0):
            return np.array([[-5.0]])

    assert forecast_future_values(NegativeStub(), scaler, [0.0, 0.0, 0.0], n_months=2) == [0, 0]
    print(" [PASS] Recursive forecast chains predictions; negatives clipped to 0")

    # Test 22: Insufficient history, missing file, missing job_count column
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        short_path = os.path.join(tmpdir, 'short.csv')
        pd.DataFrame({
            'month': [f'2026-0{m}' for m in range(1, 6)],
            'skill': ['Python'] * 5,
            'job_count': [5, 6, 7, 6, 8],
        }).to_csv(short_path, index=False)
        short = prepare_forecasting_data('Python', csv_path=short_path)
        assert short['available'] is False and 'only 5 months' in short['message']
        assert str(MIN_MONTHS) in short['message']

        missing = prepare_forecasting_data('Python', csv_path=os.path.join(tmpdir, 'nope.csv'))
        assert missing['available'] is False and 'not found' in missing['message']

        bad_path = os.path.join(tmpdir, 'bad.csv')
        pd.DataFrame({'month': ['2026-01'], 'skill': ['Python']}).to_csv(bad_path, index=False)
        bad = prepare_forecasting_data('Python', csv_path=bad_path)
        assert bad['available'] is False and 'job_count' in bad['message']

        unknown = prepare_forecasting_data('Cobol', csv_path=short_path)
        assert unknown['available'] is False and 'not found' in unknown['message']
    print(" [PASS] 5-month history, missing file, missing column, unknown skill -> clear messages")

    # Test 23: Constant series + invalid month rows handled without TensorFlow
    with tempfile.TemporaryDirectory() as tmpdir:
        const_path = os.path.join(tmpdir, 'const.csv')
        rows = [{'month': f'2026-{m:02d}', 'skill': 'Python', 'job_count': 7} for m in range(1, 11)]
        rows.append({'month': 'not-a-date', 'skill': 'Python', 'job_count': 7})  # must be dropped
        pd.DataFrame(rows).to_csv(const_path, index=False)
        result = run_skill_forecast('Python', csv_path=const_path)
        assert result['status'] == 'ok_constant', result
        assert [r['forecast'] for r in result['future']] == [7, 7, 7]
        assert result['naive']['mae'] == 0.0 and result['naive']['rmse'] == 0.0
        assert result['chart'].endswith('.png')
    print(" [PASS] Constant series: LSTM skipped honestly, naive perfect, flat forecast")

    # Test 24: Full LSTM run when TensorFlow is installed, graceful skip otherwise
    try:
        import tensorflow  # noqa: F401
        tf_available = True
    except ImportError:
        tf_available = False
    if tf_available:
        with tempfile.TemporaryDirectory() as tmpdir:
            full_path = os.path.join(tmpdir, 'full.csv')
            counts = [120, 140, 155, 170, 180, 178, 195, 210, 205, 220, 235, 250]
            pd.DataFrame({
                'month': [f'2025-{m:02d}' for m in range(1, 13)],
                'skill': ['Python'] * 12,
                'job_count': counts,
            }).to_csv(full_path, index=False)
            result = run_skill_forecast('Python', csv_path=full_path)
            assert result['status'] == 'ok', result.get('message')
            assert result['n_months'] == 12 and len(result['future']) == 3
            assert all(r['forecast'] >= 0 for r in result['future'])
            assert result['future'][0]['month'] == '2026-01'
            assert result['lstm']['mae'] >= 0 and result['chart'].endswith('.png')
            print(f" [PASS] LSTM trained: LSTM MAE={result['lstm']['mae']} vs naive MAE={result['naive']['mae']}")
    else:
        print(" [SKIP] Full LSTM training test skipped: TensorFlow not installed here")

    # Test 25: Zero job counts handled in metrics and data preparation
    zero_metrics = evaluate_forecast_model([0, 0, 5], [0, 1, 4])
    assert zero_metrics['mae'] == round((0 + 1 + 1) / 3, 2)
    with tempfile.TemporaryDirectory() as tmpdir:
        zero_path = os.path.join(tmpdir, 'zeros.csv')
        pd.DataFrame({
            'month': [f'2026-{m:02d}' for m in range(1, 11)],
            'skill': ['RAG'] * 10,
            'job_count': [0, 0, 3, 5, 0, 4, 6, 5, 7, 6],
        }).to_csv(zero_path, index=False)
        zero_prep = prepare_forecasting_data('RAG', csv_path=zero_path)
        assert zero_prep['available'] is True and zero_prep['n_months'] == 10
    print(" [PASS] Zero job counts handled in metrics and data preparation")

    print("-" * 60)
    print("PHASE 6: LLM INSIGHT LAYER (mocked - no real API calls)")
    print("-" * 60)

    # A realistic fake LATEST_ANALYSIS built from real-shaped Phase 1-5 outputs.
    fake_state = {
        'eda': {'filename': 'sample_jobs.csv', 'total_rows': 1200, 'duplicate_count': 3,
                'data_completeness': 97.5,
                'top_titles': [('Software Engineer', 300), ('Data Scientist', 250), ('AI/ML Engineer', 180)]},
        'ml': {'status': 'trained', 'classes': ['Software Engineer', 'Data Scientist', 'AI/ML Engineer'],
               'metrics': {'accuracy': 0.91, 'f1_score': 0.9}, 'test_records': 240},
        'trend': {
            'available': True, 'time_series_available': True,
            'previous_month': '2026-02', 'recent_month': '2026-03',
            'frequency_table': [
                {'skill': 'Python', 'job_count': 510, 'share': 42.5},
                {'skill': 'SQL', 'job_count': 458, 'share': 38.2},
                {'skill': 'AWS', 'job_count': 305, 'share': 25.4},
            ],
            'growth_table': [
                {'skill': 'RAG', 'growth': 31.2, 'trend': 'Growing', 'previous_count': 16, 'recent_count': 21},
                {'skill': 'Generative AI', 'growth': 24.6, 'trend': 'Growing', 'previous_count': 20, 'recent_count': 25},
                {'skill': 'Python', 'growth': 2.0, 'trend': 'Stable', 'previous_count': 100, 'recent_count': 102},
                {'skill': 'Java', 'growth': -15.0, 'trend': 'Declining', 'previous_count': 40, 'recent_count': 34},
                {'skill': 'Cobol', 'growth': None, 'trend': 'New', 'previous_count': 0, 'recent_count': 2},
            ],
            'role_skill': {'available': True,
                           'roles': [{'role': 'Data Scientist', 'postings': 250}],
                           'skills': ['Python', 'SQL'], 'matrix': [[91.0, 76.0]]},
        },
        'forecast': {'result': {
            'status': 'ok', 'skill': 'RAG', 'winner': 'LSTM',
            'history': [{'month': '2026-02', 'actual': 16}, {'month': '2026-03', 'actual': 21}],
            'future': [{'month': '2026-04', 'forecast': 320}, {'month': '2026-05', 'forecast': 345},
                       {'month': '2026-06', 'forecast': 360}],
            'lstm': {'mae': 9.7, 'rmse': 12.1}, 'naive': {'mae': 12.4, 'rmse': 15.8},
        }},
    }
    enabled_config = {'enabled': True, 'api_key': 'test-key-not-real', 'model': 'mock-model',
                      'api_url': 'https://example.invalid', 'timeout': 5}

    good_json = ('{"summary": "Python has the highest observed frequency in the analyzed dataset.",'
                 ' "key_trends": ["RAG showed positive observed growth between the two most recent months.",'
                 ' "AWS appears in about a quarter of postings."],'
                 ' "forecast_interpretation": "The LSTM forecast estimates higher RAG posting volume; this is a model estimate.",'
                 ' "role_skill_insight": "Data Scientist postings frequently mention Python and SQL.",'
                 ' "limitations": "The dataset is the only source; observed frequency is not guaranteed demand."}')

    def fake_ok_caller(system_prompt, user_prompt, config):
        # Assert the API key never appears inside the prompt text (only in headers).
        assert 'test-key-not-real' not in system_prompt and 'test-key-not-real' not in user_prompt
        return good_json, None

    # Test 26: Missing API key -> disabled with the friendly message
    summary = build_analysis_summary(fake_state)
    no_key = generate_insights(summary, config={**enabled_config, 'api_key': ''}, api_caller=fake_ok_caller)
    assert no_key['status'] == 'disabled' and no_key['message'] == UNAVAILABLE_MESSAGE
    print(" [PASS] Missing API key -> 'AI insights unavailable' message, no crash")

    # Test 27: LLM_ENABLED=false -> disabled even though a key exists
    off = generate_insights(summary, config={**enabled_config, 'enabled': False}, api_caller=fake_ok_caller)
    assert off['status'] == 'disabled'
    print(" [PASS] LLM disabled via flag -> feature off, key never used")

    # Test 28: Successful mock response -> five validated fields
    ok = generate_insights(summary, config=enabled_config, api_caller=fake_ok_caller)
    assert ok['status'] == 'ok', ok
    assert set(ok['insights'].keys()) == {'summary', 'key_trends', 'forecast_interpretation',
                                          'role_skill_insight', 'limitations'}
    assert len(ok['insights']['key_trends']) == 2
    assert ok['model'] == 'mock-model'
    fenced, err = validate_llm_response('```json\n' + good_json + '\n```')
    assert err is None and fenced['summary'].startswith('Python')
    print(" [PASS] Successful mock response parsed; code-fenced JSON also accepted")

    # Test 29: Invalid JSON / incomplete JSON -> clean error
    bad = generate_insights(summary, config=enabled_config,
                            api_caller=lambda s, u, c: ('Sure! Here are some thoughts...', None))
    assert bad['status'] == 'error' and 'not valid JSON' in bad['message']
    incomplete = generate_insights(summary, config=enabled_config,
                                   api_caller=lambda s, u, c: ('{"summary": "only one field"}', None))
    assert incomplete['status'] == 'error' and 'incomplete' in incomplete['message']
    print(" [PASS] Invalid and incomplete JSON -> friendly error, dashboard unaffected")

    # Test 30: API failure (timeout / rate limit / server error shapes) -> clean error
    for failure in ('did not respond within 5 seconds', 'rate-limiting requests (HTTP 429)',
                    'server error (HTTP 500)'):
        failed = generate_insights(summary, config=enabled_config,
                                   api_caller=lambda s, u, c, f=failure: (None, f))
        assert failed['status'] == 'error' and failed['message'] == failure
    def exploding_caller(s, u, c):
        raise RuntimeError('provider SDK blew up')
    exploded = generate_insights(summary, config=enabled_config, api_caller=exploding_caller)
    assert exploded['status'] == 'error' and 'stopped safely' in exploded['message']
    print(" [PASS] Timeout / 429 / 500 / unexpected exception -> handled without crashing")

    # Test 31: Missing forecast information -> summary flags it, insights still generated
    no_forecast_state = {**fake_state, 'forecast': {'result': None}}
    nf_summary = build_analysis_summary(no_forecast_state)
    assert nf_summary['forecast']['available'] is False
    nf = generate_insights(nf_summary, config=enabled_config, api_caller=fake_ok_caller)
    assert nf['status'] == 'ok'
    _, prompt_text = build_insight_prompt(nf_summary)
    assert 'No LSTM forecast has been run' in prompt_text
    print(" [PASS] Missing forecast -> marked unavailable in prompt, insights still work")

    # Test 32: Normal complete analysis -> only real computed values, no raw CSV rows
    assert summary['dataset']['total_jobs'] == 1200
    assert summary['top_skills'][0] == {'skill': 'Python', 'job_count': 510, 'share': 42.5}
    assert [g['skill'] for g in summary['growing_skills']] == ['Generative AI', 'RAG']  # ranked by volume
    assert summary['declining_skills'][0]['skill'] == 'Java'
    assert all(g['skill'] != 'Cobol' for g in summary['growing_skills'])  # None growth excluded
    assert summary['forecast']['skill'] == 'RAG' and summary['forecast']['next_month']['forecast'] == 320
    assert summary['forecast']['third_month']['month'] == '2026-06'
    assert summary['role_skill_matrix']['rows']['Data Scientist']['Python'] == 91.0
    assert summary['classifier']['test_accuracy'] == 0.91
    assert 'job_description' not in str(summary)  # never the raw CSV text
    empty = generate_insights(build_analysis_summary({}), config=enabled_config, api_caller=fake_ok_caller)
    assert empty['status'] == 'error' and 'Upload a dataset' in empty['message']
    print(" [PASS] Structured summary uses only computed values; empty state handled")

    print("=" * 60)
    print("ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == '__main__':
    run_tests()
