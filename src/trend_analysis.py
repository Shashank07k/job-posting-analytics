"""Monthly skill trends, growth rules and the feature table used for forecasting."""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.skill_extraction import SKILL_DICTIONARY, extract_skills_from_text
from src.data_analysis import get_column_mapping, generate_bar_chart

# Fixed analytical rules, not model output.
GROWING_THRESHOLD = 10.0
DECLINING_THRESHOLD = -10.0

# Emerging skills need real volume: a 1 -> 5 jump (+400%) is not a meaningful signal.
MIN_RECENT_MENTIONS_FLOOR = 5
MIN_RECENT_MENTIONS_SHARE = 0.03

MAX_MONTHS = 60
TOP_N_MATRIX_SKILLS = 8
MIN_POSTINGS_PER_ROLE = 3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARTS_DIR = os.path.join(BASE_DIR, 'static', 'charts')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
FEATURES_FILENAME = 'skill_trends.csv'

SKILL_CATEGORY = {skill['name']: skill['category'] for skill in SKILL_DICTIONARY}


def find_description_column(df):
    for col in df.columns:
        if str(col).strip().lower() in {'job_description', 'description', 'job_desc'}:
            return col
    return None


def find_date_column(df):
    mapping, _, _ = get_column_mapping(df)
    if mapping.get('date_posted'):
        return mapping['date_posted']

    aliases = {'date', 'posted_date', 'posting_date', 'dateposted', 'created_at', 'published_date'}
    for col in df.columns:
        if str(col).strip().lower() in aliases:
            return col
    return None


def extract_skill_records(df, desc_col, date_col=None):
    """One row per (posting, skill): each skill counts once per job description."""
    months = None
    if date_col is not None:
        parsed = pd.to_datetime(df[date_col], errors='coerce')
        months = parsed.dt.to_period('M')

    records = []
    for position, description in enumerate(df[desc_col]):
        if pd.isna(description):
            continue
        text = str(description).strip()
        if not text:
            continue
        month = None
        if months is not None:
            value = months.iloc[position]
            month = str(value) if not pd.isna(value) else None
        for skill in extract_skills_from_text(text):
            records.append({'posting_id': position, 'month': month, 'skill': skill})

    columns = ['posting_id', 'month', 'skill']
    return pd.DataFrame(records, columns=columns)


def calculate_skill_frequency(records, total_postings):
    if records.empty or total_postings == 0:
        return []

    counts = records.groupby('skill')['posting_id'].nunique().sort_values(ascending=False)
    table = []
    for skill, count in counts.items():
        table.append({
            'skill': skill,
            'category': SKILL_CATEGORY.get(skill, 'Other'),
            'job_count': int(count),
            'share': round(float(count) / total_postings * 100, 1),
        })
    return table


def build_skill_time_series(records, total_postings_by_month):
    """Month x skill counts, zero-filled so forecasting gets an unbroken series."""
    dated = records[records['month'].notna()] if not records.empty else records
    if dated.empty or not total_postings_by_month:
        return None, [], None

    counts = dated.pivot_table(index='month', columns='skill', values='posting_id',
                               aggfunc='nunique', fill_value=0)

    months = sorted(counts.index.tolist())
    if len(months) > MAX_MONTHS:
        months = months[-MAX_MONTHS:]

    full_range = pd.period_range(start=months[0], end=months[-1], freq='M').astype(str).tolist()
    full_range = full_range[-MAX_MONTHS:]

    counts = counts.reindex(full_range, fill_value=0).fillna(0).astype(int)
    counts.index.name = 'month'

    totals = pd.Series({month: total_postings_by_month.get(month, 0) for month in full_range},
                       name='total_postings')
    return counts, full_range, totals


def total_postings_per_month(df, date_col):
    if date_col is None:
        return {}, 0
    parsed = pd.to_datetime(df[date_col], errors='coerce')
    valid = parsed.dropna()
    invalid_count = int(parsed.isna().sum())
    if valid.empty:
        return {}, invalid_count
    counts = valid.dt.to_period('M').astype(str).value_counts().to_dict()
    return counts, invalid_count


def classify_trend(growth):
    if growth is None or (isinstance(growth, float) and np.isnan(growth)):
        return 'Insufficient data'
    if growth > GROWING_THRESHOLD:
        return 'Growing'
    if growth < DECLINING_THRESHOLD:
        return 'Declining'
    return 'Stable'


def calculate_skill_growth(time_series_df):
    """Compare the last two months; a zero previous count is labelled 'New', not divided."""
    if time_series_df is None or len(time_series_df) < 2:
        return [], None, None, 'Time-based growth needs at least two months of valid dates.'

    recent_month = time_series_df.index[-1]
    previous_month = time_series_df.index[-2]

    table = []
    for skill in time_series_df.columns:
        recent_count = int(time_series_df.loc[recent_month, skill])
        previous_count = int(time_series_df.loc[previous_month, skill])

        if previous_count == 0 and recent_count == 0:
            growth = None
            trend = 'Insufficient data'
        elif previous_count == 0:
            growth = None
            trend = 'New'
        else:
            growth = round((recent_count - previous_count) / previous_count * 100, 1)
            trend = classify_trend(growth)

        table.append({
            'skill': skill,
            'category': SKILL_CATEGORY.get(skill, 'Other'),
            'previous_month': previous_month,
            'recent_month': recent_month,
            'previous_count': previous_count,
            'recent_count': recent_count,
            'growth': growth,
            'trend': trend,
        })

    table.sort(key=lambda row: (row['trend'] == 'Growing',
                                row['growth'] if row['growth'] is not None else -999,
                                row['recent_count']), reverse=True)
    return table, previous_month, recent_month, None


def count_trends(growth_table):
    summary = {'Growing': 0, 'Stable': 0, 'Declining': 0, 'New': 0, 'Insufficient data': 0}
    for row in growth_table:
        summary[row['trend']] = summary.get(row['trend'], 0) + 1
    return summary


def assign_broad_roles(df, roles_from_titles):
    """Use title-mapping rules; fall back to the saved classifier when titles are unusable."""
    if roles_from_titles is not None and any(role is not None for role in roles_from_titles):
        return [role if role is not None else 'Unassigned' for role in roles_from_titles], 'Phase 3 title-mapping rules'

    try:
        import joblib
        from src.job_classifier import PIPELINE_PATH
        from src.nlp_processing import clean_text

        if os.path.exists(PIPELINE_PATH):
            pipeline = joblib.load(PIPELINE_PATH)
            roles = []
            for value in df[find_description_column(df)]:
                text = clean_text(value) if value is not None and not pd.isna(value) else ''
                if len(text.split()) < 3:
                    roles.append('Unassigned')
                    continue
                try:
                    roles.append(str(pipeline.predict([text])[0]))
                except Exception:
                    roles.append('Unassigned')
            return roles, 'Saved Phase 3 model predictions'
    except Exception:
        pass

    return ['Unassigned'] * len(df), 'Unavailable (no job_title column and no saved model)'


def build_role_skill_matrix(roles, records, frequency_table):
    """Percentage of postings within each role that mention each top skill."""
    top_skills = [row['skill'] for row in frequency_table[:TOP_N_MATRIX_SKILLS]]
    if roles is None or not top_skills or records.empty:
        return {'available': False, 'reason': 'Not enough role or skill data.'}

    role_series = pd.Series(roles, index=range(len(roles)), name='role')
    valid_roles = [role for role in role_series.unique() if role != 'Unassigned']
    if not valid_roles:
        return {'available': False, 'reason': 'No postings could be mapped to a broad job role.'}

    matrix = []
    included_roles = []
    for role in valid_roles:
        role_indexes = set(role_series[role_series == role].index)
        role_postings = len(role_indexes)
        if role_postings < MIN_POSTINGS_PER_ROLE:
            continue
        role_records = records[records['posting_id'].isin(role_indexes)]
        row = []
        for skill in top_skills:
            matching = role_records[role_records['skill'] == skill]['posting_id'].nunique()
            row.append(round(matching / role_postings * 100, 1))
        matrix.append(row)
        included_roles.append({'role': role, 'postings': role_postings})

    if not included_roles:
        return {'available': False,
                'reason': f'Each broad role needs at least {MIN_POSTINGS_PER_ROLE} postings to be compared.'}

    return {
        'available': True,
        'roles': included_roles,
        'skills': top_skills,
        'matrix': matrix,
        'note': f'Each cell = share of that role\'s postings mentioning the skill (top {len(top_skills)} skills by frequency).',
    }


def build_forecasting_features(time_series_df, monthly_totals):
    """
    Build the monthly feature table read by the forecaster.

    skill_share normalises job_count so a busier month overall is not mistaken
    for growth in a single skill.
    """
    if time_series_df is None or time_series_df.empty:
        return pd.DataFrame(columns=['month', 'skill', 'job_count', 'skill_share',
                                     'previous_month_count', 'growth_rate', 'trend'])

    rows = []
    for skill in time_series_df.columns:
        series = time_series_df[skill]
        for month in series.index:
            job_count = int(series.loc[month])
            total = int(monthly_totals.get(month, 0)) if monthly_totals is not None else 0
            share = round(job_count / total * 100, 1) if total > 0 else 0.0
            position = series.index.get_loc(month)
            previous_count = int(series.iloc[position - 1]) if position > 0 else np.nan
            if position == 0 or previous_count == 0:
                growth = np.nan
            else:
                growth = round((job_count - previous_count) / previous_count * 100, 1)
            rows.append({
                'month': month,
                'skill': skill,
                'job_count': job_count,
                'skill_share': share,
                'previous_month_count': previous_count,
                'growth_rate': growth,
                'trend': classify_trend(growth),
            })

    features = pd.DataFrame(rows)
    return features.sort_values(['skill', 'month']).reset_index(drop=True)


def save_forecasting_features(features):
    if features is None or features.empty:
        return None
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    path = os.path.join(PROCESSED_DIR, FEATURES_FILENAME)
    features.to_csv(path, index=False)
    return f'data/processed/{FEATURES_FILENAME}'


def identify_emerging_skills(growth_table, total_postings):
    """Skills with enough recent volume and growth above the threshold, ranked by volume."""
    if not growth_table:
        return {'available': False, 'reason': 'No month-over-month data available.',
                'min_recent': MIN_RECENT_MENTIONS_FLOOR, 'items': []}

    min_recent = max(MIN_RECENT_MENTIONS_FLOOR, int(round(total_postings * MIN_RECENT_MENTIONS_SHARE)))
    eligible = []
    for row in growth_table:
        if row['growth'] is None or row['growth'] <= GROWING_THRESHOLD:
            continue
        if row['recent_count'] < min_recent:
            continue
        eligible.append(row)

    eligible.sort(key=lambda row: (row['recent_count'], row['growth']), reverse=True)

    return {
        'available': True,
        'min_recent': min_recent,
        'items': eligible[:6],
        'explanation': (f'Emerging rule: recent month mentions >= {min_recent} '
                        f'(whichever is larger of 5 mentions or 3% of analyzed postings) '
                        f'AND observed growth > +{GROWING_THRESHOLD:.0f}%. Ranked by recent volume.'),
    }


def generate_growth_chart(items, filename, title):
    if not items:
        return None
    os.makedirs(CHARTS_DIR, exist_ok=True)
    filepath = os.path.join(CHARTS_DIR, filename)

    labels = [row[0] for row in items][::-1]
    values = [row[1] for row in items][::-1]
    colors = ['#28785e' if value >= 0 else '#c24b38' for value in values]

    fig, ax = plt.subplots(figsize=(8, max(3.2, len(labels) * 0.45)))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.barh(labels, values, color=colors, height=0.62)
    ax.axvline(0, color='#b7c2b9', linewidth=1)

    for index, value in enumerate(values):
        offset = 1.5 if value >= 0 else -1.5
        ax.text(value + offset, index, f"{value:+.1f}%", va='center',
                ha='left' if value >= 0 else 'right', fontsize=9, color='#24352e')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#e5eae6')
    ax.spines['bottom'].set_color('#e5eae6')
    ax.tick_params(colors='#4c5b50', labelsize=10)
    ax.xaxis.grid(True, linestyle='--', alpha=0.5, color='#e5eae6')
    plt.title(title, fontsize=12, fontweight='bold', color='#24352e', pad=15, loc='left')
    plt.xlabel('Observed growth % (recent month vs previous month)', fontsize=9, color='#738177', labelpad=8)
    plt.tight_layout()
    plt.savefig(filepath, dpi=160)
    plt.close(fig)
    return f'charts/{filename}'


def generate_skill_trend_chart(time_series_df, skills, filename, title):
    if time_series_df is None or not skills:
        return None
    os.makedirs(CHARTS_DIR, exist_ok=True)
    filepath = os.path.join(CHARTS_DIR, filename)

    palette = ['#28785e', '#c98b30', '#3f7fbf', '#a3536f', '#5a8f6b', '#8a6fbf']
    months = list(time_series_df.index)

    fig, ax = plt.subplots(figsize=(8.5, 4))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    for index, skill in enumerate(skills):
        if skill not in time_series_df.columns:
            continue
        ax.plot(months, time_series_df[skill].tolist(), marker='o', linewidth=2,
                markersize=4.5, label=skill, color=palette[index % len(palette)])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#e5eae6')
    ax.spines['bottom'].set_color('#e5eae6')
    ax.tick_params(colors='#4c5b50', labelsize=9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#e5eae6')
    plt.xticks(rotation=35, ha='right')
    ax.set_ylabel('Postings mentioning skill', fontsize=9, color='#738177')
    plt.legend(fontsize=9, frameon=False)
    plt.title(title, fontsize=11, fontweight='bold', color='#24352e', pad=12, loc='left')
    plt.tight_layout()
    plt.savefig(filepath, dpi=160)
    plt.close(fig)
    return f'charts/{filename}'


def generate_role_skill_chart(matrix, filename, title):
    if not matrix or not matrix.get('available'):
        return None
    os.makedirs(CHARTS_DIR, exist_ok=True)
    filepath = os.path.join(CHARTS_DIR, filename)

    roles = [item['role'] for item in matrix['roles']]
    skills = matrix['skills']
    values = matrix['matrix']

    palette = ['#28785e', '#4d9c74', '#79bda0', '#a7d6bf', '#c98b30', '#3f7fbf', '#a3536f', '#8a6fbf']
    y_positions = np.arange(len(roles))
    bar_height = 0.8 / max(1, len(skills))

    fig, ax = plt.subplots(figsize=(9, max(3.5, len(roles) * 0.85)))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    for index, skill in enumerate(skills):
        offsets = y_positions - 0.4 + bar_height * (index + 0.5)
        bars = [row[index] for row in values]
        ax.barh(offsets, bars, height=bar_height, label=skill,
                color=palette[index % len(palette)], edgecolor='none')

    ax.set_yticks(y_positions)
    ax.set_yticklabels(roles, fontsize=9, color='#4c5b50')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#e5eae6')
    ax.spines['bottom'].set_color('#e5eae6')
    ax.tick_params(colors='#4c5b50', labelsize=9)
    ax.xaxis.grid(True, linestyle='--', alpha=0.5, color='#e5eae6')
    ax.legend(fontsize=8, frameon=False, ncol=2, loc='lower right')
    plt.title(title, fontsize=11, fontweight='bold', color='#24352e', pad=12, loc='left')
    plt.xlabel('% of postings in that role mentioning the skill', fontsize=9, color='#738177', labelpad=8)
    plt.tight_layout()
    plt.savefig(filepath, dpi=160)
    plt.close(fig)
    return f'charts/{filename}'


def build_skill_detail(time_series_df, monthly_totals, frequency_table, skill):
    if skill is None or time_series_df is None or skill not in time_series_df.columns:
        return None

    series = time_series_df[skill]
    frequency_row = next((row for row in frequency_table if row['skill'] == skill), None)

    rows = []
    for month in series.index:
        count = int(series.loc[month])
        total = int(monthly_totals.get(month, 0)) if monthly_totals is not None else 0
        share = round(count / total * 100, 1) if total > 0 else 0.0
        position = series.index.get_loc(month)
        previous = int(series.iloc[position - 1]) if position > 0 else None
        if previous is None or previous == 0:
            growth = None
            trend = 'New' if previous == 0 else 'Insufficient data'
        else:
            growth = round((count - previous) / previous * 100, 1)
            trend = classify_trend(growth)
        rows.append({
            'month': month,
            'count': count,
            'share': share,
            'previous_count': previous,
            'growth': growth,
            'trend': trend,
        })

    return {
        'skill': skill,
        'category': SKILL_CATEGORY.get(skill, 'Other'),
        'total_job_count': frequency_row['job_count'] if frequency_row else series.sum(),
        'total_share': frequency_row['share'] if frequency_row else None,
        'months': rows,
    }


def analyze_skill_trends(df, filename='dataset.csv', selected_skill=None):
    """Run the full trend analysis. Never raises; failures return available=False."""
    result = {
        'available': False,
        'message': 'Phase 4 trend analysis could not run for this dataset.',
        'time_series_available': False,
        'month_warning': None,
        'frequency_table': [],
        'growth_table': [],
        'trend_counts': {},
        'role_skill': {'available': False, 'reason': 'Not analyzed.'},
        'emerging': {'available': False, 'reason': 'Not analyzed.', 'items': []},
        'charts': {},
        'feature_rows': [],
        'features_path': None,
        'selected_skill': selected_skill,
        'skill_detail': None,
        'skill_options': [],
        'total_postings': 0,
        'tracked_skills': 0,
        'previous_month': None,
        'recent_month': None,
        'date_column': None,
        'role_source': None,
    }

    desc_col = find_description_column(df)
    if desc_col is None:
        result['message'] = ('Skill trend analysis unavailable because the dataset does not contain '
                             'a job_description column. Phase 1 results are unaffected.')
        return result
    if len(df) == 0:
        result['message'] = 'Skill trend analysis unavailable because the dataset contains no rows.'
        return result

    date_col = find_date_column(df)
    result['date_column'] = date_col

    try:
        records = extract_skill_records(df, desc_col, date_col)
        total_postings = int(df[desc_col].apply(
            lambda value: bool(str(value).strip()) if not pd.isna(value) else False).sum())
        result['total_postings'] = total_postings

        if total_postings == 0 or records.empty:
            result['message'] = ('No skills from the Phase 2 dictionary were found in the job descriptions, '
                                 'so skill trend analysis has nothing to measure.')
            return result

        frequency_table = calculate_skill_frequency(records, total_postings)
        result['available'] = True
        result['frequency_table'] = frequency_table
        result['tracked_skills'] = len(frequency_table)
        result['skill_options'] = [row['skill'] for row in frequency_table]
        result['message'] = None

        monthly_counts, invalid_dates = total_postings_per_month(df, date_col)
        time_series_df, months, monthly_totals = build_skill_time_series(records, monthly_counts)

        if time_series_df is None:
            result['month_warning'] = (
                'Time-based trend analysis is unavailable because no valid date_posted column was found '
                'or the dates could not be parsed. Overall skill frequency and share are still shown.')
        else:
            result['time_series_available'] = True
            result['months'] = months
            result['time_series_available_skills'] = list(time_series_df.columns)
            if invalid_dates:
                result['month_warning'] = (f'{invalid_dates} posting(s) had a missing or invalid date and were '
                                           f'excluded from the monthly trend only.')
            if len(months) < 2:
                result['month_warning'] = ('Only one month of valid dates was found, so month-over-month growth '
                                           'cannot be computed. Frequency and share remain available.')

            growth_table, previous_month, recent_month, growth_note = calculate_skill_growth(time_series_df)
            result['growth_table'] = growth_table
            result['previous_month'] = previous_month
            result['recent_month'] = recent_month
            result['trend_counts'] = count_trends(growth_table)
            result['growth_note'] = growth_note

            features = build_forecasting_features(time_series_df, monthly_totals)
            result['features_path'] = save_forecasting_features(features)
            result['features_preview'] = features.head(12).to_dict(orient='records')
            result['engineered_feature_count'] = int(len(features))

            result['emerging'] = identify_emerging_skills(growth_table, total_postings)

            top_frequency = [(row['skill'], row['job_count']) for row in frequency_table[:10]]
            result['charts']['top_frequency'] = generate_bar_chart(
                top_frequency, 'phase4_top_skills_frequency.png',
                'Phase 4: Top 10 Skills by Frequency', xlabel='Job postings mentioning skill')

            growth_items = [(row['skill'], row['growth']) for row in growth_table
                            if row['growth'] is not None]
            growth_items.sort(key=lambda item: item[1], reverse=True)
            result['charts']['top_growth'] = generate_growth_chart(
                growth_items[:10], 'phase4_top_skills_growth.png',
                'Phase 4: Top 10 Skills by Observed Growth')

            trend_skills = [row['skill'] for row in frequency_table[:5]]
            detail_series = time_series_df[[skill for skill in trend_skills if skill in time_series_df.columns]]
            result['charts']['skill_trend'] = generate_skill_trend_chart(
                detail_series, list(detail_series.columns), 'phase4_skill_trend_over_time.png',
                'Monthly Mentions for the Top 5 Tracked Skills')

            # Kept so /skill-trend can redraw one skill without recomputing trends.
            result['time_series_df'] = time_series_df
            result['monthly_totals'] = monthly_totals

            chosen = selected_skill if selected_skill in time_series_df.columns else (
                frequency_table[0]['skill'] if frequency_table else None)
            if chosen:
                result['selected_skill'] = chosen
                result['skill_detail'] = build_skill_detail(
                    time_series_df, monthly_totals, frequency_table, chosen)
                result['charts']['selected_skill'] = generate_skill_trend_chart(
                    time_series_df[[chosen]], [chosen], 'phase4_selected_skill_trend.png',
                    f'Monthly Trend: {chosen}')

        from src.job_classifier import map_job_title

        title_col = None
        for col in df.columns:
            if str(col).strip().lower() in {'job_title', 'title', 'position'}:
                title_col = col
                break
        roles_from_titles = None
        if title_col is not None:
            roles_from_titles = [map_job_title(value) for value in df[title_col]]

        assigned_roles, role_source = assign_broad_roles(df, roles_from_titles)
        result['role_source'] = role_source
        role_skill = build_role_skill_matrix(assigned_roles, records, frequency_table)
        result['role_skill'] = role_skill

        if role_skill.get('available'):
            result['charts']['role_skill'] = generate_role_skill_chart(
                role_skill, 'phase4_role_skill_matrix.png',
                'Phase 4: Skill Presence by Broad Job Role')

        return result

    except Exception as error:
        result['available'] = False
        result['message'] = f'Phase 4 trend analysis stopped safely: {error}'
        return result


def render_selected_skill(trend_result, skill):
    time_series_df = trend_result.get('time_series_df')
    if time_series_df is None or skill not in getattr(time_series_df, 'columns', []):
        return trend_result, None

    detail = build_skill_detail(time_series_df, trend_result.get('monthly_totals'),
                                trend_result.get('frequency_table', []), skill)
    chart = generate_skill_trend_chart(time_series_df[[skill]], [skill],
                                       'phase4_selected_skill_trend.png',
                                       f'Monthly Trend: {skill}')
    trend_result['selected_skill'] = skill
    trend_result['skill_detail'] = detail
    trend_result.setdefault('charts', {})['selected_skill'] = chart
    return trend_result, chart
