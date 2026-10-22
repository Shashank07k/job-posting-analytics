"""Exploratory analysis and Matplotlib charts for uploaded job datasets."""

import os
import re
import matplotlib
matplotlib.use('Agg')  # headless backend for the Flask server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# None of these are mandatory; analysis adapts to whichever are present.
EXPECTED_COLUMNS = [
    'job_title', 'company', 'location', 'job_description',
    'date_posted', 'employment_type', 'experience_level', 'salary'
]

CHARTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static', 'charts')


def normalize_col_name(col):
    return re.sub(r'[\s-]+', '_', str(col).strip().lower())


def get_column_mapping(df):
    normalized_map = {}
    for col in df.columns:
        norm = normalize_col_name(col)
        normalized_map[norm] = col

    mapping = {}
    present_expected = []
    missing_expected = []

    for exp in EXPECTED_COLUMNS:
        if exp in normalized_map:
            mapping[exp] = normalized_map[exp]
            present_expected.append(exp)
        else:
            missing_expected.append(exp)

    return mapping, present_expected, missing_expected


def compute_basic_statistics(df):
    stats_list = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Text columns that are mostly numeric (e.g. salary stored as strings) are included too.
    for col in df.columns:
        if col not in numeric_cols:
            converted = pd.to_numeric(df[col], errors='coerce')
            valid_ratio = converted.notna().sum() / len(df) if len(df) > 0 else 0
            if valid_ratio > 0.8:
                df_col_numeric = converted
                stats_list.append({
                    'column': col,
                    'count': int(df_col_numeric.count()),
                    'mean': round(float(df_col_numeric.mean()), 2) if df_col_numeric.count() > 0 else 0,
                    'std': round(float(df_col_numeric.std()), 2) if df_col_numeric.count() > 1 else 0,
                    'min': round(float(df_col_numeric.min()), 2) if df_col_numeric.count() > 0 else 0,
                    'q25': round(float(df_col_numeric.quantile(0.25)), 2) if df_col_numeric.count() > 0 else 0,
                    'median': round(float(df_col_numeric.median()), 2) if df_col_numeric.count() > 0 else 0,
                    'q75': round(float(df_col_numeric.quantile(0.75)), 2) if df_col_numeric.count() > 0 else 0,
                    'max': round(float(df_col_numeric.max()), 2) if df_col_numeric.count() > 0 else 0
                })

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) > 0:
            stats_list.append({
                'column': col,
                'count': int(series.count()),
                'mean': round(float(series.mean()), 2),
                'std': round(float(series.std()), 2) if len(series) > 1 else 0,
                'min': round(float(series.min()), 2),
                'q25': round(float(series.quantile(0.25)), 2),
                'median': round(float(series.median()), 2),
                'q75': round(float(series.quantile(0.75)), 2),
                'max': round(float(series.max()), 2)
            })

    return stats_list


def generate_bar_chart(items, filename, title, xlabel="Count", color="#28785e"):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    filepath = os.path.join(CHARTS_DIR, filename)

    if not items:
        return None

    labels = [str(x[0]) for x in items]
    values = [x[1] for x in items]

    # barh draws bottom-up; reverse so the largest value is on top.
    labels = labels[::-1]
    values = values[::-1]

    fig, ax = plt.subplots(figsize=(8, max(3.5, len(labels) * 0.45)))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    bars = ax.barh(labels, values, color=color, height=0.65, edgecolor='none')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#e5eae6')
    ax.spines['bottom'].set_color('#e5eae6')
    ax.tick_params(colors='#4c5b50', labelsize=10)
    ax.xaxis.grid(True, linestyle='--', alpha=0.5, color='#e5eae6')

    for bar in bars:
        width = bar.get_width()
        ax.text(width + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{int(width):,}",
                va='center', ha='left', color='#24352e', fontsize=9, fontweight='500')

    plt.title(title, fontsize=12, fontweight='bold', color='#24352e', pad=15, loc='left')
    plt.xlabel(xlabel, fontsize=10, color='#738177', labelpad=10)
    plt.tight_layout()
    plt.savefig(filepath, dpi=160)
    plt.close(fig)

    return f"charts/{filename}"


def generate_pie_chart(items, filename, title):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    filepath = os.path.join(CHARTS_DIR, filename)

    if not items:
        return None

    labels = [str(x[0]) for x in items[:5]]
    values = [x[1] for x in items[:5]]

    if len(items) > 5:
        other_sum = sum(x[1] for x in items[5:])
        labels.append('Other')
        values.append(other_sum)

    colors = ['#28785e', '#459779', '#6eb396', '#99cfb6', '#c6ebd9', '#e0f2e9']

    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors[:len(values)],
        pctdistance=0.75,
        textprops={'color': '#24352e', 'fontsize': 9},
        wedgeprops=dict(width=0.45, edgecolor='#ffffff', linewidth=2)
    )

    for autotext in autotexts:
        autotext.set_color('#ffffff')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(8)

    plt.title(title, fontsize=11, fontweight='bold', color='#24352e', pad=15, loc='left')
    plt.tight_layout()
    plt.savefig(filepath, dpi=160)
    plt.close(fig)

    return f"charts/{filename}"


def generate_trend_chart(df, date_col, filename):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    filepath = os.path.join(CHARTS_DIR, filename)

    try:
        parsed_dates = pd.to_datetime(df[date_col], errors='coerce')
        valid_dates = parsed_dates.dropna()
        if len(valid_dates) < 3:
            return None

        monthly_counts = valid_dates.dt.to_period('M').value_counts().sort_index()
        if len(monthly_counts) < 2:
            return None

        labels = [str(period) for period in monthly_counts.index]
        values = monthly_counts.values

        fig, ax = plt.subplots(figsize=(8, 3.8))
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#ffffff')

        ax.plot(labels, values, marker='o', color='#28785e', linewidth=2.2, markersize=5, markerfacecolor='#ffffff', markeredgewidth=2)
        ax.fill_between(range(len(labels)), values, color='#eaf3ed', alpha=0.6)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#e5eae6')
        ax.spines['bottom'].set_color('#e5eae6')
        ax.tick_params(colors='#4c5b50', labelsize=9)
        ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#e5eae6')
        plt.xticks(rotation=35, ha='right')

        plt.title('Job Posting Volume Over Time (Monthly)', fontsize=11, fontweight='bold', color='#24352e', pad=12, loc='left')
        plt.xlabel('Posting Month', fontsize=9, color='#738177', labelpad=8)
        plt.ylabel('Postings Count', fontsize=9, color='#738177', labelpad=8)
        plt.tight_layout()
        plt.savefig(filepath, dpi=160)
        plt.close(fig)

        return f"charts/{filename}"
    except Exception as e:
        print(f"Error generating trend chart: {e}")
        return None


def perform_eda(df, filename="dataset.csv"):
    mapping, present_expected, missing_expected = get_column_mapping(df)

    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols

    missing_counts = df.isna().sum().to_dict()
    missing_percentages = {col: round((count / total_rows * 100), 1) if total_rows > 0 else 0
                           for col, count in missing_counts.items()}
    total_missing_cells = sum(missing_counts.values())
    overall_missing_pct = round((total_missing_cells / total_cells * 100), 1) if total_cells > 0 else 0
    data_completeness = round(100.0 - overall_missing_pct, 1)

    duplicate_count = int(df.duplicated().sum())

    columns_info = []
    for col in df.columns:
        series = df[col]
        non_missing = int(series.count())
        missing = int(total_rows - non_missing)
        missing_pct = missing_percentages.get(col, 0)
        unique_vals = int(series.nunique(dropna=True))
        dtype_str = str(series.dtype)

        columns_info.append({
            'name': col,
            'dtype': dtype_str,
            'non_missing': non_missing,
            'missing': missing,
            'missing_pct': missing_pct,
            'unique_count': unique_vals
        })

    numerical_stats = compute_basic_statistics(df)

    charts = {}

    top_titles = []
    if 'job_title' in mapping:
        col = mapping['job_title']
        vc = df[col].dropna().value_counts().head(8)
        top_titles = list(zip(vc.index, vc.values))
        charts['job_titles'] = generate_bar_chart(top_titles, 'top_job_titles.png', 'Top In-Demand Job Titles', color='#28785e')

    top_companies = []
    if 'company' in mapping:
        col = mapping['company']
        vc = df[col].dropna().value_counts().head(8)
        top_companies = list(zip(vc.index, vc.values))
        charts['companies'] = generate_bar_chart(top_companies, 'top_companies.png', 'Top Hiring Companies', color='#3b8a6f')

    top_locations = []
    if 'location' in mapping:
        col = mapping['location']
        vc = df[col].dropna().value_counts().head(8)
        top_locations = list(zip(vc.index, vc.values))
        charts['locations'] = generate_bar_chart(top_locations, 'top_locations.png', 'Top Job Locations', color='#4f9c81')

    employment_dist = []
    if 'employment_type' in mapping:
        col = mapping['employment_type']
        vc = df[col].dropna().value_counts()
        employment_dist = list(zip(vc.index, vc.values))
        charts['employment_type'] = generate_pie_chart(employment_dist, 'employment_type.png', 'Employment Type Distribution')

    experience_dist = []
    if 'experience_level' in mapping:
        col = mapping['experience_level']
        vc = df[col].dropna().value_counts()
        experience_dist = list(zip(vc.index, vc.values))
        charts['experience_level'] = generate_pie_chart(experience_dist, 'experience_level.png', 'Experience Level Distribution')

    if 'date_posted' in mapping:
        col = mapping['date_posted']
        charts['posting_trend'] = generate_trend_chart(df, col, 'posting_trend.png')

    preview_df = df.head(10).fillna('')
    preview_rows = preview_df.to_dict(orient='records')

    return {
        'filename': filename,
        'total_rows': total_rows,
        'total_cols': total_cols,
        'total_cells': total_cells,
        'total_missing_cells': total_missing_cells,
        'overall_missing_pct': overall_missing_pct,
        'data_completeness': data_completeness,
        'duplicate_count': duplicate_count,
        'present_expected': present_expected,
        'missing_expected': missing_expected,
        'columns_info': columns_info,
        'numerical_stats': numerical_stats,
        'top_titles': top_titles,
        'top_companies': top_companies,
        'top_locations': top_locations,
        'employment_dist': employment_dist,
        'experience_dist': experience_dist,
        'charts': charts,
        'preview_headers': list(df.columns),
        'preview_rows': preview_rows
    }
