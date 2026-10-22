"""TF-IDF + Logistic Regression classifier for broad job-role categories."""

import os
import re
from collections import Counter
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.nlp_processing import clean_text

ROLE_NAMES = [
    'Software Engineer',
    'Data Scientist',
    'Data Analyst',
    'AI/ML Engineer',
    'DevOps/Cloud Engineer',
    'Cybersecurity Engineer'
]

# Order matters: the first matching role wins.
MAPPING_RULES = [
    ('Cybersecurity Engineer', [
        r'\bcyber\s*security\b', r'\binformation\s+security\b',
        r'\bsecurity\s+(?:analyst|engineer|specialist|architect)\b',
        r'\b(?:soc\s+analyst|penetration\s+tester|infosec)\b'
    ]),
    ('AI/ML Engineer', [
        r'\b(?:machine\s+learning|deep\s+learning|ml|ai)\s+(?:engineer|developer|specialist|researcher)\b',
        r'\b(?:mlops|nlp|computer\s+vision)\s+(?:engineer|developer)\b'
    ]),
    ('Data Scientist', [
        r'\bdata\s+scientist\b', r'\bdata\s+science\b', r'\bdecision\s+scientist\b'
    ]),
    ('Data Analyst', [
        r'\bdata\s+analyst\b', r'\bbusiness\s+analyst\b', r'\bbi\s+analyst\b',
        r'\banalytics?\s+analyst\b', r'\breporting\s+analyst\b', r'\bdata\s+analytics\b'
    ]),
    ('DevOps/Cloud Engineer', [
        r'\bdevops\b', r'\bcloud\s+(?:engineer|developer|architect|consultant)\b',
        r'\bsite\s+reliability\s+engineer\b', r'\bsre\b', r'\bplatform\s+engineer\b',
        r'\binfrastructure\s+engineer\b'
    ]),
    ('Software Engineer', [
        r'\bsoftware\s+(?:engineer|developer)\b',
        r'\b(?:back\s*end|front\s*end|full\s*stack)\s+(?:engineer|developer)\b',
        r'\bweb\s+developer\b', r'\b(?:java|python|javascript|c\+\+|dotnet|\.net)\s+developer\b',
        r'\bapplication\s+developer\b', r'\bsystem\s+engineer\b'
    ])
]

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
PIPELINE_PATH = os.path.join(MODELS_DIR, 'job_classifier_pipeline.pkl')
MODEL_PATH = os.path.join(MODELS_DIR, 'job_classifier.pkl')
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')


def map_job_title(title):
    """Map a detailed title to a broad role, or None; unmapped titles are never guessed."""
    if not title or pd.isna(title):
        return None

    cleaned_title = re.sub(r'[^a-z0-9]+', ' ', str(title).lower()).strip()

    for role, patterns in MAPPING_RULES:
        for pattern in patterns:
            if re.search(pattern, cleaned_title):
                return role
    return None


def train_job_classifier(df, min_per_class=10, test_size=0.2, random_state=42):
    if df is None or len(df) == 0:
        return {'status': 'unavailable', 'message': 'Dataset is empty.'}

    cols = {re.sub(r'[\s-]+', '_', str(c).strip().lower()): c for c in df.columns}

    if 'job_description' not in cols:
        return {
            'status': 'unavailable',
            'message': 'Classification unavailable: Dataset does not contain a job_description column.'
        }

    if 'job_title' not in cols:
        return {
            'status': 'unavailable',
            'message': 'Classification unavailable: Dataset does not contain a job_title column to derive training labels.'
        }

    titles = df[cols['job_title']]
    descriptions = df[cols['job_description']]

    usable_texts = []
    usable_labels = []
    seen_texts = {}
    ambiguous_texts = set()

    skipped_reasons = {
        'unsupported_title': 0,
        'empty_description': 0,
        'short_description': 0,
        'duplicate_description': 0,
        'ambiguous_description': 0
    }

    for raw_title, raw_desc in zip(titles, descriptions):
        role = map_job_title(raw_title)
        if not role:
            skipped_reasons['unsupported_title'] += 1
            continue

        if not raw_desc or pd.isna(raw_desc):
            skipped_reasons['empty_description'] += 1
            continue

        text_str = str(raw_desc).strip()
        cleaned = clean_text(text_str)
        tokens = cleaned.split()
        if len(tokens) < 3:
            skipped_reasons['short_description'] += 1
            continue

        # Identical texts in both splits would leak test data into training.
        if cleaned in seen_texts:
            skipped_reasons['duplicate_description'] += 1
            if seen_texts[cleaned] != role:
                ambiguous_texts.add(cleaned)
            continue

        seen_texts[cleaned] = role
        usable_texts.append(cleaned)
        usable_labels.append(role)

    if ambiguous_texts:
        filtered_texts = []
        filtered_labels = []
        for t, l in zip(usable_texts, usable_labels):
            if t not in ambiguous_texts:
                filtered_texts.append(t)
                filtered_labels.append(l)
            else:
                skipped_reasons['ambiguous_description'] += 1
        usable_texts = filtered_texts
        usable_labels = filtered_labels

    class_counts = Counter(usable_labels)
    valid_classes = [r for r in ROLE_NAMES if class_counts[r] >= min_per_class]

    if len(valid_classes) < 2 or len(usable_texts) < 20:
        return {
            'status': 'unavailable',
            'message': f'Insufficient labeled records. Found {len(valid_classes)} role classes with at least {min_per_class} distinct samples. At least 2 classes are required.',
            'found_classes': dict(class_counts),
            'skipped': skipped_reasons
        }

    final_texts = []
    final_labels = []
    for t, l in zip(usable_texts, usable_labels):
        if l in valid_classes:
            final_texts.append(t)
            final_labels.append(l)

    X_train, X_test, y_train, y_test = train_test_split(
        final_texts,
        final_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=final_labels
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000, lowercase=False)
    classifier = LogisticRegression(max_iter=400, random_state=random_state)
    pipeline = Pipeline([
        ('tfidf', vectorizer),
        ('clf', classifier)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = float(accuracy_score(y_test, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred, labels=valid_classes)

    eval_examples = []
    for actual, pred, raw_t in zip(y_test[:8], y_pred[:8], X_test[:8]):
        eval_examples.append({
            'actual': actual,
            'predicted': pred,
            'excerpt': raw_t[:150] + ('...' if len(raw_t) > 150 else '')
        })

    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(pipeline, PIPELINE_PATH)
        joblib.dump(vectorizer, VECTORIZER_PATH)
        joblib.dump(classifier, MODEL_PATH)
    except Exception as e:
        print(f"Warning: could not persist models: {e}")

    report = {
        'status': 'trained',
        'total_records_processed': len(df),
        'usable_records': len(final_texts),
        'train_records': len(X_train),
        'test_records': len(X_test),
        'classes': valid_classes,
        'class_counts': [{'role': r, 'count': class_counts[r]} for r in valid_classes],
        'metrics': {
            'accuracy': round(accuracy, 3),
            'precision': round(float(precision), 3),
            'recall': round(float(recall), 3),
            'f1_score': round(float(f1), 3)
        },
        'confusion_matrix': cm.tolist(),
        'eval_examples': eval_examples,
        'vocabulary_size': len(pipeline.named_steps['tfidf'].vocabulary_),
        'skipped': skipped_reasons
    }
    return report


def predict_job_role(text, pipeline=None):
    if not text or not str(text).strip():
        return {
            'status': 'error',
            'message': 'Please enter a job description to predict.'
        }

    cleaned = clean_text(text)
    if len(cleaned.split()) < 3:
        return {
            'status': 'error',
            'message': 'Please enter a more detailed job description (at least 3 words).'
        }

    if pipeline is None:
        if os.path.exists(PIPELINE_PATH):
            try:
                pipeline = joblib.load(PIPELINE_PATH)
            except Exception as e:
                return {
                    'status': 'error',
                    'message': f'Failed to load trained model: {str(e)}'
                }
        else:
            return {
                'status': 'error',
                'message': 'No trained model found. Please upload a dataset to train the classifier first.'
            }

    features = pipeline.named_steps['tfidf'].transform([cleaned])
    if features.nnz == 0:
        return {
            'status': 'error',
            'message': 'The description does not contain any vocabulary words learned during training.'
        }

    probabilities = pipeline.named_steps['clf'].predict_proba(features)[0]
    best_idx = np.argmax(probabilities)
    predicted_role = pipeline.named_steps['clf'].classes_[best_idx]
    confidence = float(probabilities[best_idx])

    all_probs = []
    for cls_name, prob in zip(pipeline.named_steps['clf'].classes_, probabilities):
        all_probs.append({
            'role': cls_name,
            'probability': round(float(prob) * 100, 1)
        })
    all_probs.sort(key=lambda x: x['probability'], reverse=True)

    return {
        'status': 'success',
        'predicted_role': str(predicted_role),
        'confidence': round(confidence * 100, 1),
        'all_probabilities': all_probs
    }
