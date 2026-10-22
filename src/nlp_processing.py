"""Lightweight text preprocessing for job descriptions."""

import re

STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any',
    'are', 'aren', 'arent', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below',
    'between', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', 'couldnt', 'd',
    'did', 'didn', 'didnt', 'do', 'does', 'doesn', 'doesnt', 'doing', 'don', 'dont', 'down',
    'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', 'hadnt', 'has', 'hasn',
    'hasnt', 'have', 'haven', 'havent', 'having', 'he', 'her', 'here', 'hers', 'herself',
    'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'isn', 'isnt', 'it', 'its',
    'itself', 'just', 'll', 'm', 'ma', 'me', 'might', 'mightn', 'mightnt', 'more', 'most',
    'must', 'mustn', 'mustnt', 'my', 'myself', 'needn', 'neednt', 'no', 'nor', 'not', 'now',
    'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out',
    'over', 'own', 're', 's', 'same', 'shan', 'shant', 'she', 'should', 'shouldn', 'shouldnt',
    'so', 'some', 'such', 't', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves',
    'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under',
    'until', 'up', 've', 'very', 'was', 'wasn', 'wasnt', 'we', 'were', 'weren', 'werent',
    'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won',
    'wont', 'would', 'wouldn', 'wouldnt', 'y', 'you', 'your', 'yours', 'yourself', 'yourselves'
}


def clean_text(text):
    if text is None:
        return ""
    text = str(text).lower()

    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+\.\S+', ' ', text)

    # Keep +, # and inner dots so tokens like c++, c# and node.js survive.
    text = re.sub(r'[^a-z0-9+#.\s]', ' ', text)
    text = re.sub(r'\.(?![a-z])', ' ', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text


def tokenize_text(text):
    if not text:
        return []
    return [token for token in text.split(' ') if token]


def remove_stopwords(tokens):
    return [
        token for token in tokens
        if token not in STOPWORDS and not re.match(r'^[0-9.]+$', token)
    ]


def lemmatize_token(word):
    """Light suffix stripping; short words and tech terms like 'c++' are left as-is."""
    if len(word) < 5 or any(char in word for char in ('+', '#', '.')):
        return word
    if word.endswith('ies'):
        return word[:-3] + 'y'
    if word.endswith('sses'):
        return word[:-2]
    if word.endswith('s') and not word.endswith(('ss', 'us', 'is')):
        return word[:-1]
    if word.endswith('ing') and len(word) > 6:
        return word[:-3]
    if word.endswith('ed') and len(word) > 5:
        return word[:-2]
    return word


def preprocess_text(text):
    cleaned = clean_text(text)
    tokens = tokenize_text(cleaned)
    stopwords_removed = remove_stopwords(tokens)
    lemmatized = [lemmatize_token(t) for t in stopwords_removed]

    return {
        'raw': text,
        'cleaned': cleaned,
        'tokens': tokens,
        'tokens_count': len(tokens),
        'keywords': lemmatized,
        'processed_text': ' '.join(lemmatized)
    }
