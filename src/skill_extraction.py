"""Dictionary-based skill extraction from job descriptions."""

from collections import Counter
from src.nlp_processing import clean_text

SKILL_DICTIONARY = [
    {'name': 'Python', 'category': 'Programming', 'aliases': ['python', 'python3']},
    {'name': 'Java', 'category': 'Programming', 'aliases': ['java', 'core java']},
    {'name': 'C++', 'category': 'Programming', 'aliases': ['c++', 'cpp']},
    {'name': 'JavaScript', 'category': 'Programming', 'aliases': ['javascript', 'js', 'es6']},
    {'name': 'SQL', 'category': 'Programming', 'aliases': ['sql', 'structured query language']},

    {'name': 'Machine Learning', 'category': 'Data / ML', 'aliases': ['machine learning', 'ml']},
    {'name': 'Deep Learning', 'category': 'Data / ML', 'aliases': ['deep learning', 'dl']},
    {'name': 'NLP', 'category': 'Data / ML', 'aliases': ['nlp', 'natural language processing']},
    {'name': 'TensorFlow', 'category': 'Data / ML', 'aliases': ['tensorflow', 'tensor flow']},
    {'name': 'PyTorch', 'category': 'Data / ML', 'aliases': ['pytorch', 'torch']},
    {'name': 'Pandas', 'category': 'Data / ML', 'aliases': ['pandas']},
    {'name': 'NumPy', 'category': 'Data / ML', 'aliases': ['numpy']},
    {'name': 'Scikit-learn', 'category': 'Data / ML', 'aliases': ['scikit-learn', 'scikit learn', 'sklearn']},

    {'name': 'HTML', 'category': 'Web', 'aliases': ['html', 'html5']},
    {'name': 'CSS', 'category': 'Web', 'aliases': ['css', 'css3']},
    {'name': 'React', 'category': 'Web', 'aliases': ['react', 'reactjs', 'react.js']},
    {'name': 'Node.js', 'category': 'Web', 'aliases': ['node.js', 'nodejs', 'node js', 'node']},

    {'name': 'AWS', 'category': 'Cloud / DevOps', 'aliases': ['aws', 'amazon web services']},
    {'name': 'Azure', 'category': 'Cloud / DevOps', 'aliases': ['azure', 'microsoft azure']},
    {'name': 'Docker', 'category': 'Cloud / DevOps', 'aliases': ['docker']},
    {'name': 'Kubernetes', 'category': 'Cloud / DevOps', 'aliases': ['kubernetes', 'k8s']},

    {'name': 'MySQL', 'category': 'Databases', 'aliases': ['mysql']},
    {'name': 'PostgreSQL', 'category': 'Databases', 'aliases': ['postgresql', 'postgres']},
    {'name': 'MongoDB', 'category': 'Databases', 'aliases': ['mongodb', 'mongo']},

    {'name': 'Cybersecurity', 'category': 'Cybersecurity', 'aliases': ['cybersecurity', 'cyber security', 'infosec', 'information security']},
    {'name': 'Network Security', 'category': 'Cybersecurity', 'aliases': ['network security']},

    {'name': 'Generative AI', 'category': 'AI', 'aliases': ['generative ai', 'genai', 'gen ai']},
    {'name': 'LLM', 'category': 'AI', 'aliases': ['llm', 'llms', 'large language model', 'large language models']},
    {'name': 'RAG', 'category': 'AI', 'aliases': ['rag', 'retrieval augmented generation', 'retrieval-augmented generation']}
]

# Aliases are cleaned with the same function as descriptions so matching stays consistent.
COMPILED_SKILLS = []
for skill in SKILL_DICTIONARY:
    cleaned_aliases = [clean_text(a) for a in skill['aliases'] if clean_text(a)]
    COMPILED_SKILLS.append({
        'name': skill['name'],
        'category': skill['category'],
        'aliases': list(set(cleaned_aliases))
    })


def extract_skills_from_text(text):
    """Return each matched skill at most once; matches whole phrases only."""
    if not text:
        return []
    cleaned = clean_text(text)
    # Space padding gives whole-word matching, so "java" never matches "javascript".
    padded = f" {cleaned} "

    matched = []
    for skill in COMPILED_SKILLS:
        for alias in skill['aliases']:
            if f" {alias} " in padded:
                matched.append(skill['name'])
                break

    return matched


def analyze_skills_in_series(descriptions_series):
    if descriptions_series is None or len(descriptions_series) == 0:
        return {
            'available': False,
            'message': 'No job description data found.'
        }

    total_rows = len(descriptions_series)
    non_empty_count = 0
    empty_count = 0
    short_count = 0
    postings_with_skills = 0
    skill_counts = Counter()

    example_processed = None

    for val in descriptions_series:
        if val is None or (isinstance(val, float) and val != val):
            empty_count += 1
            continue

        text_str = str(val).strip()
        if not text_str or text_str.lower() in {'nan', 'na', 'null', 'none', '<na>'}:
            empty_count += 1
            continue

        non_empty_count += 1
        if len(text_str) < 20:
            short_count += 1

        skills = extract_skills_from_text(text_str)
        if skills:
            postings_with_skills += 1
            for s in skills:
                skill_counts[s] += 1

        if example_processed is None and len(skills) >= 2:
            from src.nlp_processing import preprocess_text
            example_processed = {
                'raw': text_str[:350],
                'cleaned': clean_text(text_str)[:350],
                'skills': skills,
                'tokens': preprocess_text(text_str)['keywords'][:15]
            }

    if example_processed is None and non_empty_count > 0:
        for val in descriptions_series:
            text_str = str(val).strip()
            if text_str and text_str.lower() not in {'nan', 'na', 'null', 'none'}:
                from src.nlp_processing import preprocess_text
                example_processed = {
                    'raw': text_str[:350],
                    'cleaned': clean_text(text_str)[:350],
                    'skills': extract_skills_from_text(text_str),
                    'tokens': preprocess_text(text_str)['keywords'][:15]
                }
                break

    skill_freq_list = []
    category_map = {s['name']: s['category'] for s in SKILL_DICTIONARY}
    for skill_name, count in skill_counts.most_common():
        percentage = round((count / non_empty_count * 100), 1) if non_empty_count > 0 else 0
        skill_freq_list.append({
            'name': skill_name,
            'category': category_map.get(skill_name, 'Other'),
            'count': count,
            'percentage': percentage
        })

    return {
        'available': True,
        'total_descriptions': total_rows,
        'processed_count': non_empty_count,
        'empty_count': empty_count,
        'short_count': short_count,
        'postings_with_skills': postings_with_skills,
        'unique_skills_found': len(skill_freq_list),
        'dictionary_size': len(SKILL_DICTIONARY),
        'top_skills': skill_freq_list,
        'example_processed': example_processed
    }
