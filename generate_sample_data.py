"""
Script to create sample_jobs.csv for quick development and interview demonstration.
Contains ~240 realistic, synthetic job posting records across all 6 roles.
"""

import csv
import os
import random

random.seed(42)

ROLES_DATA = [
    {
        'title': 'Senior Software Engineer',
        'desc_templates': [
            "We are seeking an experienced Senior Software Engineer to design and develop scalable backend microservices using Java, Spring Boot, and REST APIs. Experience with Docker and SQL databases required. Collaborate with product managers and mentors juniors.",
            "Full stack software developer needed to build responsive web applications with JavaScript, React, HTML, CSS, and Node.js. Knowledge of Git, MySQL, and clean software architecture required.",
            "Backend engineer to maintain and build reliable software platforms using Java, Python, and C++. Work with relational databases like PostgreSQL, implement unit tests, and support CI/CD pipelines.",
            "Software engineer responsible for enterprise application services. Proficient in Java, JavaScript, and Docker. Experience debugging production issues and writing maintainable code."
        ]
    },
    {
        'title': 'Data Scientist',
        'desc_templates': [
            "Join our core analytics team as a Data Scientist to formulate predictive models, analyze large data sets using Python, Pandas, NumPy, and Scikit-learn. Formulate hypotheses and present findings to leadership.",
            "Data Scientist required to develop machine learning algorithms and statistical models in Python. Experience with Pandas, data cleaning, exploratory data analysis, and feature engineering is essential.",
            "Looking for a Data Scientist with expertise in statistical inference, machine learning, Python, NumPy, and SQL. Conduct A/B testing and build automated decision systems.",
            "Applied Data Scientist to analyze complex data patterns. Must be proficient in Python, Scikit-learn, Pandas, and data visualization. Experience collaborating with product engineering teams."
        ]
    },
    {
        'title': 'Data Analyst',
        'desc_templates': [
            "Seeking a Data Analyst to extract business metrics, clean datasets, and build executive dashboards using SQL, Excel, and Python. Must have strong analytical communication skills.",
            "Business Data Analyst responsible for translating business questions into SQL queries, data summaries, and reports. Basic knowledge of Python, Pandas, and data visualization tools is preferred.",
            "Data Analyst needed to monitor company KPIs, validate data pipelines, and write optimized SQL queries. Must be comfortable cleaning messy data and presenting findings to stakeholders.",
            "Analytics professional to build recurring reports and perform ad-hoc analysis using SQL and Pandas. Identify trends in customer data and collaborate across business units."
        ]
    },
    {
        'title': 'Machine Learning Engineer',
        'desc_templates': [
            "We are hiring an AI/ML Engineer to deploy deep learning models into production using Python, PyTorch, TensorFlow, and AWS. Work closely with data scientists to optimize inference pipelines.",
            "Machine Learning Engineer with strong background in NLP, deep learning, Python, and Docker. Experience building scalable ML systems and monitoring models in cloud environments.",
            "AI/ML Engineer to research and implement generative AI and LLM pipelines using PyTorch, Hugging Face, and RAG architectures. Strong Python programming and cloud deployment skills required.",
            "Machine learning developer to build automated training workflows and maintain predictive pipelines using TensorFlow, Python, and Kubernetes. Strong mathematical foundations required."
        ]
    },
    {
        'title': 'Cloud DevOps Engineer',
        'desc_templates': [
            "DevOps and Cloud Engineer to manage infrastructure as code, automate CI/CD pipelines, and administer containerized clusters using Docker, Kubernetes, and AWS.",
            "Cloud Infrastructure Engineer required to maintain high availability in Azure and AWS. Experience with Docker, Kubernetes, Linux systems, and automated monitoring tools.",
            "Site Reliability and DevOps Engineer to improve system reliability, configure cloud networks, and automate server provisioning using AWS, Docker, and shell scripting.",
            "Platform DevOps Engineer to support software developers with deployment automation, Kubernetes orchestration, and cloud security monitoring in an agile environment."
        ]
    },
    {
        'title': 'Cybersecurity Analyst',
        'desc_templates': [
            "Seeking a Cybersecurity Engineer to monitor security operations, evaluate vulnerabilities, and maintain network security defenses across enterprise systems.",
            "Information Security and Cybersecurity Analyst to conduct penetration testing, manage SOC alerts, and implement network security policies to protect corporate data assets.",
            "Cybersecurity Specialist responsible for incident response, threat detection, and firewall administration. Strong understanding of cybersecurity frameworks and vulnerability scanning.",
            "Security engineer to safeguard cloud infrastructure and enterprise applications. Experience in network security, cybersecurity audits, and compliance monitoring."
        ]
    }
]

COMPANIES = ['Accenture', 'Infosys', 'TCS', 'Deloitte', 'Cognizant', 'Wipro', 'Microsoft', 'Amazon', 'IBM', 'Oracle']
LOCATIONS = ['Bengaluru', 'Hyderabad', 'Pune', 'Mumbai', 'Delhi NCR', 'Chennai', 'Remote']
EMPLOYMENT_TYPES = ['Full-time', 'Full-time', 'Full-time', 'Contract', 'Internship']
EXPERIENCE_LEVELS = ['Entry-level', 'Mid-level', 'Mid-level', 'Senior-level']

output_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'sample_jobs.csv')

with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([
        'job_title', 'company', 'location', 'job_description',
        'date_posted', 'employment_type', 'experience_level', 'salary'
    ])

    for i in range(240):
        role_info = random.choice(ROLES_DATA)
        title = role_info['title']
        desc = random.choice(role_info['desc_templates'])

        # Introduce realistic variation
        company = random.choice(COMPANIES) if random.random() > 0.03 else ''
        location = random.choice(LOCATIONS) if random.random() > 0.02 else ''
        month = random.randint(1, 6)
        day = random.randint(1, 28)
        date_posted = f"2026-0{month:01d}-{day:02d}"
        emp_type = random.choice(EMPLOYMENT_TYPES)
        exp_level = random.choice(EXPERIENCE_LEVELS)
        salary = str(random.randint(45, 180) * 10000) if random.random() > 0.15 else ''

        writer.writerow([
            title, company, location, desc,
            date_posted, emp_type, exp_level, salary
        ])

print(f"Created {output_file} with 240 sample records.")
