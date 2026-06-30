"""
ai/extractor.py
Phase 2: Structured Field Extraction
Pulls skills, years of experience, email, and education level out of raw text.
Works for both resumes and job descriptions (same underlying logic).
"""

import re

# A reasonably broad skills dictionary - extend this as needed for your domain.
# Using a fixed vocabulary (rather than NER) keeps this fast and deterministic,
# which makes it easy to explain and evaluate in a report.
SKILLS_VOCAB = [
     "python", "java", "javascript", "typescript", "c++", "c#", "sql", "nosql",
    "react", "angular", "vue", "node.js", "django", "flask", "fastapi",
    "machine learning", "deep learning", "data science", "nlp",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
    "docker", "kubernetes", "aws", "azure", "gcp", "git", "linux",
    "html", "css", "rest api", "graphql", "mongodb", "postgresql", "mysql",
    "excel", "tableau", "power bi", "communication", "leadership",
    "project management", "agile", "scrum", "data analysis", "statistics",
    "matplotlib", "seaborn", "plotly", "keras", "opencv", "spark",
    "hadoop", "r", "julia", "go", "rust", "swift", "kotlin", "php",
    "ruby", "next.js", "express", "spring boot", "firebase", "redis",
    "ci/cd", "jenkins", "terraform", "time series", "forecasting",
    "prophet", "computer vision", "data visualization", "etl",
]

EDUCATION_LEVELS = {
    "phd": 4, "doctorate": 4,
    "master": 3, "msc": 3, "m.tech": 3, "mba": 3,
    "bachelor": 2, "bsc": 2, "b.tech": 2, "be": 2, "undergraduate": 2,
    "diploma": 1, "associate": 1,
}


def extract_skills(text):
    """Return the list of known skills found in the text (case-insensitive)."""
    text_lower = text.lower()
    found = []
    for skill in SKILLS_VOCAB:
        # word-boundary match so "r" doesn't match inside "doctor", etc.
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found.append(skill)
    return found


def extract_years_of_experience(text):
    """
    Look for patterns like '4 years of experience', '5+ years', '3-5 years'.
    Returns the highest number found, or 0 if nothing matches.
    """
    text_lower = text.lower()
    patterns = [
        r"(\d+)\+?\s*years?\s+of\s+experience",
        r"(\d+)\+?\s*years?\s+experience",
        r"experience\s*:\s*(\d+)\+?\s*years?",
    ]
    years_found = []
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        years_found.extend(int(m) for m in matches)

    return max(years_found) if years_found else 0


def extract_email(text):
    match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else None


def extract_phone(text):
    match = re.search(r"(\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}", text)
    return match.group(0) if match else None


def extract_education_level(text):
    """Return the highest education level found, as a string, or None."""
    text_lower = text.lower()
    best_level = None
    best_rank = 0
    for keyword, rank in EDUCATION_LEVELS.items():
        if keyword in text_lower and rank > best_rank:
            best_rank = rank
            best_level = keyword
    return best_level


def extract_fields(text):
    """
    Main entry point. Takes raw resume or JD text and returns a dict
    of structured fields used downstream by features.py.
    """
    return {
        "skills": extract_skills(text),
        "experience_years": extract_years_of_experience(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "education_level": extract_education_level(text),
    }


if __name__ == "__main__":
    sample = """
    John Doe
    Email: john.doe@email.com | Phone: 555-123-4567
    Bachelor's degree in Computer Science.
    5+ years of experience in Python, SQL, and Machine Learning.
    Worked with Docker, AWS, and React in past roles.
    """
    print(extract_fields(sample))