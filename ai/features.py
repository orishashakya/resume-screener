"""
ai/features.py
Phase 3: Feature Engineering
Converts a (resume_text, job_description_text) pair into a numeric feature
vector that the classifier (Phase 4) is trained on.

Why features instead of raw text into the classifier?
With a small/synthetic training set, a model trained on a handful of
interpretable numeric signals (skill overlap %, experience gap, education
match, text similarity) generalizes far better and is far easier to explain
in a report than a model trained directly on raw text embeddings.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from extractor import extract_fields, EDUCATION_LEVELS


def skill_match_ratio(resume_skills, jd_skills):
    """
    Fraction of the JD's required skills that are present in the resume.
    Returns 0.0 if the JD lists no recognizable skills (avoids div-by-zero).
    """
    if not jd_skills:
        return 0.0
    resume_set = set(resume_skills)
    matched = [s for s in jd_skills if s in resume_set]
    return len(matched) / len(jd_skills)


def experience_match(resume_years, required_years):
    """
    1.0 if the candidate meets or exceeds the requirement.
    Partial credit if close; 0 if far below.
    """
    if required_years == 0:
        return 1.0  # no requirement stated, don't penalize
    if resume_years >= required_years:
        return 1.0
    gap = required_years - resume_years
    # partial credit: lose 0.25 per missing year, floor at 0
    return max(0.0, 1.0 - 0.25 * gap)


def education_match(resume_level, required_level):
    """
    1.0 if resume's education rank >= required rank (or no requirement).
    0.0 otherwise.
    """
    if not required_level:
        return 1.0
    resume_rank = EDUCATION_LEVELS.get(resume_level, 0) if resume_level else 0
    required_rank = EDUCATION_LEVELS.get(required_level, 0)
    return 1.0 if resume_rank >= required_rank else 0.0


def text_similarity(resume_text, jd_text):
    """
    TF-IDF cosine similarity between the full resume and JD text.
    Captures overlap beyond the fixed skills vocabulary (e.g. domain
    phrasing, soft skills, responsibilities).
    """
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(sim)
    except ValueError:
        # happens if both texts are empty or contain only stopwords
        return 0.0


def build_feature_vector(resume_text, jd_text, required_skills=None,
                          required_experience_years=0, required_education=None):
    """
    Main entry point. Takes raw resume text and JD text (plus optional
    explicit JD requirements - if not given, they're extracted from jd_text
    automatically) and returns:
      - a feature dict (numeric values used by the classifier)
      - an explanation dict (matched/missing skills, human-readable)
    """
    resume_fields = extract_fields(resume_text)

    # If the recruiter didn't supply structured requirements explicitly,
    # extract them from the JD text the same way we do for resumes.
    if required_skills is None:
        jd_fields = extract_fields(jd_text)
        required_skills = jd_fields["skills"]
        if required_experience_years == 0:
            required_experience_years = jd_fields["experience_years"]
        if required_education is None:
            required_education = jd_fields["education_level"]

    skill_ratio = skill_match_ratio(resume_fields["skills"], required_skills)
    exp_score = experience_match(resume_fields["experience_years"], required_experience_years)
    edu_score = education_match(resume_fields["education_level"], required_education)
    sim_score = text_similarity(resume_text, jd_text)

    features = {
        "skill_match_ratio": skill_ratio,
        "experience_match": exp_score,
        "education_match": edu_score,
        "text_similarity": sim_score,
    }

    matched_skills = [s for s in required_skills if s in resume_fields["skills"]]
    missing_skills = [s for s in required_skills if s not in resume_fields["skills"]]

    explanation = {
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "resume_experience_years": resume_fields["experience_years"],
        "required_experience_years": required_experience_years,
        "resume_education": resume_fields["education_level"],
        "required_education": required_education,
    }

    return features, explanation


if __name__ == "__main__":
    resume = """
    5+ years of experience in Python, SQL, and Machine Learning.
    Bachelor's degree in Computer Science. Worked with Docker, AWS, React.
    """
    jd = """
    Looking for a Python developer with Machine Learning and Docker experience.
    Requires 3+ years of experience and a Bachelor's degree.
    """
    features, explanation = build_feature_vector(resume, jd)
    print("Features:", features)
    print("Explanation:", explanation)