"""
ai/matcher.py
Phase 5: The Matcher
Ties parser -> features -> trained classifier into one function.
This is what the backend calls when a job seeker uploads a resume.
"""

import os
import joblib
import pandas as pd

from parser import parse_resume
from features import build_feature_vector

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "classifier.pkl")

_model = None  # loaded once, reused across calls (avoids reloading from disk every request)


def load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}. "
                f"Run train_classifier.py first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def score_candidate(resume_text, jd_text):
    """
    Main entry point used by the backend.
    Takes raw resume text and JD text, returns a result dict:
      - prediction: "Yes" or "No"
      - confidence: float 0-1 (model's probability for the predicted class)
      - match_score: float 0-1 (probability of "Yes", useful for ranking candidates)
      - explanation: dict with matched/missing skills, experience/education comparison
    """
    model = load_model()

    features, explanation = build_feature_vector(resume_text, jd_text)

    feature_order = ["skill_match_ratio", "experience_match", "education_match", "text_similarity"]
    feature_vector = pd.DataFrame([[features[f] for f in feature_order]], columns=feature_order)

    prediction = model.predict(feature_vector)[0]
    probabilities = model.predict_proba(feature_vector)[0]  # [P(No), P(Yes)]

    prob_yes = probabilities[1]
    confidence = max(probabilities)

    return {
        "prediction": "Yes" if prediction == 1 else "No",
        "confidence": round(float(confidence), 3),
        "match_score": round(float(prob_yes), 3),
        "features": features,
        "explanation": explanation,
    }


def score_candidate_from_file(resume_filepath, jd_text):
    """Convenience wrapper: parses a resume file path directly, then scores it."""
    resume_text = parse_resume(resume_filepath)
    return score_candidate(resume_text, jd_text)


if __name__ == "__main__":
    sample_resume = """
    5+ years of experience in Python, SQL, and Machine Learning.
    Bachelor's degree in Computer Science. Worked with Docker, AWS, React.
    """
    sample_jd = """
    Looking for a Python developer with Machine Learning and Docker experience.
    Requires 3+ years of experience and a Bachelor's degree.
    """
    result = score_candidate(sample_resume, sample_jd)
    print(result)