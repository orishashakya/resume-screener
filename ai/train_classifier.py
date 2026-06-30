"""
ai/train_classifier.py
Phase 4: Train the viability classifier.

No public dataset of (resume, job_description, hired/not_hired) pairs exists,
so this uses "distant supervision": we pair each resume with several
synthetic job descriptions built from the Kaggle Resume Dataset's category
labels (e.g. INFORMATION-TECHNOLOGY, HR, CHEF). A resume paired with a JD
from its own category is labeled "Yes" (viable); paired with a JD from a
clearly different category, it's labeled "No". This is a standard weak-
supervision technique - documented explicitly here so it's transparent in
the final report rather than hidden.

Usage:
    python train_classifier.py
"""

import os
import random
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from features import build_feature_vector

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "Resume.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "classifier.pkl")

# Concrete JD templates per category - explicit required skills (matching our
# SKILLS_VOCAB), years of experience, and education level, so all 4 features
# carry real signal (a generic, vague JD makes skill/experience/education
# features collapse to constant values and become statistically useless).
CATEGORY_REQUIREMENTS = {
    "INFORMATION-TECHNOLOGY": {
        "skills": "python, sql, java, git, cloud computing and software development",
        "years": 3, "education": "bachelor",
    },
    "HR": {
        "skills": "communication, leadership, project management and HR policy",
        "years": 2, "education": "bachelor",
    },
    "CHEF": {
        "skills": "food preparation, menu development and kitchen management",
        "years": 3, "education": "diploma",
    },
    "SALES": {
        "skills": "communication, negotiation, client relationships and sales targets",
        "years": 2, "education": "bachelor",
    },
    "FINANCE": {
        "skills": "excel, financial analysis, statistics and accounting",
        "years": 3, "education": "bachelor",
    },
    "HEALTHCARE": {
        "skills": "patient care, clinical procedures and medical recordkeeping",
        "years": 2, "education": "bachelor",
    },
    "ENGINEERING": {
        "skills": "technical design, project management and statistics",
        "years": 3, "education": "bachelor",
    },
    "TEACHER": {
        "skills": "lesson planning, classroom management and communication",
        "years": 2, "education": "bachelor",
    },
    "ACCOUNTANT": {
        "skills": "excel, bookkeeping, financial statements and accounting",
        "years": 2, "education": "bachelor",
    },
    "BUSINESS-DEVELOPMENT": {
        "skills": "communication, leadership, project management and sales",
        "years": 3, "education": "bachelor",
    },
    "DESIGNER": {
        "skills": "data visualization, communication and creative design tools",
        "years": 2, "education": "bachelor",
    },
    "CONSULTANT": {
        "skills": "communication, project management, statistics and data analysis",
        "years": 3, "education": "bachelor",
    },
}

DEFAULT_REQUIREMENTS = {
    "skills": "communication, project management and relevant technical skills",
    "years": 2, "education": "bachelor",
}


def get_jd_text(category):
    req = CATEGORY_REQUIREMENTS.get(category, DEFAULT_REQUIREMENTS)
    return (
        f"We are hiring for a {category.replace('-', ' ').title()} role. "
        f"Required skills: {req['skills']}. "
        f"Requires {req['years']}+ years of experience and a {req['education']} degree."
    )


def generate_training_pairs(df, n_negative_per_resume=2):
    """
    For each resume:
      - 1 positive pair: resume + its own category's JD -> label 1 (Yes)
      - n negative pairs: resume + a different category's JD -> label 0 (No)
    Returns a list of dicts: {resume_text, jd_text, label}
    """
    categories = df["Category"].unique().tolist()
    pairs = []

    for _, row in df.iterrows():
        resume_text = row["Resume_str"]
        true_category = row["Category"]

        # Positive pair
        pairs.append({
            "resume_text": resume_text,
            "jd_text": get_jd_text(true_category),
            "label": 1,
        })

        # Negative pairs: sample categories different from the true one
        other_categories = [c for c in categories if c != true_category]
        n_sample = min(n_negative_per_resume, len(other_categories))
        sampled = random.sample(other_categories, n_sample)
        for neg_category in sampled:
            pairs.append({
                "resume_text": resume_text,
                "jd_text": get_jd_text(neg_category),
                "label": 0,
            })

    return pairs


def build_dataset(pairs):
    """Run every pair through features.py to get the numeric feature vectors."""
    rows = []
    for pair in pairs:
        features, _ = build_feature_vector(pair["resume_text"], pair["jd_text"])
        features["label"] = pair["label"]
        rows.append(features)
    return pd.DataFrame(rows)


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"\n--- {name}: Evaluation on held-out test set ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.3f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["No", "Yes"]))
    return f1_score(y_test, y_pred)


def train_and_evaluate(feature_df):
    X = feature_df[["skill_match_ratio", "experience_match", "education_match", "text_similarity"]]
    y = feature_df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # Model 1: Logistic Regression with stronger regularization (C=0.1) to
    # prevent coefficient blow-up from the near-separable text_similarity feature.
    lr_model = LogisticRegression(random_state=RANDOM_SEED, class_weight="balanced", C=0.1)
    lr_model.fit(X_train, y_train)
    lr_f1 = evaluate_model("Logistic Regression (C=0.1)", lr_model, X_test, y_test)

    print("\nLogistic Regression coefficients:")
    for name, coef in zip(X.columns, lr_model.coef_[0]):
        print(f"  {name}: {coef:.3f}")

    # Model 2: Random Forest - tree-based, doesn't suffer from the same
    # separation pathology, naturally captures feature interactions.
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=6, random_state=RANDOM_SEED, class_weight="balanced"
    )
    rf_model.fit(X_train, y_train)
    rf_f1 = evaluate_model("Random Forest", rf_model, X_test, y_test)

    print("\nRandom Forest feature importances:")
    for name, importance in zip(X.columns, rf_model.feature_importances_):
        print(f"  {name}: {importance:.3f}")

    # Pick whichever model scored higher F1 on "Yes" detection
    if rf_f1 >= lr_f1:
        print(f"\nSelected model: Random Forest (F1={rf_f1:.3f} vs Logistic Regression F1={lr_f1:.3f})")
        return rf_model
    else:
        print(f"\nSelected model: Logistic Regression (F1={lr_f1:.3f} vs Random Forest F1={rf_f1:.3f})")
        return lr_model


def main():
    print(f"Loading dataset from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} resumes across {df['Category'].nunique()} categories.")

    print("Generating synthetic training pairs (distant supervision)...")
    pairs = generate_training_pairs(df, n_negative_per_resume=2)
    print(f"Generated {len(pairs)} pairs "
          f"({sum(p['label'] for p in pairs)} positive, "
          f"{len(pairs) - sum(p['label'] for p in pairs)} negative).")

    print("Building feature vectors (this may take a minute for large datasets)...")
    feature_df = build_dataset(pairs)

    print("Training Logistic Regression classifier...")
    model = train_and_evaluate(feature_df)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()