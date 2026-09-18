"""
Prediction pipeline module for Fake Job Posting Detection.
Transforms raw input data, executes prediction, and generates explainability details.
"""

from typing import Dict, Any, Union
import pandas as pd
import numpy as np

from src.explain import explain_prediction, get_confidence_and_risk_score

EXPECTED_COLUMNS = [
    "title", "location", "department", "salary_range", "company_profile",
    "description", "requirements", "benefits", "telecommuting",
    "has_company_logo", "has_questions", "employment_type",
    "required_experience", "required_education", "industry", "function"
]


def _format_raw_input(job_data: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
    """Formats single-dictionary or partial DataFrame into canonical schema."""
    if isinstance(job_data, dict):
        df = pd.DataFrame([job_data])
    else:
        df = job_data.copy()

    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            if col in ["telecommuting", "has_company_logo", "has_questions"]:
                df[col] = 0
            else:
                df[col] = ""

    binary_cols = ["telecommuting", "has_company_logo", "has_questions"]
    for col in binary_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    text_cols = [c for c in EXPECTED_COLUMNS if c not in binary_cols]
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str)

    return df[EXPECTED_COLUMNS]


def predict_job(
    model: Any,
    preprocessor: Any,
    job_data: Union[Dict[str, Any], pd.DataFrame],
    feature_names: Any = None
) -> Dict[str, Any]:
    """
    Main prediction entry point for Person 5 Streamlit application.
    """
    df = _format_raw_input(job_data)

    transformed_input = preprocessor.transform(df)

    score_info = get_confidence_and_risk_score(model, transformed_input)
    pred_class = score_info["predicted_class"]

    if feature_names is None or len(feature_names) == 0:
        if hasattr(preprocessor, "get_feature_names_out"):
            try:
                feature_names = preprocessor.get_feature_names_out()
            except Exception:
                feature_names = []
        else:
            feature_names = []

    explanation = explain_prediction(
        model=model,
        transformed_input=transformed_input,
        feature_names=feature_names,
        predicted_class=pred_class
    )

    label_str = "Fake" if pred_class == 1 else "Real"

    return {
        "label": label_str,
        "class_id": pred_class,
        "score": score_info["fraud_risk_score"],
        "score_type": score_info["score_type"],
        "explanation": explanation
    }
