"""
Prediction explainability module for Fake Job Posting Detection.
Supports model linear coefficient extraction, Random Forest feature importance,
feature recovery, and stable structured prediction output API.
"""

from typing import Dict, List, Any, Union, Optional
import numpy as np
import scipy.sparse as sp
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV


EXPLANATION_DISCLAIMER = (
    "This is an ML-based prediction, not proof that a job posting is fraudulent."
)


def get_feature_names_from_preprocessor(preprocessor: Any) -> List[str]:
    """
    Extracts transformed feature names safely from ColumnTransformer or Pipeline.
    Falls back to readable generic names if names cannot be automatically extracted.
    """
    if hasattr(preprocessor, "get_feature_names_out"):
        try:
            raw_names = preprocessor.get_feature_names_out()
            cleaned_names = []
            for name in raw_names:
                # Strip preprocessor prefixes (e.g., 'text_pipeline__tfidf__word')
                clean_name = name.split("__")[-1]
                cleaned_names.append(clean_name)
            return cleaned_names
        except Exception:
            pass

    # If preprocessor is a ColumnTransformer, inspect transformers individually
    if isinstance(preprocessor, ColumnTransformer):
        feature_names = []
        for name, transformer, cols in preprocessor.transformers_:
            if name == "remainder" and transformer == "drop":
                continue
            if hasattr(transformer, "get_feature_names_out"):
                try:
                    names = transformer.get_feature_names_out()
                    feature_names.extend([f.split("__")[-1] for f in names])
                    continue
                except Exception:
                    pass
            if isinstance(cols, list):
                feature_names.extend(cols)
        if feature_names:
            return feature_names

    return []


def _extract_linear_coefficients(model: Any) -> Optional[np.ndarray]:
    """Extracts 1D coefficient vector from various linear model structures."""
    if hasattr(model, "coef_"):
        coefs = model.coef_
        return coefs[0] if coefs.ndim > 1 else coefs

    # Support CalibratedClassifierCV (used by LinearSVC in Person 3)
    if isinstance(model, CalibratedClassifierCV):
        if hasattr(model, "calibrated_classifiers_") and len(model.calibrated_classifiers_) > 0:
            coef_list = []
            for clf in model.calibrated_classifiers_:
                base_estimator = getattr(clf, "estimator", getattr(clf, "base_estimator", None))
                if base_estimator is not None and hasattr(base_estimator, "coef_"):
                    c = base_estimator.coef_
                    coef_list.append(c[0] if c.ndim > 1 else c)
            if coef_list:
                return np.mean(coef_list, axis=0)

    if isinstance(model, Pipeline):
        final_step = model.steps[-1][1]
        return _extract_linear_coefficients(final_step)

    return None


def _explain_linear_model(
    model: Any,
    transformed_input: Any,
    feature_names: List[str],
    top_k: int = 10
) -> Dict[str, List[Dict[str, float]]]:
    """Computes signed feature contributions: contribution = x_i * w_i."""
    coefs = _extract_linear_coefficients(model)
    if coefs is None:
        return {"supporting_features": [], "opposing_features": []}

    if sp.issparse(transformed_input):
        x_dense = transformed_input.toarray().ravel()
    else:
        x_dense = np.asarray(transformed_input).ravel()

    n_features = min(len(x_dense), len(coefs))
    if len(feature_names) < n_features:
        feature_names = feature_names + [f"feature_{i}" for i in range(len(feature_names), n_features)]

    contributions = x_dense[:n_features] * coefs[:n_features]

    nonzero_idx = np.where(contributions != 0)[0]
    if len(nonzero_idx) == 0:
        return {"supporting_features": [], "opposing_features": []}

    active_indices = nonzero_idx[np.argsort(contributions[nonzero_idx])]
    
    # Positive contributions support Fraudulent class (1)
    supporting_idx = [i for i in reversed(active_indices) if contributions[i] > 0][:top_k]
    # Negative contributions oppose Fraudulent class (support Real class 0)
    opposing_idx = [i for i in active_indices if contributions[i] < 0][:top_k]

    supporting = [
        {"feature": str(feature_names[i]), "contribution": float(round(contributions[i], 4))}
        for i in supporting_idx
    ]
    opposing = [
        {"feature": str(feature_names[i]), "contribution": float(round(contributions[i], 4))}
        for i in opposing_idx
    ]

    return {
        "supporting_features": supporting,
        "opposing_features": opposing
    }


def _explain_tree_model(
    model: Any,
    transformed_input: Any,
    feature_names: List[str],
    top_k: int = 10
) -> Dict[str, List[Dict[str, float]]]:
    """Lightweight explanation strategy for Random Forest / Tree models."""
    estimator = model.steps[-1][1] if isinstance(model, Pipeline) else model

    if sp.issparse(transformed_input):
        x_dense = transformed_input.toarray().ravel()
    else:
        x_dense = np.asarray(transformed_input).ravel()

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
        n_features = min(len(x_dense), len(importances))

        if len(feature_names) < n_features:
            feature_names = feature_names + [f"feature_{i}" for i in range(len(feature_names), n_features)]

        local_impact = x_dense[:n_features] * importances[:n_features]
        active_idx = np.where(local_impact > 0)[0]

        if len(active_idx) > 0:
            top_active = active_idx[np.argsort(local_impact[active_idx])[::-1]][:top_k]
            supporting = [
                {
                    "feature": str(feature_names[i]),
                    "contribution": float(round(local_impact[i], 4))
                }
                for i in top_active
            ]
            return {"supporting_features": supporting, "opposing_features": []}

    return {"supporting_features": [], "opposing_features": []}


def explain_prediction(
    model: Any,
    transformed_input: Any,
    feature_names: Optional[List[str]] = None,
    predicted_class: Optional[int] = None,
    top_k: int = 10
) -> Dict[str, Any]:
    """
    Stable integration API for Person 5 to call predictions explainability.
    Returns JSON-serializable dictionary.
    """
    if feature_names is None:
        feature_names = []

    if sp.issparse(transformed_input):
        n_input_features = transformed_input.shape[1]
    else:
        n_input_features = np.asarray(transformed_input).shape[-1] if np.asarray(transformed_input).ndim > 0 else 0

    if len(feature_names) < n_input_features:
        feature_names = list(feature_names) + [
            f"feature_{i}" for i in range(len(feature_names), n_input_features)
        ]

    estimator = model.steps[-1][1] if isinstance(model, Pipeline) else model

    if _extract_linear_coefficients(estimator) is not None:
        explanation = _explain_linear_model(estimator, transformed_input, feature_names, top_k=top_k)
    elif hasattr(estimator, "feature_importances_"):
        explanation = _explain_tree_model(estimator, transformed_input, feature_names, top_k=top_k)
    else:
        explanation = {"supporting_features": [], "opposing_features": []}

    explanation["disclaimer"] = EXPLANATION_DISCLAIMER
    return explanation


def get_confidence_and_risk_score(
    model: Any,
    transformed_input: Any
) -> Dict[str, Any]:
    """
    Computes confidence score and user-facing risk labels without presenting raw decision scores as fake probabilities.
    """
    estimator = model.steps[-1][1] if isinstance(model, Pipeline) else model

    if hasattr(estimator, "predict_proba"):
        probas = estimator.predict_proba(transformed_input)[0]
        fraud_risk = float(probas[1])
        score_type = "probability"
    elif hasattr(estimator, "decision_function"):
        dec_score = float(estimator.decision_function(transformed_input)[0])
        fraud_risk = float(1.0 / (1.0 + np.exp(-dec_score)))
        score_type = "decision_score"
    else:
        fraud_risk = 0.5
        score_type = "unknown"

    pred_class = 1 if fraud_risk >= 0.5 else 0

    return {
        "predicted_class": pred_class,
        "predicted_label": "Fake / Fraudulent" if pred_class == 1 else "Real / Legitimate",
        "fraud_risk_score": round(fraud_risk, 4),
        "score_type": score_type,
        "disclaimer": EXPLANATION_DISCLAIMER
    }
