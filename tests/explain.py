"""
Unit tests for Person 4 Evaluation & Explainability modules.
"""

import json
import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.evaluate import evaluate_model_performance
from src.explain import (
    explain_prediction,
    get_confidence_and_risk_score,
    EXPLANATION_DISCLAIMER
)


@pytest.fixture
def synthetic_linear_data():
    X = np.array([[1.0, 0.0, 0.5], [0.0, 2.0, 0.0], [1.5, 0.1, 0.0], [0.0, 0.0, 1.0]])
    y = np.array([0, 1, 0, 1])
    feature_names = ["urgent_wire", "reputable_company", "work_from_home"]
    model = LogisticRegression().fit(X, y)
    return model, X, y, feature_names


@pytest.fixture
def synthetic_tree_data():
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    y = np.array([1, 0, 1, 0])
    feature_names = ["wire_transfer", "office_location"]
    model = RandomForestClassifier(n_estimators=5, random_state=42).fit(X, y)
    return model, X, y, feature_names


def test_evaluate_model_performance_structure():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 0]
    y_scores = [0.1, 0.8, 0.2, 0.4]

    metrics = evaluate_model_performance(y_true, y_pred, y_scores)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["class_labels"][0] == "Real / Legitimate"
    assert metrics["class_labels"][1] == "Fake / Fraudulent"


def test_explain_linear_model(synthetic_linear_data):
    model, X, _, feature_names = synthetic_linear_data
    sample_input = X[0:1]

    exp = explain_prediction(
        model=model,
        transformed_input=sample_input,
        feature_names=feature_names,
        predicted_class=0
    )

    assert "supporting_features" in exp
    assert "opposing_features" in exp
    assert "disclaimer" in exp
    assert exp["disclaimer"] == EXPLANATION_DISCLAIMER

    # JSON Serializability Check
    json_str = json.dumps(exp)
    assert isinstance(json_str, str)


def test_explain_tree_model(synthetic_tree_data):
    model, X, _, feature_names = synthetic_tree_data
    sample_input = X[0:1]

    exp = explain_prediction(
        model=model,
        transformed_input=sample_input,
        feature_names=feature_names,
        predicted_class=1
    )

    assert "supporting_features" in exp
    assert isinstance(exp["supporting_features"], list)


def test_explain_empty_input(synthetic_linear_data):
    model, _, _, feature_names = synthetic_linear_data
    empty_input = np.zeros((1, len(feature_names)))

    exp = explain_prediction(
        model=model,
        transformed_input=empty_input,
        feature_names=feature_names
    )

    assert exp["supporting_features"] == []
    assert exp["opposing_features"] == []


def test_no_hardcoded_keywords(synthetic_linear_data):
    model, X, _, _ = synthetic_linear_data
    exp = explain_prediction(
        model=model,
        transformed_input=X[0:1],
        feature_names=[]
    )

    for item in exp["supporting_features"] + exp["opposing_features"]:
        assert item["feature"].startswith("feature_")


def test_confidence_and_risk_score(synthetic_linear_data):
    model, X, _, _ = synthetic_linear_data
    score_info = get_confidence_and_risk_score(model, X[0:1])

    assert "fraud_risk_score" in score_info
    assert "predicted_label" in score_info
    assert score_info["predicted_label"] in ["Real / Legitimate", "Fake / Fraudulent"]
    assert 0.0 <= score_info["fraud_risk_score"] <= 1.0
