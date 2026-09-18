"""
Evaluation and visualization module for fake job posting detection models.
Provides evaluation metric computation and non-hardcoded visualization plotting.
"""

from typing import Dict, Any, List, Union, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

# Label definitions according to shared team contract
LABEL_MAP = {0: "Real / Legitimate", 1: "Fake / Fraudulent"}
CLASS_NAMES = ["Real / Legitimate", "Fake / Fraudulent"]


def evaluate_model_performance(
    y_true: Union[np.ndarray, pd.Series, List[int]],
    y_pred: Union[np.ndarray, pd.Series, List[int]],
    y_scores: Union[np.ndarray, pd.Series, List[float]],
) -> Dict[str, Any]:
    """
    Computes standard academic classification metrics for binary fake job detection.
    
    Target label convention:
        0 -> Real / Legitimate
        1 -> Fake / Fraudulent
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_scores = np.asarray(y_scores)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_scores))
    except ValueError:
        roc_auc = 0.0

    try:
        pr_auc = float(average_precision_score(y_true, y_scores))
    except ValueError:
        pr_auc = 0.0

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": cm,
        "class_labels": {0: "Real / Legitimate", 1: "Fake / Fraudulent"},
    }


def plot_class_distribution(
    y: Union[np.ndarray, pd.Series, List[int]],
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """Plots class frequency distribution for target variable."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    y_arr = np.asarray(y)
    counts = pd.Series(y_arr).value_counts().rename(index=LABEL_MAP)
    
    sns.barplot(x=counts.index, y=counts.values, palette=["#2b5c8f", "#d9534f"], ax=ax)
    ax.set_title("Class Distribution: Real vs Fraudulent Job Postings")
    ax.set_ylabel("Count")
    ax.set_xlabel("Class")

    total = len(y_arr)
    for p in ax.patches:
        height = p.get_height()
        pct = (height / total) * 100 if total > 0 else 0
        ax.annotate(
            f"{int(height)}\n({pct:.2f}%)",
            (p.get_x() + p.get_width() / 2.0, height / 2.0),
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )
    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    cm: Union[List[List[int]], np.ndarray],
    model_name: str = "Model",
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """Plots confusion matrix given raw count matrix."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.get_figure()

    cm_arr = np.asarray(cm)
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
        cbar=False
    )
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_ylabel("Actual Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    return fig


def plot_roc_curves(
    model_scores: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plots ROC curves for multiple models.
    model_scores: Dict mapping model_name -> (y_true, y_scores)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    for name, (y_true, y_scores) in model_scores.items():
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_val = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", label="Random Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves")
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


def plot_precision_recall_curves(
    model_scores: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plots Precision-Recall curves for multiple models.
    model_scores: Dict mapping model_name -> (y_true, y_scores)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    for name, (y_true, y_scores) in model_scores.items():
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = average_precision_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0
        ax.plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc:.3f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall (PR) Curves")
    ax.legend(loc="lower left")
    plt.tight_layout()
    return fig


def plot_model_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    metric_name: str = "f1_score",
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plots comparative bar chart across models for a specific metric.
    metrics_dict: Dict mapping model_name -> evaluation_metrics_dict
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    models = list(metrics_dict.keys())
    values = [metrics_dict[m].get(metric_name, 0.0) for m in models]

    df_plot = pd.DataFrame({"Model": models, metric_name: values})
    sns.barplot(data=df_plot, x="Model", y=metric_name, palette="Blues_d", ax=ax)

    ax.set_title(f"Model Comparison by {metric_name.replace('_', ' ').title()}")
    ax.set_ylim([0, 1.05])
    
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{height:.3f}",
            (p.get_x() + p.get_width() / 2.0, height + 0.02),
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    return fig
