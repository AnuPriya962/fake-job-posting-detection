import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ROCCurveDisplay
)
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# Import Person 1 & 2 dependencies
from src.data_loader import load_dataset, validate_schema, clean_dataset, TARGET_COLUMN
from src.preprocessing import prepare_features, build_preprocessor

# Save artifacts path
MODEL_DIR = Path("../models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_and_split_data(dataset_path: str, test_size: float = 0.2, random_state: int = 42):
    """Loads, cleans, prepares, and splits the data into train and test sets."""
    raw_df = load_dataset(dataset_path)
    validate_schema(raw_df)
    df_clean = clean_dataset(raw_df)
    
    # Extract feature matrix and target series
    y = df_clean[TARGET_COLUMN].values
    df_features = prepare_features(df_clean)
    
    # Stratified Split to preserve severe class imbalance (95% Real vs 5% Fake)
    X_train, X_test, y_train, y_test = train_test_split(
        df_features, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    print(f"[INFO] Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"[INFO] Target class balance - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")
    return X_train, X_test, y_train, y_test


def build_model_pipelines():
    """
    Defines full pipelines combining Person 2's ColumnTransformer preprocessor 
    with each specified classification algorithm.
    """
    preprocessor = build_preprocessor()
    
    pipelines = {
        "Naive Bayes": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', MultinomialNB())
        ]),
        "Logistic Regression": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1))
        ]),
        "Support Vector Machine": Pipeline([
            ('preprocessor', preprocessor),
            # LinearSVC is wrapped in CalibratedClassifierCV to enable predict_proba for ROC-AUC & SHAP
            ('classifier', CalibratedClassifierCV(LinearSVC(class_weight='balanced', max_iter=2000, random_state=42)))
        ])
    }
    return pipelines


def get_hyperparameter_grids():
    """Defines search parameters tuned for high-dimensional TF-IDF feature space."""
    param_grids = {
        "Naive Bayes": {
            'classifier__alpha': [0.01, 0.1, 0.5, 1.0]
        },
        "Logistic Regression": {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__solver': ['lbfgs', 'liblinear']
        },
        "Random Forest": {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [20, 50, None],
            'classifier__min_samples_split': [2, 5]
        },
        "Support Vector Machine": {
            'classifier__estimator__C': [0.01, 0.1, 1.0, 10.0]
        }
    }
    return param_grids


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Executes GridSearchCV across all 4 candidate models, compares metrics on the 
    unseen test split, and exports performance plots.
    """
    pipelines = build_model_pipelines()
    param_grids = get_hyperparameter_grids()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    best_models = {}
    
    plt.figure(figsize=(10, 7))
    
    for name, pipeline in pipelines.items():
        print(f"\n[INFO] Starting Hyperparameter Tuning for: {name}...")
        
        # Optimize for PR-AUC or Macro F1 due to severe fraud class imbalance
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grids[name],
            cv=cv,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_models[name] = best_model
        
        # Predict on Test Set
        y_pred = best_model.predict(X_test)
        
        if hasattr(best_model, "predict_proba"):
            y_proba = best_model.predict_proba(X_test)[:, 1]
        else:
            y_proba = best_model.decision_function(X_test)

        # Collect Evaluation Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=1)
        rec = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        results.append({
            'Model': name,
            'Best Params': str(grid_search.best_params_),
            'Accuracy': round(acc, 4),
            'Precision (Fraud)': round(prec, 4),
            'Recall (Fraud)': round(rec, 4),
            'F1-Score (Fraud)': round(f1, 4),
            'ROC-AUC': round(roc_auc, 4)
        })
        
        # Plot ROC Curve
        ROCCurveDisplay.from_predictions(y_test, y_proba, name=name, ax=plt.gca())
        
        # Plot Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(MODEL_DIR / f"confusion_matrix_{name.replace(' ', '_').lower()}.png")
        plt.close()

    # Finalize combined ROC Curve plot
    plt.title('ROC-AUC Curve Comparison across Models')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.legend()
    plt.savefig(MODEL_DIR / "roc_curves_comparison.png")
    plt.show()

    # Display metric comparison table
    results_df = pd.DataFrame(results).sort_values(by='F1-Score (Fraud)', ascending=False)
    print("\n================ MODEL PERFORMANCE COMPARISON TABLE ================")
    print(results_df[['Model', 'Accuracy', 'Precision (Fraud)', 'Recall (Fraud)', 'F1-Score (Fraud)', 'ROC-AUC']].to_string(index=False))
    
    return results_df, best_models


def export_winning_artifacts(results_df, best_models, X_train, y_train):
    """Selects the top-performing model and exports it for Persons 4 & 5."""
    best_model_name = results_df.iloc[0]['Model']
    winning_model = best_models[best_model_name]
    
    print(f"\n[SUCCESS] Winning Model selected: '{best_model_name}'")
    
    # Save the pipeline containing preprocessor + best fitted model
    pipeline_export_path = MODEL_DIR / "best_model_pipeline.joblib"
    joblib.dump(winning_model, pipeline_export_path)
    
    # Save test/sample artifacts needed by Person 4 for SHAP analysis
    sample_data_path = MODEL_DIR / "sample_train_features.joblib"
    joblib.dump(X_train.sample(n=min(500, len(X_train)), random_state=42), sample_data_path)
    
    print(f"[INFO] Complete Pipeline exported to: {pipeline_export_path.resolve()}")
    print(f"[INFO] Training sample exported for Person 4 (SHAP) to: {sample_data_path.resolve()}")


if __name__ == "__main__":
    DATASET_PATH = "../data/fake_job_postings.csv"
    
    X_train, X_test, y_train, y_test = load_and_split_data(DATASET_PATH)
    results_df, best_models = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    export_winning_artifacts(results_df, best_models, X_train, y_train)
