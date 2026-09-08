
# Fake Job Posting Detection

A machine learning and natural language processing system designed to analyze job postings—combining unstructured text and structured metadata—to classify them as **Real** or **Fake**. Built with custom data validation, automated model selection across four classical algorithms, feature-level explainability, and an interactive Streamlit web application.

---

## Project Architecture & Methodology

```text
Raw Dataset (CSV)
      │
      ▼
1. Data Validation & Cleaning (src/data_loader.py)
      │
      ▼
2. Preprocessing & Feature Engineering (src/preprocessing.py)
   ├── Text Fields  ──► TF-IDF Vectorization (1-2 N-Grams)
   ├── Categorical  ──► One-Hot Encoding (Unknown handling)
   └── Binary/Num   ──► Numeric Normalization
      │
      ▼
3. Model Training & Comparison (src/train.py, src/evaluate.py)
   ├── Naive Bayes (MultinomialNB)
   ├── Logistic Regression (Balanced Weights)
   ├── Random Forest (Dimensionality Reduction via TruncatedSVD)
   └── Support Vector Machine (LinearSVC with Calibration)
      │
      ▼
4. Selection & Artifact Saving (models/)
   ├── Selection Rule: Fraudulent-Class F1 / Recall / PR-AUC
   └── Saved Output: Model, Preprocessor, Metadata, Metrics JSON
      │
      ▼
5. Interactive Web Application (app.py)
   ├── Full Job Text Parsing
   ├── Structured Form Submission
   └── Prediction Risk Score & Keyword Explanations (src/explain.py)

## Directory Structure
fake-job-posting-detection/
├── app.py                      # Interactive Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .gitignore                  # Git untracked pattern rules
├── data/
│   └── fake_job_postings.csv   # Dataset path
├── models/                     # Saved training artifacts & results
│   ├── best_model.joblib
│   ├── preprocessing.joblib
│   ├── model_metadata.joblib
│   └── evaluation_results.json
├── src/                        # Modular source code
│   ├── __init__.py
│   ├── data_loader.py          # Data ingestion & schema validation
│   ├── preprocessing.py       # TF-IDF & structured preprocessing
│   ├── train.py                # Model training pipeline
│   ├── evaluate.py             # Evaluation metrics & reports
│   ├── explain.py              # Prediction feature attribution
│   └── predict.py              # Single-instance prediction interface
├── notebooks/
│   └── exploration.ipynb       # Exploratory Data Analysis (EDA) notebook
└── tests/                      # Unit tests
    └── test_data.py
