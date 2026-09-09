import re
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer

def clean_text(text: str) -> str:
    """Cleans individual text strings safely while preserving fraud-related phrasing."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text) # Remove HTML tags
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def combine_text_fields(df: pd.DataFrame) -> pd.Series:
    """Combines key text columns into a single descriptive text field per row."""
    text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    processed_df = df.copy()
    
    for col in text_cols:
        if col not in processed_df.columns:
            processed_df[col] = ""
        else:
            processed_df[col] = processed_df[col].fillna("").astype(str)
            
    combined = (
        "TITLE: " + processed_df['title'] + " \n " +
        "COMPANY_PROFILE: " + processed_df['company_profile'] + " \n " +
        "DESCRIPTION: " + processed_df['description'] + " \n " +
        "REQUIREMENTS: " + processed_df['requirements'] + " \n " +
        "BENEFITS: " + processed_df['benefits']
    )
    return combined.apply(clean_text)

def build_preprocessor() -> ColumnTransformer:
    """
    Builds and returns the scikit-learn ColumnTransformer handling text (TF-IDF),
    categorical columns (OneHotEncoder with sparse output), and binary/numeric flags.
    """
    categorical_cols = [
        'location', 'department', 'salary_range', 
        'employment_type', 'required_experience', 
        'required_education', 'industry', 'function'
    ]
    binary_cols = ['telecommuting', 'has_company_logo', 'has_questions']
    
    # Text pipeline using TfidfVectorizer
    text_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        ))
    ])
    
    # Categorical pipeline: sparse_output=True keeps the feature matrix sparse and memory-efficient
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
    ])
    
    # Binary/numeric pipeline
    binary_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_pipeline, 'clean_combined_text'),
            ('cat', categorical_pipeline, categorical_cols),
            ('bin', binary_pipeline, binary_cols)
        ],
        remainder='drop'
    )
    
    return preprocessor

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares raw data frame by appending the engineered combined text column 
    and dropping restricted attributes like 'job_id' to prevent data leakage.
    """
    df_prep = df.copy()
    
    # Ensure job_id is never used as a predictive feature
    if 'job_id' in df_prep.columns:
        df_prep = df_prep.drop(columns=['job_id'])
        
    # Generate combined text feature column
    df_prep['clean_combined_text'] = combine_text_fields(df_prep)
    
    return df_prep
