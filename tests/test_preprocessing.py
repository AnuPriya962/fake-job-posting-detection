# tests/test_preprocessing.py
import pytest
import pandas as pd
import numpy as np
from src.preprocessing import clean_text, combine_text_fields, build_preprocessor, prepare_features

def test_clean_text():
    assert clean_text("  Hello   World! <br> ") == "hello world!"
    assert clean_text(None) == ""

def test_combine_text_fields():
    df = pd.DataFrame([{
        "title": "Software Engineer",
        "company_profile": "Tech Corp",
        "description": "Coding python",
        "requirements": "BS degree",
        "benefits": "Health insurance"
    }])
    combined = combine_text_fields(df)
    assert "TITLE: software engineer" in combined.iloc[0]
    assert "COMPANY_PROFILE: tech corp" in combined.iloc[0]

def test_job_id_exclusion():
    df = pd.DataFrame([{
        "job_id": 999,
        "title": "Developer",
        "location": "NY"
    }])
    prepared = prepare_features(df)
    assert "job_id" not in prepared.columns
    assert "clean_combined_text" in prepared.columns

def test_preprocessor_fit_transform():
    df = pd.DataFrame([
        {
            "job_id": 1,
            "title": "Data Analyst",
            "location": "US, NY, New York",
            "department": "Analytics",
            "salary_range": "",
            "company_profile": "Good company",
            "description": "Analyze data using python and SQL",
            "requirements": "SQL skills",
            "benefits": "PTO",
            "telecommuting": 0,
            "has_company_logo": 1,
            "has_questions": 0,
            "employment_type": "Full-time",
            "required_experience": "Mid-Senior level",
            "required_education": "Bachelor's Degree",
            "industry": "Computer Software",
            "function": "Analyst"
        }
    ])
    
    prepared = prepare_features(df)
    preprocessor = build_preprocessor()
    
    # Fit and transform on training data
    transformed = preprocessor.fit_transform(prepared)
    assert transformed.shape[0] == 1
    
    # Test single-row prediction compatibility with unknown category
    new_job = pd.DataFrame([
        {
            "title": "New Role",
            "location": "Unknown City", # Unknown category test
            "department": "Unknown Dept",
            "salary_range": "",
            "company_profile": "",
            "description": "New description",
            "requirements": "",
            "benefits": "",
            "telecommuting": 1,
            "has_company_logo": 0,
            "has_questions": 1,
            "employment_type": "Part-time",
            "required_experience": "Entry level",
            "required_education": "High School",
            "industry": "Finance",
            "function": "Other"
        }
    ])
    new_prepared = prepare_features(new_job)
    new_transformed = preprocessor.transform(new_prepared)
    assert new_transformed.shape[0] == 1
