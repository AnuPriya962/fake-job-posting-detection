import pytest
import pandas as pd
from src.data_loader import (
    validate_schema,
    clean_dataset,
    TARGET_COLUMN
)

@pytest.fixture
def sample_valid_df():
    """Generates synthetic DataFrame for testing without needing the full CSV."""
    return pd.DataFrame({
        "job_id": [1, 2],
        "title": ["Software Engineer", None],
        "location": ["US, NY", None],
        "department": ["Engineering", "Data"],
        "salary_range": ["100k-120k", None],
        "company_profile": ["Good Co", None],
        "description": ["Code stuff", "Analyze stuff"],
        "requirements": ["B.Tech", None],
        "benefits": ["Health insurance", None],
        "telecommuting": [1, None],
        "has_company_logo": [1, 0],
        "has_questions": [0, 1],
        "employment_type": ["Full-time", None],
        "required_experience": ["Mid-Senior", None],
        "required_education": ["Bachelor's", "High School"],
        "industry": ["IT", "Services"],
        "function": ["Engineering", "Support"],
        "fraudulent": [0, 1]
    })


def test_schema_validation_success(sample_valid_df):
    """Tests that valid schema passes validation without error."""
    validate_schema(sample_valid_df)


def test_schema_validation_missing_target(sample_valid_df):
    """Tests that missing fraudulent column raises ValueError."""
    df_no_target = sample_valid_df.drop(columns=[TARGET_COLUMN])
    with pytest.raises(ValueError, match="Target column 'fraudulent' is missing"):
        validate_schema(df_no_target)


def test_clean_dataset_imputation(sample_valid_df):
    """Tests that missing values are correctly imputed."""
    df_cleaned = clean_dataset(sample_valid_df)
    assert df_cleaned.loc[1, "title"] == ""
    assert df_cleaned.loc[1, "location"] == "Unspecified"
    assert df_cleaned.loc[1, "telecommuting"] == 0