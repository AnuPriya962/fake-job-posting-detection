from pathlib import Path
from typing import Union, List
import pandas as pd

# Schema Constants (Exported for team integration)
TARGET_COLUMN: str = "fraudulent"
ID_COLUMN: str = "job_id"

TEXT_COLUMNS: List[str] = [
    "title", "company_profile", "description", "requirements", "benefits"
]

CATEGORICAL_COLUMNS: List[str] = [
    "location", "department", "salary_range", "employment_type", 
    "required_experience", "required_education", "industry", "function"
]

BINARY_COLUMNS: List[str] = [
    "telecommuting", "has_company_logo", "has_questions"
]

EXPECTED_COLUMNS: List[str] = (
    [ID_COLUMN] + TEXT_COLUMNS + CATEGORICAL_COLUMNS + BINARY_COLUMNS + [TARGET_COLUMN]
)


def load_dataset(file_path: Union[str, Path]) -> pd.DataFrame:
    """Loads the CSV dataset from disk."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found at: {path.resolve()}")
    df = pd.read_csv(path)
    print(f"[INFO] Successfully loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Validates the presence of required schema columns."""
    df.columns = df.columns.str.strip()
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    
    if TARGET_COLUMN in missing_cols:
        raise ValueError(
            f"Critical schema failure: Target column '{TARGET_COLUMN}' is missing from the dataset!"
        )
        
    if missing_cols:
        print(f"[WARNING] Dataset is missing non-target expected columns: {missing_cols}")
        
    print("[INFO] Schema validation passed successfully.")


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Applies missing value imputation and type normalization."""
    df_clean = df.copy()
    
    # Text missing values -> empty string
    for col in TEXT_COLUMNS:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna("").astype(str)
            
    # Categorical missing values -> "Unspecified"
    for col in CATEGORICAL_COLUMNS:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna("Unspecified").astype(str)
            
    # Binary missing values -> 0
    for col in BINARY_COLUMNS:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(int)
            
    # Remove exact duplicate rows
    dedup_cols = [c for c in df_clean.columns if c != ID_COLUMN]
    initial_count = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=dedup_cols).reset_index(drop=True)
    removed_duplicates = initial_count - len(df_clean)
    
    if removed_duplicates > 0:
        print(f"[INFO] Removed {removed_duplicates} duplicate records.")
        
    return df_clean