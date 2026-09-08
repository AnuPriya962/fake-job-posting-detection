"""
Data loader package for Fake Job Posting Detection.
"""

from .data_loader import (
    load_dataset,
    validate_schema,
    clean_dataset,
    TARGET_COLUMN,
    ID_COLUMN,
    TEXT_COLUMNS,
    CATEGORICAL_COLUMNS,
    BINARY_COLUMNS,
)

__all__ = [
    "load_dataset",
    "validate_schema",
    "clean_dataset",
    "TARGET_COLUMN",
    "ID_COLUMN",
    "TEXT_COLUMNS",
    "CATEGORICAL_COLUMNS",
    "BINARY_COLUMNS",
]