"""Validation for the standardized response table."""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("study", "cell_id", "drug_id", "sensitivity")
KEY_COLUMNS = ("study", "cell_id", "drug_id")


def validate_responses(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized copy or raise on an invalid response table."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required response columns: {missing}")

    result = frame.copy()
    for column in ("study", "cell_id", "drug_id"):
        if result[column].isna().any():
            raise ValueError(f"{column} contains missing values")
        result[column] = result[column].astype(str).str.strip()
        if (result[column] == "").any():
            raise ValueError(f"{column} contains empty identifiers")

    result["sensitivity"] = pd.to_numeric(result["sensitivity"], errors="raise")
    if not np.isfinite(result["sensitivity"].to_numpy()).all():
        raise ValueError("sensitivity must contain only finite values")
    if result.duplicated(list(KEY_COLUMNS)).any():
        duplicate_count = int(result.duplicated(list(KEY_COLUMNS), keep=False).sum())
        raise ValueError(f"Response keys are not unique ({duplicate_count} rows)")

    return result.reset_index(drop=True)
