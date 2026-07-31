"""Small helpers shared by more than one pipeline stage."""

import pandas as pd


def chronological_split_by_days(df: pd.DataFrame, date_col: str, test_days: int):
    """Split a time-ordered dataframe so the LAST `test_days` rows are the
    test set. Every model must be evaluated on the same held-out days for
    the comparison in evaluation.py to be a fair, apples-to-apples table."""
    df = df.sort_values(date_col).reset_index(drop=True)
    split_idx = len(df) - test_days
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]
    return train, test
