import pandas as pd
import numpy as np
from config import (
    OUTLIER_IQR_MULTIPLIER,
    OUTLIER_ZSCORE_THRESHOLD,
    OUTLIER_DROP_MAX_PCT,
    OUTLIER_DROP_MIN_ROWS,
)

def detect_outliers_iqr(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Returns a boolean mask — True where a value is an outlier.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - OUTLIER_IQR_MULTIPLIER * IQR
    upper = Q3 + OUTLIER_IQR_MULTIPLIER * IQR
    return (df[column] < lower) | (df[column] > upper)


def detect_outliers_zscore(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Returns a boolean mask using Z-score method.
    """
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    return z_scores > OUTLIER_ZSCORE_THRESHOLD


def cap_outliers_iqr(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, int]:
    """
    Caps outliers at the IQR boundaries (Winsorization).
    Less aggressive than dropping — preferred for small datasets.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outlier_count = int(detect_outliers_iqr(df, column).sum())
    df[column] = df[column].clip(lower=lower, upper=upper)
    return df, outlier_count


def drop_outliers_iqr(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, int]:
    """
    Drops rows where the column value is an outlier.
    More aggressive — can lead to data loss, so use with caution.
    """
    mask = detect_outliers_iqr(df, column)
    outlier_count = int(mask.sum())
    df = df[~mask].reset_index(drop=True)
    return df, outlier_count


def apply_outlier_fixes(df: pd.DataFrame, profile: dict, strategy: str = "cap") -> tuple[pd.DataFrame, list]:
    """
    Applies outlier handling to all numeric columns with detected outliers.
    Strategy: 'cap' (Winsorize) or 'drop'.
    Cap is the default — safer for small datasets.
    """
    log = []

    for col, col_profile in profile["columns"].items():
        if col_profile.get("type_category") != "numeric":
            continue
        if col_profile.get("outlier_count", 0) == 0:
            continue
        if col not in df.columns:
            continue

        if strategy == "cap":
            df, count = cap_outliers_iqr(df, col)
            log.append(f"'{col}': {count} outliers capped using IQR boundaries.")
        elif strategy == "drop":
            df, count = drop_outliers_iqr(df, col)
            log.append(f"'{col}': {count} outlier rows dropped.")

    return df, log