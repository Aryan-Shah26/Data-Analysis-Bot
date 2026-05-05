import pandas as pd
from config import NULL_DROP_COLUMN_THRESHOLD

def get_null_strategy(col_profile: dict) -> str:
    """
    Decides the best null-filling strategy based on column statistics.
    This is the logic that stops the LLM from blindly using mean everywhere.
    """
    # Drop the column if more than 60% is missing
    if col_profile["null_percent"] > NULL_DROP_COLUMN_THRESHOLD * 100 :
        return "drop_column"

    type_cat = col_profile.get("type_category")

    if type_cat == "numeric":
        # Use median for skewed data, mean for symmetric
        skew_label = col_profile.get("skew_label", "symmetric")
        if skew_label in ("moderate_skew", "high_skew"):
            return "fill_median"
        return "fill_mean"

    elif type_cat == "categorical":
        return "fill_mode"

    elif type_cat == "datetime":
        return "fill_forward"

    return "drop_rows"


def fill_with_mean(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df[column] = df[column].fillna(df[column].mean())
    return df


def fill_with_median(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df[column] = df[column].fillna(df[column].median())
    return df


def fill_with_mode(df: pd.DataFrame, column: str) -> pd.DataFrame:
    mode_val = df[column].mode()
    if not mode_val.empty:
        df[column] = df[column].fillna(mode_val[0])
    return df


def fill_forward(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df[column] = df[column].ffill()
    return df


def drop_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return df.drop(columns=[column])


def drop_rows_with_nulls(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return df.dropna(subset=[column]).reset_index(drop=True)


def apply_null_fixes(df: pd.DataFrame, profile: dict) -> tuple[pd.DataFrame, list]:
    """
    Applies null handling to every column using the strategy
    determined by get_null_strategy(). Returns cleaned df and log.
    """
    log = []

    for col, col_profile in profile["columns"].items():
        if col_profile["null_count"] == 0:
            continue

        if col not in df.columns:
            continue

        strategy = get_null_strategy(col_profile)

        if strategy == "drop_column":
            df = drop_column(df, col)
            log.append(f"'{col}': dropped (>{60}% missing).")
        elif strategy == "fill_mean":
            df = fill_with_mean(df, col)
            log.append(f"'{col}': nulls filled with mean ({col_profile['mean']}).")
        elif strategy == "fill_median":
            df = fill_with_median(df, col)
            log.append(f"'{col}': nulls filled with median ({col_profile['median']}) — high skew detected.")
        elif strategy == "fill_mode":
            df = fill_with_mode(df, col)
            log.append(f"'{col}': nulls filled with mode.")
        elif strategy == "fill_forward":
            df = fill_forward(df, col)
            log.append(f"'{col}': nulls forward-filled (datetime column).")
        elif strategy == "drop_rows":
            before = len(df)
            df = drop_rows_with_nulls(df, col)
            log.append(f"'{col}': {before - len(df)} rows dropped due to nulls.")

    return df, log