import pandas as pd

def detect_dtype_issues(df: pd.DataFrame) -> dict:
    """
    Scans every column for likely dtype mismatches.
    Returns a dict of column -> suggested fix.
    """
    issues = {}

    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(100)

            # Check if it should be numeric
            numeric_converted = pd.to_numeric(sample, errors='coerce')
            if numeric_converted.notna().mean() > 0.8:
                issues[col] = "should_be_numeric"
                continue

            # Check if it should be datetime
            try:
                datetime_converted = pd.to_datetime(sample, errors='coerce')
                if datetime_converted.notna().mean() > 0.8:
                    issues[col] = "should_be_datetime"
                    continue
            except Exception:
                pass

            # Check if it should be categorical (low cardinality)
            if df[col].nunique() / len(df) < 0.05:
                issues[col] = "should_be_categorical"

    return issues


def fix_numeric_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Coerces a column to numeric. Non-convertible values become NaN.
    """
    df[column] = pd.to_numeric(df[column], errors='coerce')
    return df


def fix_datetime_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Coerces a column to datetime. Non-convertible values become NaT.
    """
    df[column] = pd.to_datetime(df[column], errors='coerce')
    return df


def fix_categorical_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Converts a low-cardinality string column to categorical dtype.
    """
    df[column] = df[column].astype('category')
    return df


def apply_dtype_fixes(df: pd.DataFrame, issues: dict) -> tuple[pd.DataFrame, list]:
    """
    Applies all detected dtype fixes.
    Returns the cleaned df and a log of what was changed.
    """
    log = []

    for col, issue in issues.items():
        if issue == "should_be_numeric":
            df = fix_numeric_column(df, col)
            log.append(f"'{col}': converted to numeric.")
        elif issue == "should_be_datetime":
            df = fix_datetime_column(df, col)
            log.append(f"'{col}': converted to datetime.")
        elif issue == "should_be_categorical":
            df = fix_categorical_column(df, col)
            log.append(f"'{col}': converted to categorical.")

    return df, log