import pandas as pd
import numpy as np
from scipy import stats
from config import (
    OUTLIER_IQR_MULTIPLIER,
    SAMPLE_SIZE,
    NUMERIC_INFERENCE_THRESHOLD,
    DATETIME_INFERENCE_THRESHOLD,
    CATEGORICAL_CARDINALITY_RATIO,
    HIGH_CARDINALITY_THRESHOLD,
    TOP_VALUES_COUNT,
    SKEW_SYMMETRIC_MAX,
    SKEW_MODERATE_MAX,
    CHUNK_SIZE_ROWS
)


def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Full dataset profile. This is the first tool the agent calls.
    Returns everything the LLM needs to make cleaning decisions.
    """
    profile = {
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": {}
    }

    for col in df.columns:
        col_profile = {}
        col_profile["dtype"] = str(df[col].dtype)
        col_profile["null_count"] = int(df[col].isnull().sum())
        col_profile["null_percent"] = round(df[col].isnull().mean() * 100, 2)
        col_profile["unique_count"] = int(df[col].nunique())
        col_profile["sample_values"] = df[col].dropna().head(5).tolist()

        # --- Numeric ---
        if pd.api.types.is_numeric_dtype(df[col]):
            col_profile["type_category"] = "numeric"
            col_profile["mean"] = round(float(df[col].mean()), 4)
            col_profile["median"] = round(float(df[col].median()), 4)
            col_profile["std"] = round(float(df[col].std()), 4)
            col_profile["min"] = round(float(df[col].min()), 4)
            col_profile["max"] = round(float(df[col].max()), 4)
            col_profile["skewness"] = round(float(df[col].skew()), 4)

            skew = abs(col_profile["skewness"])
            if skew < SKEW_SYMMETRIC_MAX:
                col_profile["skew_label"] = "symmetric"
            elif skew < SKEW_MODERATE_MAX:
                col_profile["skew_label"] = "moderate_skew"
            else:
                col_profile["skew_label"] = "high_skew"

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = (df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)
            col_profile["outlier_count"] = int(outlier_mask.sum())

        # --- Datetime ---
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_profile["type_category"] = "datetime"
            col_profile["min_date"] = str(df[col].min())
            col_profile["max_date"] = str(df[col].max())

        # --- Object: try to infer deeper type ---
        elif df[col].dtype == object:
            sample = df[col].dropna().head(SAMPLE_SIZE)

            # Check if it should be numeric
            numeric_converted = pd.to_numeric(sample, errors='coerce')
            if numeric_converted.notna().mean() > NUMERIC_INFERENCE_THRESHOLD:
                col_profile["type_category"] = "should_be_numeric"
                # Compute stats on converted values for better LLM context
                converted_full = pd.to_numeric(df[col], errors='coerce')
                col_profile["mean"] = round(float(converted_full.mean()), 4)
                col_profile["median"] = round(float(converted_full.median()), 4)
                skew = abs(float(converted_full.skew()))
                col_profile["skewness"] = round(float(converted_full.skew()), 4)
                col_profile["skew_label"] = (
                    "symmetric" if skew < 0.5
                    else "moderate_skew" if skew < 1.0
                    else "high_skew"
                )
                Q1 = converted_full.quantile(0.25)
                Q3 = converted_full.quantile(0.75)
                IQR = Q3 - Q1
                outlier_mask = (converted_full < Q1 - 1.5 * IQR) | (converted_full > Q3 + 1.5 * IQR)
                col_profile["outlier_count"] = int(outlier_mask.sum())

            else:
                # Check if it should be datetime
                try:
                    datetime_converted = pd.to_datetime(sample, errors='coerce', format="mixed")
                    if datetime_converted.notna().mean() > DATETIME_INFERENCE_THRESHOLD:
                        col_profile["type_category"] = "should_be_datetime"
                        col_profile["min_date"] = str(datetime_converted.min())
                        col_profile["max_date"] = str(datetime_converted.max())
                    else:
                        # Treat as categorical
                        col_profile["type_category"] = "categorical"
                        value_counts = df[col].value_counts()
                        col_profile["top_values"] = value_counts.head(5).to_dict()
                        col_profile["cardinality"] = (
                            "high" if col_profile["unique_count"] > 20 else "low"
                        )
                except Exception:
                    col_profile["type_category"] = "categorical"
                    value_counts = df[col].value_counts()
                    col_profile["top_values"] = value_counts.head(TOP_VALUES_COUNT).to_dict()
                    col_profile["cardinality"] = (
                        "high" if col_profile["unique_count"] > HIGH_CARDINALITY_THRESHOLD else "low"
                    )

        else:
            col_profile["type_category"] = "categorical"
            value_counts = df[col].value_counts()
            col_profile["top_values"] = value_counts.head(5).to_dict()
            col_profile["cardinality"] = (
                "high" if col_profile["unique_count"] > HIGH_CARDINALITY_THRESHOLD else "low"
            )

        profile["columns"][col] = col_profile

    return profile


def generate_summary_text(profile: dict) -> str:
    """
    Converts a dataset profile into a plain-English summary card.
    """
    rows = profile["shape"]["rows"]
    cols = profile["shape"]["columns"]
    dupes = profile["duplicate_rows"]

    high_null_cols = [
        col for col, info in profile["columns"].items()
        if info["null_percent"] > 20
    ]
    high_skew_cols = [
        col for col, info in profile["columns"].items()
        if info.get("skew_label") == "high_skew"
    ]
    outlier_cols = [
        col for col, info in profile["columns"].items()
        if info.get("outlier_count", 0) > 0
    ]

    lines = [
        f"Dataset contains {rows:,} rows and {cols} columns.",
        f"Duplicate rows: {dupes}.",
    ]

    if high_null_cols:
        lines.append(f"High missing data (>20%): {', '.join(high_null_cols)}.")
    else:
        lines.append("No columns with critically high missing data.")

    if high_skew_cols:
        lines.append(
            f"Highly skewed columns (median recommended over mean): {', '.join(high_skew_cols)}."
        )

    if outlier_cols:
        lines.append(f"Outliers detected in: {', '.join(outlier_cols)}.")

    return " ".join(lines)


def drop_duplicate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Drops duplicate rows. Returns cleaned df and count dropped.
    """
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df, before - len(df)


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts column names to snake_case.
    """
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r'[\s\-]+', '_', regex=True)
        .str.replace(r'[^\w]', '', regex=True)
    )
    return df

def profile_dataset_chunked(df : pd.DataFrame) -> dict :
    """
    Profiles a dataset in chunks to handle large datasets without memory issues.
    Aggrerates numeric stats across chunks and samples categorical values.
    Only the profile is chunked, df remains intact.
    For files where full load is possible but slow.
    """
    total_rows = len(df)
    total_cols = len(df.columns)

    profile = {
        "shape" : {"rows": total_rows, "columns" : total_cols},
        "duplicate_rows" : int(df.duplicated().sum()),
        "columns" : {}
    }

    for col in df.columns :
        #Aggregate stats across chunks
        null_count = 0
        numeric_values = []
        value_counts = {}
        sample_values = []

        chunks = [
            df[col].iloc[i : i + CHUNK_SIZE_ROWS] for i in range(0, total_rows, CHUNK_SIZE_ROWS)
        ]

        for chunk in chunks :
            null_count += chunk.isnull().sum()

            if pd.api.types.is_numeric_dtype(chunk) :
                numeric_values.append(chunk.dropna().tolist())

            elif chunk.dtype == object :
                for val, count in chunk.value_counts().items() :
                    value_counts[val] = value_counts.get(val, 0) + count
                
            if not sample_values :
                sample_values = chunk.dropna().head(TOP_VALUES_COUNT).tolist()

        
        #Build column profile from aggregated stats
        col_profile = {}
        col_profile["dtype"] = str(df[col].dtype)
        col_profile["null_count"] = null_count
        col_profile["null_percent"] = round(null_count / total_rows * 100, 6)
        col_profile["unique_count"] = int(df[col].nunique())
        col_profile["sample_values"] = sample_values

        if numeric_values :
            arr = np.array(numeric_values)
            col_profile["type_category"] = "numeric"
            col_profile["mean"] = round(float(np.nanmean(arr)), 4)
            col_profile["median"] = round(float(np.median(arr)), 4)
            col_profile["std"] = round(float(arr.std()), 4)
            col_profile["min"] = round(float(arr.min()), 4)
            col_profile["max"] = round(float(arr.max()), 4)
            col_profile["skewness"] = round(float(df[col].skew()), 4)

            skew = abs(col_profile["skewness"])
            col_profile["skew_label"] = (
                "symmetric" if skew < SKEW_SYMMETRIC_MAX else
                "moderate_skew" if skew < SKEW_MODERATE_MAX else
                "high_skew"
            )

            Q1 = np.percentile(arr, 25)
            Q3 = np.percentile(arr, 75)
            IQR = Q3 - Q1
            lower = Q1 - OUTLIER_IQR_MULTIPLIER * IQR
            upper = Q3 + OUTLIER_IQR_MULTIPLIER * IQR
            col_profile["outlier_count"] = int(
                np.sum((arr < lower) | (arr > upper))
            )

        elif value_counts :
            col_profile["type_category"] = "categorical"
            sorted_counts = dict(sorted(value_counts.items(), key=lambda x: x[1], reverse=True))
            col_profile["top_values"] = dict(list(sorted_counts.items())[:TOP_VALUES_COUNT])
            col_profile["cardinality"] = (
                "high" if col_profile["unique_count"] > HIGH_CARDINALITY_THRESHOLD else "low"
            )

        else :
            col_profile["type_category"] = "other"
        
        profile["columns"][col] = col_profile

    return profile