import pytest
import pandas as pd
import numpy as np
from src.tools.outlier_tools import (
    detect_outliers_iqr,
    detect_outliers_zscore,
    cap_outliers_iqr,
    drop_outliers_iqr,
    apply_outlier_fixes
)


@pytest.fixture
def df_with_outliers():
    # 300 is a clear outlier in this distribution
    return pd.DataFrame({
        "age": [25.0, 22.0, 28.0, 24.0, 26.0, 300.0],
        "salary": [50000.0, 55000.0, 60000.0, 52000.0, 58000.0, 57000.0]
    })


def test_detect_outliers_iqr_finds_outlier(df_with_outliers):
    mask = detect_outliers_iqr(df_with_outliers, "age")
    assert mask.sum() == 1
    assert mask.iloc[5] == True


def test_detect_outliers_iqr_no_outliers(df_with_outliers):
    mask = detect_outliers_iqr(df_with_outliers, "salary")
    assert mask.sum() == 0


def test_detect_outliers_zscore(df_with_outliers):
    mask = detect_outliers_zscore(df_with_outliers, "age")
    assert mask.sum() == 1
    assert mask.iloc[5] == True


def test_cap_outliers_preserves_row_count(df_with_outliers):
    df, count = cap_outliers_iqr(df_with_outliers.copy(), "age")
    assert len(df) == len(df_with_outliers)
    assert count == 1


def test_cap_outliers_reduces_max(df_with_outliers):
    original_max = df_with_outliers["age"].max()
    df, _ = cap_outliers_iqr(df_with_outliers.copy(), "age")
    assert df["age"].max() < original_max


def test_drop_outliers_reduces_row_count(df_with_outliers):
    df, count = drop_outliers_iqr(df_with_outliers.copy(), "age")
    assert len(df) == len(df_with_outliers) - 1
    assert count == 1


def test_drop_outliers_removes_outlier_value(df_with_outliers):
    df, _ = drop_outliers_iqr(df_with_outliers.copy(), "age")
    assert 300.0 not in df["age"].values


def test_apply_outlier_fixes_cap(df_with_outliers):
    profile = {
        "columns": {
            "age": {"type_category": "numeric", "outlier_count": 1},
            "salary": {"type_category": "numeric", "outlier_count": 0}
        }
    }
    df, log = apply_outlier_fixes(df_with_outliers.copy(), profile, strategy="cap")
    assert len(df) == len(df_with_outliers)
    assert any("age" in entry for entry in log)


def test_apply_outlier_fixes_skips_no_outliers(df_with_outliers):
    profile = {
        "columns": {
            "salary": {"type_category": "numeric", "outlier_count": 0}
        }
    }
    df, log = apply_outlier_fixes(df_with_outliers.copy(), profile, strategy="cap")
    assert len(log) == 0


def test_no_outliers_in_uniform_data():
    df = pd.DataFrame({"col": [5.0, 5.0, 5.0, 5.0, 5.0]})
    mask = detect_outliers_iqr(df, "col")
    assert mask.sum() == 0