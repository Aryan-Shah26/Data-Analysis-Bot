import pytest
import pandas as pd
from src.tools.dtype_tools import (
    detect_dtype_issues,
    fix_numeric_column,
    fix_datetime_column,
    fix_categorical_column,
    apply_dtype_fixes
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "numeric_as_str": ["1.0", "2.0", "3.0", "4.0", "5.0"],
        "date_as_str": ["2021-01-01", "2021-02-01", "2021-03-01",
                        "2021-04-01", "2021-05-01"],
        "low_cardinality": ["cat", "dog", "cat", "dog", "cat"],
        "already_numeric": [1.0, 2.0, 3.0, 4.0, 5.0],
    })


def test_detect_numeric_issue(sample_df):
    issues = detect_dtype_issues(sample_df)
    assert "numeric_as_str" in issues
    assert issues["numeric_as_str"] == "should_be_numeric"


def test_detect_datetime_issue(sample_df):
    issues = detect_dtype_issues(sample_df)
    assert "date_as_str" in issues
    assert issues["date_as_str"] == "should_be_datetime"


def test_detect_categorical_issue(sample_df):
    issues = detect_dtype_issues(sample_df)
    assert "low_cardinality" in issues
    assert issues["low_cardinality"] == "should_be_categorical"


def test_no_issue_for_already_numeric(sample_df):
    issues = detect_dtype_issues(sample_df)
    assert "already_numeric" not in issues


def test_fix_numeric_column(sample_df):
    df = fix_numeric_column(sample_df.copy(), "numeric_as_str")
    assert pd.api.types.is_numeric_dtype(df["numeric_as_str"])


def test_fix_numeric_column_invalid_values():
    df = pd.DataFrame({"col": ["1", "2", "not_a_number", "4"]})
    df = fix_numeric_column(df, "col")
    assert pd.api.types.is_numeric_dtype(df["col"])
    assert df["col"].isna().sum() == 1  # "not_a_number" becomes NaN


def test_fix_datetime_column(sample_df):
    df = fix_datetime_column(sample_df.copy(), "date_as_str")
    assert pd.api.types.is_datetime64_any_dtype(df["date_as_str"])


def test_fix_categorical_column(sample_df):
    df = fix_categorical_column(sample_df.copy(), "low_cardinality")
    assert str(df["low_cardinality"].dtype) == "category"


def test_apply_dtype_fixes_returns_log(sample_df):
    issues = {"numeric_as_str": "should_be_numeric"}
    df, log = apply_dtype_fixes(sample_df.copy(), issues)
    assert len(log) == 1
    assert "numeric_as_str" in log[0]


def test_apply_dtype_fixes_empty_issues(sample_df):
    df, log = apply_dtype_fixes(sample_df.copy(), {})
    assert len(log) == 0


def test_fix_numeric_all_nulls():
    df = pd.DataFrame({"col": [None, None, None]})
    df = fix_numeric_column(df, "col")
    assert pd.api.types.is_numeric_dtype(df["col"])