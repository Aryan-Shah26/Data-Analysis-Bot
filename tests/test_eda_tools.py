import pytest
import pandas as pd
import numpy as np
from src.tools.eda_tools import (
    profile_dataset,
    generate_summary_text,
    drop_duplicate_rows,
    standardize_column_names
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Name": ["Alice", "Bob", None, "Dave", "Alice"],
        "Age": [25.0, 300.0, 22.0, None, 25.0],
        "Salary": ["50000", "60000", "55000", "70000", "50000"],
        "Join Date": ["2021-01-01", "2021-06-15", None,
                      "2022-03-01", "2021-01-01"]
    })


def test_profile_returns_correct_shape(sample_df):
    profile = profile_dataset(sample_df)
    assert profile["shape"]["rows"] == 5
    assert profile["shape"]["columns"] == 4


def test_profile_detects_duplicates(sample_df):
    profile = profile_dataset(sample_df)
    assert profile["duplicate_rows"] == 1


def test_profile_detects_nulls(sample_df):
    profile = profile_dataset(sample_df)
    assert profile["columns"]["Age"]["null_count"] == 1
    assert profile["columns"]["Age"]["null_percent"] == 20.0


def test_profile_numeric_has_stats(sample_df):
    profile = profile_dataset(sample_df)
    age = profile["columns"]["Age"]
    assert "mean" in age
    assert "median" in age
    assert "skewness" in age
    assert "outlier_count" in age


def test_profile_detects_should_be_numeric(sample_df):
    profile = profile_dataset(sample_df)
    assert profile["columns"]["Salary"]["type_category"] == "should_be_numeric"


def test_profile_detects_should_be_datetime(sample_df):
    profile = profile_dataset(sample_df)
    assert profile["columns"]["Join Date"]["type_category"] == "should_be_datetime"


def test_profile_skew_label_assigned(sample_df):
    profile = profile_dataset(sample_df)
    assert profile["columns"]["Age"]["skew_label"] in (
        "symmetric", "moderate_skew", "high_skew"
    )


def test_generate_summary_text_contains_shape(sample_df):
    profile = profile_dataset(sample_df)
    summary = generate_summary_text(profile)
    assert "5" in summary
    assert "4" in summary


def test_drop_duplicate_rows(sample_df):
    df, count = drop_duplicate_rows(sample_df)
    assert count == 1
    assert len(df) == 4


def test_drop_duplicate_rows_no_duplicates():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df, count = drop_duplicate_rows(df)
    assert count == 0
    assert len(df) == 3


def test_standardize_column_names():
    df = pd.DataFrame({"First Name": [1], "Last-Name": [2], "Age!": [3]})
    df = standardize_column_names(df)
    assert "first_name" in df.columns
    assert "last_name" in df.columns
    assert "age" in df.columns


def test_standardize_already_clean():
    df = pd.DataFrame({"name": [1], "age": [2]})
    df = standardize_column_names(df)
    assert list(df.columns) == ["name", "age"]


def test_profile_empty_dataframe():
    df = pd.DataFrame()
    profile = profile_dataset(df)
    assert profile["shape"]["rows"] == 0
    assert profile["shape"]["columns"] == 0


def test_profile_all_nulls_column():
    df = pd.DataFrame({"col": [None, None, None]})
    profile = profile_dataset(df)
    assert profile["columns"]["col"]["null_percent"] == 100.0