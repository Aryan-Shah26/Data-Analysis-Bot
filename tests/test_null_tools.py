import pytest
import pandas as pd
import numpy as np
from src.tools.null_tools import (
    get_null_strategy,
    fill_with_mean,
    fill_with_median,
    fill_with_mode,
    fill_forward,
    drop_column,
    drop_rows_with_nulls,
    apply_null_fixes
)


@pytest.fixture
def numeric_skewed_profile():
    return {
        "null_percent": 20.0,
        "null_count": 2,
        "type_category": "numeric",
        "skew_label": "high_skew"
    }


@pytest.fixture
def numeric_symmetric_profile():
    return {
        "null_percent": 20.0,
        "null_count": 2,
        "type_category": "numeric",
        "skew_label": "symmetric"
    }


@pytest.fixture
def categorical_profile():
    return {
        "null_percent": 10.0,
        "null_count": 1,
        "type_category": "categorical"
    }


@pytest.fixture
def high_null_profile():
    return {
        "null_percent": 75.0,
        "null_count": 75,
        "type_category": "numeric",
        "skew_label": "symmetric"
    }


def test_strategy_high_null_drops_column(high_null_profile):
    assert get_null_strategy(high_null_profile) == "drop_column"


def test_strategy_skewed_numeric_uses_median(numeric_skewed_profile):
    assert get_null_strategy(numeric_skewed_profile) == "fill_median"


def test_strategy_symmetric_numeric_uses_mean(numeric_symmetric_profile):
    assert get_null_strategy(numeric_symmetric_profile) == "fill_mean"


def test_strategy_categorical_uses_mode(categorical_profile):
    assert get_null_strategy(categorical_profile) == "fill_mode"


def test_strategy_datetime_uses_forward_fill():
    profile = {"null_percent": 10.0, "null_count": 1, "type_category": "datetime"}
    assert get_null_strategy(profile) == "fill_forward"


def test_fill_with_mean():
    df = pd.DataFrame({"col": [1.0, 2.0, None, 4.0]})
    df = fill_with_mean(df, "col")
    assert df["col"].isna().sum() == 0
    assert df["col"].iloc[2] == pytest.approx(7 / 3, rel=1e-3)


def test_fill_with_median():
    df = pd.DataFrame({"col": [1.0, 2.0, None, 4.0]})
    df = fill_with_median(df, "col")
    assert df["col"].isna().sum() == 0
    assert df["col"].iloc[2] == 2.0


def test_fill_with_mode():
    df = pd.DataFrame({"col": ["cat", "dog", "cat", None]})
    df = fill_with_mode(df, "col")
    assert df["col"].isna().sum() == 0
    assert df["col"].iloc[3] == "cat"


def test_fill_forward():
    df = pd.DataFrame({"col": [1.0, None, None, 4.0]})
    df = fill_forward(df, "col")
    assert df["col"].iloc[1] == 1.0
    assert df["col"].iloc[2] == 1.0


def test_drop_column():
    df = pd.DataFrame({"keep": [1, 2, 3], "drop": [None, None, None]})
    df = drop_column(df, "drop")
    assert "drop" not in df.columns
    assert "keep" in df.columns


def test_drop_rows_with_nulls():
    df = pd.DataFrame({"col": [1.0, None, 3.0, None]})
    df = drop_rows_with_nulls(df, "col")
    assert len(df) == 2
    assert df["col"].isna().sum() == 0


def test_fill_mean_no_nulls():
    df = pd.DataFrame({"col": [1.0, 2.0, 3.0]})
    df = fill_with_mean(df, "col")
    assert df["col"].tolist() == [1.0, 2.0, 3.0]


def test_fill_mode_empty_column():
    df = pd.DataFrame({"col": [None, None, None]})
    df = fill_with_mode(df, "col")
    # No mode available — should not crash
    assert len(df) == 3