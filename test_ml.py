import pytest
import pandas as pd

from ml.model import compute_model_metrics

# TODO: implement the first test. Change the function name and input as needed
def test_dataframe_not_empty():
    """
    Verify that the census dataset contains records.
    """
    df = pd.read_csv("data/census.csv")
    assert len(df) > 0


# TODO: implement the second test. Change the function name and input as needed
def test_compute_model_metrics():
    """
    Verify precision, recall, and F1 are between 0 and 1.
    """
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 0, 0]

    precision, recall, f1 = compute_model_metrics(y_true, y_pred)

    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= f1 <= 1


# TODO: implement the third test. Change the function name and input as needed
def test_salary_column_exists():
    """
    Verify the target column exists in the census dataset.
    """
    df = pd.read_csv("data/census.csv")
    assert "salary" in df.columns
