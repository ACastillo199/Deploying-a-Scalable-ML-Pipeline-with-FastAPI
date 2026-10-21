import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml.data import apply_label, process_data
from ml.model import compute_model_metrics, train_model

# TODO: implement the first test. Change the function name and input as needed
def test_apply_label():
    """
    Test the apply_label function to ensure it correctly maps binary predictions to string outputs.
    """
    assert apply_label([1]) == ">50K"
    assert apply_label([0]) == "<=50K"


# TODO: implement the second test. Change the function name and input as needed
def test_train_model():
    """
    Test if the train_model function successfully returns a model using the expected RandomForestClassifier algorithm.
    """
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    y = np.array([0, 1, 0, 1])
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


# TODO: implement the third test. Change the function name and input as needed
def test_compute_model_metrics():
    """
    Test if the compute_model_metrics function returns valid float values for precision, recall, and fbeta.
    """
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
