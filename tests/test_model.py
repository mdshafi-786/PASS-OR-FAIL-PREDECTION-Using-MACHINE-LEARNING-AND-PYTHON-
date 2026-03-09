import pytest
import joblib
import numpy as np
import os

MODEL_PATH = os.path.join('models', 'best_model.pkl')
SCALER_PATH = os.path.join('models', 'scaler.pkl')


@pytest.fixture
def model_and_scaler():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def test_model_files_exist():
    assert os.path.exists(MODEL_PATH), "Model file not found"
    assert os.path.exists(SCALER_PATH), "Scaler file not found"


def test_high_performer_passes(model_and_scaler):
    model, scaler = model_and_scaler
    data = scaler.transform([[10, 95, 98]])
    prediction = model.predict(data)
    assert prediction[0] == 1, "High-performing student should PASS"


def test_low_performer_fails(model_and_scaler):
    model, scaler = model_and_scaler
    data = scaler.transform([[1, 30, 40]])
    prediction = model.predict(data)
    assert prediction[0] == 0, "Low-performing student should FAIL"


def test_prediction_is_binary(model_and_scaler):
    model, scaler = model_and_scaler
    data = scaler.transform([[5, 70, 75]])
    prediction = model.predict(data)
    assert prediction[0] in [0, 1], "Prediction should be 0 or 1"


def test_batch_predictions(model_and_scaler):
    model, scaler = model_and_scaler
    data = scaler.transform([[10, 95, 98], [1, 30, 40], [5, 70, 75]])
    predictions = model.predict(data)
    assert len(predictions) == 3, "Should return 3 predictions"
    assert all(p in [0, 1] for p in predictions), "All predictions should be binary"
