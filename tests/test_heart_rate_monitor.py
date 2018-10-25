import os
import json
import pytest
from file_handler import FileHandler
from heart_rate_monitor import HeartRateMonitor
from detection_algorithm import Threshold, Wavelet


@pytest.fixture()
def test_variables():
    rel_info = {
        "test_1_file": "tests/test_data/test_data1.csv",
        "test_21_file": "tests/test_data/test_data21.csv"
    }
    return rel_info


@pytest.fixture()
def hrm_wavelet_obj():
    return HeartRateMonitor("tests/test_data/test_data1.csv", Wavelet)


@pytest.fixture()
def hrm_threshold_obj():
    return HeartRateMonitor("tests/test_data/test_data1.csv", Threshold)


# --------------- test constructor -------------------
@pytest.mark.parametrize("analyzer", [
    Threshold, Wavelet])
def test_constructor(test_variables, analyzer):
    # attempt to create a variable
    assert HeartRateMonitor(test_variables["test_1_file"], analyzer)


@pytest.mark.parametrize("filename, analyzer, error", [
    ("tests/test_data/test_fake.csv", Wavelet, FileNotFoundError),
    ("tests/test_data/test_data1.csv", bool, TypeError),
    ("tests/test_data/test_data1.csv", HeartRateMonitor, TypeError)
])
def test_hrm_constructor_bad_file(filename, analyzer, error):
    # attempt to create a variable
    with pytest.raises(error):
        HeartRateMonitor(filename, analyzer)


@pytest.mark.parametrize("filename, analyzer", [
    ("tests/test_data/test_data1.csv", bool),
    ("tests/test_data/test_data1.csv", HeartRateMonitor)
])
def test_hrm_constructor_bad_analyzer_type(filename, analyzer):
    with pytest.raises(TypeError):
        HeartRateMonitor(filename, analyzer)


# ----------------- test to_json ---------------------
def test_hrm_to_json(hrm_wavelet_obj):
    # attempt to create a variable
    ret_json = hrm_wavelet_obj.to_json()
    must_ret = ["beats", "duration", "num_beats", "mean_hr_bpm", "voltage_extremes"]

    assert set(ret_json.keys()).issubset(set(must_ret))


# ----------------- test write_json -------------------
def test_hrm_write_json_file(hrm_wavelet_obj):
    # attempt to create a variable
    filename = hrm_wavelet_obj.write_json()
    assert os.path.exists(filename)


def test_hrm_write_json_file_read(hrm_wavelet_obj):
    # attempt to create a variable
    filename = hrm_wavelet_obj.write_json()

    with open(filename) as data_file:
        data = json.load(data_file)

    # I really just care if it loads/wrote properly
    assert data
