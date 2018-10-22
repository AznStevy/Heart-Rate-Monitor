import pytest

from filtered_signal import FilteredSignal
from detection_algorithm import Threshold, Convolution, Wavelet


@pytest.fixture()
def test_1_data():
    """Normal signal, with noise"""
    metrics = {"num_beats": 35, "duration": 27.775, "voltage_extremes": [-0.68, 1.05],
               "beats": [0.217, 1.031, 1.844, 2.633, 3.422, 4.211, 5.028, 5.681, 6.678, 7.517, 8.328, 9.122, 9.889,
                         10.733, 11.586, 12.406, 13.239, 14.058, 14.853, 15.65, 16.444, 17.264, 18.133, 18.956, 19.739,
                         20.533, 21.306, 22.094, 22.906, 23.722, 24.55, 25.394, 26.2, 26.975, 27.772],
               "mean_hr_bpm": 75.60756075607561}
    return metrics


@pytest.fixture()
def test_21_data():
    """This data is very clean/no noise good for testing."""
    metrics = {"duration": 13.887, "num_beats": 19, "voltage_extremes": [-0.375, 0.60625],
               "beats": [0.044, 0.794, 1.544, 2.294, 3.044, 3.794, 4.544, 5.294, 6.044, 6.794, 7.544, 8.294, 9.044,
                         9.794, 10.544, 11.294, 12.044, 12.794, 13.544], "mean_hr_bpm": 82.09116439835817}
    return metrics


@pytest.fixture()
def test_variables():
    rel_info = {
        "test_1_file": "tests/test_data/test_data1.csv",
        "test_21_file": "tests/test_data/test_data21.csv"
    }


# --------------- test constructor -------------------
def test_constructor(test_variables):
    time_list = test_variables["time"]
    signal_line = test_variables["line"]
    _ = FilteredSignal(time_list, signal_line)
    assert True
