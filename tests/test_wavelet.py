import pytest
import numpy as np
from file_handler import FileHandler
from filtered_signal import FilteredSignal
from detection_algorithm import Wavelet


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


# test variables
def filtered_signal_obj(file_num):
    filename = "tests/test_data/test_data{}.csv".format(file_num)
    file = FileHandler(filename)
    filtered_signal_sp = FilteredSignal(file.time, file.signal)
    return filtered_signal_sp


@pytest.fixture()
def test_1_filtered_signal_obj():
    return filtered_signal_obj(1)


@pytest.fixture()
def test_21_filtered_signal_obj():
    return filtered_signal_obj(21)


# ---------------- Wavelet class fixtures ----------------------
@pytest.fixture()
def WaveletObj_1(test_1_filtered_signal_obj):
    time = test_1_filtered_signal_obj.time
    signal = test_1_filtered_signal_obj.raw_signal
    return Wavelet(time, signal)


@pytest.fixture()
def WaveletObj_21(test_21_filtered_signal_obj):
    time = test_21_filtered_signal_obj.time
    signal = test_21_filtered_signal_obj.raw_signal
    return Wavelet(time, signal)


# ---------------- test wavelet instantiation -------------------
def test_wavelet_instantiation(test_21_filtered_signal_obj):
    time = test_21_filtered_signal_obj.time
    signal = test_21_filtered_signal_obj.raw_signal
    return Wavelet(time, signal)


# ----------------- test wavelet find_beats ---------------------
@pytest.mark.parametrize("reverse_threshold", [
    True, False])
def test_wavelet_find_beats_inputs(WaveletObj_21, reverse_threshold):
    assert WaveletObj_21.find_beats(reverse_threshold=reverse_threshold).size != 0


@pytest.mark.parametrize("reverse_threshold, error", [
    ("1", TypeError),
    ([1, 2, 3], TypeError)])
def test_wavelet_find_beats_bad_inputs(WaveletObj_21, reverse_threshold, error):
    with pytest.raises(error):
        WaveletObj_21.find_beats(reverse_threshold=reverse_threshold)


def test_wavelet_find_beats_output_type(WaveletObj_21):
    resp = WaveletObj_21.find_beats()
    assert type(resp) == np.ndarray


# ----------------- test wavelet _wavelet_transform ---------------------
def test_wavelet__wavelet_transform(WaveletObj_21):
    # I trust that numpy checks that it does what it's supposed to.
    resp = WaveletObj_21._wavelet_transform()
    assert resp.size != 0


def test_wavelet__wavelet_transform_output_type(WaveletObj_21):
    # I trust that numpy checks that it does what it's supposed to.
    resp = WaveletObj_21._wavelet_transform()
    assert type(resp) == np.ndarray
