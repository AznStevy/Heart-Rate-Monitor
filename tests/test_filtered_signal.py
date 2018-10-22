import pytest
import numpy as np

from file_handler import FileHandler
from filtered_signal import FilteredSignal


# useful, keep in mind
# https://github.com/pytest-dev/pytest/issues/349

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


@pytest.fixture()
def test_1_filtered_signal_data():
    """Normal signal, with noise."""
    properties = {'low_pass_cutoff': 30, 'fs': 333.3333333333207, 'high_pass_cutoff': 1, 'period': 100}
    return properties


@pytest.fixture()
def test_21_filtered_signal_data():
    """This data is very clean/no noise good for testing."""
    properties = {'low_pass_cutoff': 30, 'fs': 1000.0000000005542, 'high_pass_cutoff': 1, 'period': 100}
    return properties


@pytest.fixture()
def test_variables():
    time_list = np.linspace(0, 10, 10)
    signal_line = np.ones(len(time_list))
    freq = 1  # hz
    fs = 2  # sampling freq

    # = np.arange(time_list * fs)
    # signal_sinusoid = np.sin(2 * np.pi * freq * samples)

    object = {
        "freq": freq,
        "fs": fs,
        "time": time_list,
        "line": signal_line,
        # "sinusoid": signal_sinusoid
    }
    return object


# ---------------------- tests -----------------------------
def test_constructor(test_variables):
    time_list = test_variables["time"]
    signal_line = test_variables["line"]
    _ = FilteredSignal(time_list, signal_line)
    assert True


@pytest.mark.parametrize("low_pass_cutoff", [
    20])
def test_constructor_with_low_pass(test_variables, low_pass_cutoff):
    time_list = test_variables["time"]
    signal_line = test_variables["line"]
    _ = FilteredSignal(time_list, signal_line, low_pass_cutoff=low_pass_cutoff)
    assert True


@pytest.mark.parametrize("low_pass_cutoff", [
    "test",
    23.4,
])
def test_constructor_with_bad_low_pass(test_variables, low_pass_cutoff):
    time_list = test_variables["time"]
    signal_line = test_variables["line"]
    with pytest.raises(TypeError):
        _ = FilteredSignal(time_list, signal_line, low_pass_cutoff=low_pass_cutoff)


@pytest.mark.parametrize("high_pass_cutoff", [
    3])
def test_constructor_with_high_pass(test_variables, high_pass_cutoff):
    time_list = test_variables["time"]
    signal_line = test_variables["line"]
    _ = FilteredSignal(time_list, signal_line, high_pass_cutoff=high_pass_cutoff)
    assert True


@pytest.mark.parametrize("high_pass_cutoff", [
    "test",
    3.4, ])
def test_constructor_with_bad_high_pass(test_variables, high_pass_cutoff):
    with pytest.raises(TypeError):
        time_list = test_variables["time"]
        signal_line = test_variables["line"]
        _ = FilteredSignal(time_list, signal_line,
                           high_pass_cutoff=high_pass_cutoff)


@pytest.mark.parametrize("high_pass_cutoff, low_pass_cutoff", [
    (3, 50),
    (5, 10), ])
def test_constructor_with_high_low_pass(
        test_variables, high_pass_cutoff, low_pass_cutoff):
    time_list = test_variables["time"]
    signal_line = test_variables["line"]
    _ = FilteredSignal(time_list, signal_line,
                       high_pass_cutoff=high_pass_cutoff,
                       low_pass_cutoff=low_pass_cutoff)
    assert True


@pytest.mark.parametrize("high_pass_cutoff, low_pass_cutoff", [
    (50, 3),
    (10, 5), ])
def test_constructor_with_bad_high_low_pass(
        test_variables, high_pass_cutoff, low_pass_cutoff):
    with pytest.raises(ValueError):
        time_list = test_variables["time"]
        signal_line = test_variables["line"]
        _ = FilteredSignal(time_list, signal_line,
                           high_pass_cutoff=high_pass_cutoff,
                           low_pass_cutoff=low_pass_cutoff)


# --------------- check time -----------------------
@pytest.mark.parametrize("time_list, error", [
    (["1", "2", "3"], ValueError),
    (np.array(["1", "2", "3"]), ValueError),
    (("1", "2", "3"), TypeError)])
def test_constructor_values(test_variables, time_list, error):
    with pytest.raises(error):
        signal_line = test_variables["line"]
        _ = FilteredSignal(time_list, signal_line)


# --------------- check determine frequency -----------------------
def test_determine_frequency_no_input(test_1_filtered_signal_obj, test_1_filtered_signal_data):
    freq = test_1_filtered_signal_obj.determine_frequency()
    assert round(freq) == round(test_1_filtered_signal_data["fs"])


def test_determine_frequency_time_input(
        test_1_filtered_signal_obj, test_21_filtered_signal_obj,
        test_1_filtered_signal_data, test_21_filtered_signal_data):
    freq = test_1_filtered_signal_obj.determine_frequency(test_21_filtered_signal_obj.time)
    assert round(freq) == round(test_21_filtered_signal_data["fs"])


def test_determine_frequency_time_bad_input_none(test_1_filtered_signal_obj, test_1_filtered_signal_data):
    freq = test_1_filtered_signal_obj.determine_frequency(None)
    assert round(freq) == round(test_1_filtered_signal_data["fs"])


# ----------------- check get_properties ----------------------------
def test_get_properties_output(test_1_filtered_signal_obj):
    output = test_1_filtered_signal_obj.get_properties()
    required = ["high_pass_cutoff", "low_pass_cutoff", "period", "fs"]
    assert set(output.keys()).issubset(required)


# ----------------- check moving_average_sub -------------------------
def test_apply_moving_average_sub(test_1_filtered_signal_obj):
    """Testing moving average."""
    assert test_1_filtered_signal_obj.bg_sub_signal is not None


@pytest.mark.parametrize("signal, error", [
    (["1", "2", "3"], ValueError),
    (np.array(["1", "2", "3"]), ValueError),
    (("1", "2", "3"), TypeError)])
def test_apply_moving_average_sub_bad_input(test_1_filtered_signal_obj, signal, error):
    with pytest.raises(error):
        test_1_filtered_signal_obj.apply_moving_average_sub(signal)


def test_apply_moving_average_sub_output(test_1_filtered_signal_obj):
    assert type(test_1_filtered_signal_obj.apply_moving_average_sub()) == np.ndarray


# ---------------- test noise_reduction ----------------------------
@pytest.mark.parametrize("signal, error", [
    (("1", "2", "3"), TypeError),
    (["1", "2", "3"], ValueError)])
def test_apply_noise_reduction_bad_inputs(test_1_filtered_signal_obj, signal, error):
    with pytest.raises(error):
        test_1_filtered_signal_obj.apply_moving_average_sub(signal)


@pytest.mark.parametrize("signal", [
    (["1", "2", "3"])])
def test_apply_noise_reduction_outputs(test_1_filtered_signal_obj, signal):
    assert type(test_1_filtered_signal_obj.apply_noise_reduction(signal)) == np.ndarray


# ------------------ test apply_high_pass --------------------------
@pytest.mark.parametrize("signal, high_cutoff, order, error", [
    (("1", "2", "3"), None, None, TypeError),
    (["1", "2", "3"], None, None, ValueError),
    ([1, 2, 3], -1, None, ValueError),
    ([1, 2, 3], None, -1, ValueError)
])
def test_apply_high_pass_bad_inputs(test_1_filtered_signal_obj, signal, high_cutoff, order, error):
    with pytest.raises(error):
        test_1_filtered_signal_obj.apply_high_pass(signal, high_cutoff=high_cutoff, order=order)


def test_apply_high_pass_output(test_1_filtered_signal_obj):
    assert type(test_1_filtered_signal_obj.apply_high_pass()) == np.ndarray


# -------------------- test apply_low_pass --------------------------
@pytest.mark.parametrize("signal, low_cutoff, order, error", [
    (("1", "2", "3"), None, None, TypeError),
    (["1", "2", "3"], None, None, ValueError),
    ([1, 2, 3], -1, None, ValueError),
    ([1, 2, 3], None, -1, ValueError)
])
def test_apply_low_pass_bad_inputs(test_1_filtered_signal_obj, signal, low_cutoff, order, error):
    with pytest.raises(error):
        test_1_filtered_signal_obj.apply_low_pass(signal, low_cutoff=low_cutoff, order=order)


def test_apply_low_pass_output(test_1_filtered_signal_obj):
    assert type(test_1_filtered_signal_obj.apply_low_pass()) == np.ndarray


# --------------------- test get_fft ---------------------------------
@pytest.mark.parametrize("signal, is_filtered, error", [
    (("1", "2", "3"), None, TypeError),
    (["1", "2", "3"], 2, TypeError),
])
def test_get_fft_bad_inputs(test_1_filtered_signal_obj, signal, is_filtered, error):
    with pytest.raises(error):
        test_1_filtered_signal_obj.get_fft(signal=signal, is_filtered=is_filtered)


def test_get_fft_outputs(test_1_filtered_signal_obj):
    output = test_1_filtered_signal_obj.get_fft()
    assert len(output) == 2 and type(output[0]) == np.ndarray and type(output[1]) == np.ndarray
