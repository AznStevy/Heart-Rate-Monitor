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


def test_apply_moving_average_sub_functionatlity_same(test_1_filtered_signal_obj):
    assert np.array_equal(test_1_filtered_signal_obj.bg_sub_signal,
                          test_1_filtered_signal_obj.apply_moving_average_sub(test_1_filtered_signal_obj.raw_signal))


def test_apply_moving_average_sub_functionatlity_cross(
        test_1_filtered_signal_obj, test_21_filtered_signal_obj):
    assert np.array_equal(test_1_filtered_signal_obj.bg_sub_signal,
                          test_21_filtered_signal_obj.apply_moving_average_sub(test_1_filtered_signal_obj.raw_signal))


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


# ------------------ filtering helper methods -------------------------
@pytest.fixture()
def freq_values():
    return [50, 100, 150, 200]


def sinusoid(freq, fs, a=1):
    duration = 20  # seconds
    t = np.linspace(0, duration, duration * fs)
    return t, a * np.sin(2 * np.pi * freq * t)


@pytest.fixture()
def sinu_signal_data():
    fs = 1000
    t, sinusoid_50 = sinusoid(50, fs)
    _, sinusoid_100 = sinusoid(100, fs)
    _, sinusoid_150 = sinusoid(150, fs)
    _, sinusoid_200 = sinusoid(200, fs)
    sinusoid_combined = sinusoid_50 + sinusoid_100 + sinusoid_150 + sinusoid_200

    sinu_signal = {
        "signal": sinusoid_combined,
        "fs": fs,
        "time": t
    }

    return sinu_signal


def _determine_good_filter(acceptable, percent_change):
    is_filtered_properly = []
    if len(acceptable) == len(percent_change):
        for i, acc_per in enumerate(acceptable):
            if acc_per >= 0:
                # test to see if it's within this range
                if abs(percent_change[i]) < acc_per:
                    is_filtered_properly.append(True)
                else:
                    is_filtered_properly.append(False)
            else:
                # if negative, test to see if the mag drop is lower
                if percent_change[i] < acc_per:
                    is_filtered_properly.append(True)
                else:
                    is_filtered_properly.append(False)

    is_filtered_properly = np.array(is_filtered_properly)
    return is_filtered_properly.size != 0 and is_filtered_properly.all()


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


@pytest.mark.parametrize("high_cutoff, order, acceptable", [
    (70, 7, [-70, 8, 8, 8]),
    (120, 7, [-70, -70, 8, 8]),
    (170, 7, [-70, -70, -70, 8]),
    (220, 7, [-70, -70, -70, -70])
])
def test_high_pass_functionality(test_1_filtered_signal_obj, sinu_signal_data, freq_values,
                                 high_cutoff, order, acceptable):
    # freq_values are [50, 100, 150, 200]
    data = sinu_signal_data
    sinusoid_combined = data["signal"]
    fs = data["fs"]

    filtered_sinusoid = test_1_filtered_signal_obj.apply_high_pass(
        sinusoid_combined, high_cutoff=high_cutoff, order=order, fs=fs)

    frq, original_mag = test_1_filtered_signal_obj.get_fft(sinusoid_combined)
    _, filtered_mag = test_1_filtered_signal_obj.get_fft(filtered_sinusoid)

    indices = [1000, 2000, 3000, 4000]
    original_mag = abs(original_mag[indices])
    filtered_mag = abs(filtered_mag[indices])
    percent_change = ((filtered_mag - original_mag) / original_mag) * 100
    assert _determine_good_filter(acceptable, percent_change)


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


@pytest.mark.parametrize("low_cutoff, order, acceptable", [
    (70, 7, [8, -70, -70, -70]),
    (130, 7, [8, 8, -70, -70]),
    (180, 7, [8, 8, 8, -70]),
    (230, 7, [8, 8, 8, 8])
])
def test_low_pass_functionality(test_1_filtered_signal_obj, freq_values, sinu_signal_data,
                                low_cutoff, order, acceptable):
    # test frequencies are [50, 100, 150, 200]
    data = sinu_signal_data
    sinusoid_combined = data["signal"]
    fs = data["fs"]

    filtered_sinusoid = test_1_filtered_signal_obj.apply_low_pass(
        sinusoid_combined, low_cutoff=low_cutoff, order=order, fs=fs)

    frq, original_mag = test_1_filtered_signal_obj.get_fft(sinusoid_combined)
    _, filtered_mag = test_1_filtered_signal_obj.get_fft(filtered_sinusoid)

    indices = [1000, 2000, 3000, 4000]
    original_mag = abs(original_mag[indices])
    filtered_mag = abs(filtered_mag[indices])
    percent_change = ((filtered_mag - original_mag) / original_mag) * 100
    assert _determine_good_filter(acceptable, percent_change)
