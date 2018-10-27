import pytest
import numpy as np
from file_handler import FileHandler
from filtered_signal import FilteredSignal
from detection_algorithm import ECGDetectionAlgorithm, Threshold


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
    return rel_info


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


# ---------- Abstract class fixtures --------------
@pytest.fixture()
def ECGDetectionAlgObj_1():
    time = test_1_filtered_signal_obj().time
    signal = test_1_filtered_signal_obj().raw_signal
    return ECGDetectionAlgorithm(time, signal)


@pytest.fixture()
def ECGDetectionAlgObj_21():
    time = test_21_filtered_signal_obj().time
    signal = test_21_filtered_signal_obj().raw_signal
    return ECGDetectionAlgorithm(time, signal)


# -------------------- test ECGDetectionAlgorithms ------------
def test_ECG_obj_constructor(test_1_filtered_signal_obj):
    time = test_1_filtered_signal_obj.time
    signal = test_1_filtered_signal_obj.raw_signal

    return ECGDetectionAlgorithm(time, signal)


@pytest.mark.parametrize("time, signal, error", [
    ((1, 2, 3), [1, 2, 3], TypeError),
    ([1, 2, 3], (1, 2, 3), TypeError),
    ([], [1, 2, 3], ValueError),
    ([1, 2, 3], [], ValueError),
    ([1, 2, 3], [1, 2], ValueError),
])
def test_ECG_constructor_bad_parameters(time, signal, error):
    with pytest.raises(error):
        ECGDetectionAlgorithm(time, signal)


# ----------- test find_voltage_extremes --------------
@pytest.mark.parametrize("signal, expected", [
    ([1, 2, 3], (1, 3)),
    ([-5, 6, 1, 2, 3.5, 4], (-5, 6))])
def test_ECG__find_voltage_extremes(ECGDetectionAlgObj_1, signal, expected):
    assert ECGDetectionAlgObj_1._find_voltage_extremes(signal) == expected


@pytest.mark.parametrize("signal, error", [
    ((1, 2, 3), TypeError)])
def test_ECG__find_voltage_extremes_bad_input(ECGDetectionAlgObj_1, signal, error):
    # test if first is smaller than second
    with pytest.raises(error):
        ECGDetectionAlgObj_1._find_voltage_extremes(signal)


@pytest.mark.parametrize("signal", [
    [1, 2, 3],
    [-5, 6, 1, 2, 3.5, 4]])
def test_ECG__find_voltage_extremes_output(ECGDetectionAlgObj_1, signal):
    assert type(ECGDetectionAlgObj_1._find_voltage_extremes(signal)) == tuple


# ----------- test find_duration --------------
def test_ECG_find_duration(ECGDetectionAlgObj_1, test_1_data):
    assert ECGDetectionAlgObj_1.find_duration() == test_1_data["duration"]


def test_ECG_find_duration_output(ECGDetectionAlgObj_1):
    assert type(ECGDetectionAlgObj_1.find_duration()) == float


# ---------- Threshold class fixtures --------------
@pytest.fixture()
def ThresholdObj_1(test_1_filtered_signal_obj):
    time = test_1_filtered_signal_obj.time
    signal = test_1_filtered_signal_obj.raw_signal
    return Threshold(time, signal)


@pytest.fixture()
def ThresholdObj_21(test_21_filtered_signal_obj):
    time = test_21_filtered_signal_obj.time
    signal = test_21_filtered_signal_obj.raw_signal
    return Threshold(time, signal)


# -------------------- test THRESHOLD constructor -----------------------
def test_threshold_constructor(test_1_filtered_signal_obj):
    time = test_1_filtered_signal_obj.time
    signal = test_1_filtered_signal_obj.raw_signal
    assert Threshold(time, signal)


@pytest.mark.parametrize(
    "time, signal, high_cutoff, low_cutoff, threshold_frac, error", [
        ([1, 2, 3], [1, 2, 3], 1, "test", 0.5, TypeError),
        ([1, 2, 3], [1, 2, 3], "test", 1, 0.5, TypeError),
        ([1, 2, 3], [1, 2, 3], "test", None, 0.5, TypeError),
        ([1, 2, 3], [1, 2, 3], 1, 50, "test", TypeError), ])
def test_threshold_constructor_bad_args_type_error(
        time, signal, high_cutoff, low_cutoff, threshold_frac, error):
    with pytest.raises(error):
        Threshold(time=time, signal=signal,
                  high_pass_cutoff=high_cutoff, low_pass_cutoff=low_cutoff,
                  threshold_frac=threshold_frac)


@pytest.mark.parametrize(
    "time, signal, high_cutoff, low_cutoff, threshold_frac, error", [
        ([1, 2, 3], [1, 2, 3], 1, 40, 3, ValueError),
        ([1, 2, 3], [1, 2, 3], 1, 40, -1, ValueError)])
def test_threshold_constructor_bad_args_value_error(
        time, signal, high_cutoff, low_cutoff, threshold_frac, error):
    with pytest.raises(error):
        Threshold(time=time, signal=signal,
                  high_pass_cutoff=high_cutoff, low_pass_cutoff=low_cutoff,
                  threshold_frac=threshold_frac)


# ----------------- test threshold find_beats -----------------
def test_threshold_find_beats_output(ThresholdObj_21, test_21_data):
    assert abs(len(ThresholdObj_21.find_beats()) - len(test_21_data["beats"])) < 3


def test_threshold_find_beats_output_type(ThresholdObj_21):
    assert type(ThresholdObj_21.find_beats()) == np.ndarray


# ----------- test find_num_beats --------------
def test_threshold_find_num_beats(ThresholdObj_21, test_21_data):
    assert abs(ThresholdObj_21.find_num_beats() - len(test_21_data["beats"])) < 2


def test_threshold_find_num_beats_output(ThresholdObj_21):
    assert type(ThresholdObj_21.find_num_beats()) == int


# ----------- test _find_nearest_index --------------
@pytest.mark.parametrize("data_list, value, expected", [
    ([1, 2, 3], 3, 2),
    (range(10), 3, 2),
    (np.arange(10), 3, 2),
    ([1.1, 2.2, 3.3], 3, 2),
])
def test_threshold__find_nearest_index(ThresholdObj_21, data_list, value, expected):
    assert ThresholdObj_21._find_nearest_index(data_list, value) == expected


@pytest.mark.parametrize("data_list, value", [
    ([1, 2, 3], 3),
    (range(10), 3),
    (np.arange(10), 3),
    ([1.1, 2.2, 3.3], 3),
])
def test_threshold__find_nearest_index(ThresholdObj_21, data_list, value):
    resp_type = type(ThresholdObj_21._find_nearest_index(data_list, value))
    assert resp_type == int or resp_type == np.int64


# ----------- test find_mean_hr_bpm --------------
def test_threshold_find_mean_hr_bpm(ThresholdObj_21, test_21_data):
    assert abs(ThresholdObj_21.find_mean_hr_bpm() - test_21_data["mean_hr_bpm"]) < 5


def test_threshold_find_mean_hr_bpm_output(ThresholdObj_21):
    assert type(ThresholdObj_21.find_mean_hr_bpm()) == float


@pytest.mark.parametrize(
    "time_interval, error", [
        ([.1, .12], TypeError),
        ((.1, .12, .13), ValueError),
        ((".1", .12), TypeError),
        ((.1, 2), ValueError),
        ((.12, .11), ValueError),
        ((0, 1.5), ValueError),
    ])
def test_threshold_find_mean_hr_bpm_inputs_bad(ThresholdObj_21, time_interval, error):
    with pytest.raises(error):
        ThresholdObj_21.find_mean_hr_bpm(time_interval=time_interval)


@pytest.mark.parametrize(
    "time_interval, expected_hr", [
        ((.03, .06), 67),
        ((.13, .17), 75), ])
def test_threshold_find_mean_hr_bpm_inputs_range(ThresholdObj_21, time_interval, expected_hr):
    test_hr = ThresholdObj_21.find_mean_hr_bpm(time_interval=time_interval)
    assert abs(test_hr - expected_hr) < 2


# ----------------- test threshold apply_threshold -----------------
@pytest.mark.parametrize("signal, background, abs_signal, reverse_threshold, error", [
    ([1, 2, 3], (0, 0, 0), True, True, TypeError),
    ((1, 2, 3), [0, 0, 0], "1", True, TypeError),
    ([1, 2, 3], [0, 0, 0], True, "1", TypeError),
])
def test_threshold_apply_threshold_bad_inputs(ThresholdObj_1,
                                              signal, background, abs_signal, reverse_threshold, error):
    with pytest.raises(error):
        ThresholdObj_1.apply_threshold(signal,
                                       background=background,
                                       abs_signal=abs_signal,
                                       reverse_threshold=reverse_threshold)


def test_threshold_apply_threshold_output_type(ThresholdObj_21):
    assert type(ThresholdObj_21.apply_threshold()) == np.ndarray


# ----------------- test threshold _find_threshold -----------------
@pytest.mark.parametrize("signal, background, expected", [
    ([1, 2, 3], [0, 0, 0], (np.array([3, 3, 3]), False)), ])
def test_threshold__find_threshold(ThresholdObj_21, signal, background, expected):
    resp_thresh, is_neg = ThresholdObj_21._find_threshold(signal, background, filter_bg=False)

    assert np.array_equal(resp_thresh, expected[0]) and is_neg == expected[1]


@pytest.mark.parametrize("signal, background", [
    ([1, 2, 3], [0, 0, 0]), ])
def test_threshold__find_threshold_output(ThresholdObj_21, signal, background):
    resp = ThresholdObj_21._find_threshold(signal, background, filter_bg=False)
    assert type(resp) == tuple and \
           type(resp[0]) == list and \
           type(resp[1]) == bool


# ----------------- test threshold _find_num_pm -----------------
@pytest.mark.parametrize("signal, expected", [
    ([-1, -2, 3], (1, 2)),
    ([-1.1, -2.2, 3], (1, 2)),
    (np.array([-1.1, -2.2, 3]), (1, 2)),
])
def test_threshold__find_num_pm(ThresholdObj_21, signal, expected):
    assert ThresholdObj_21._find_num_pm(signal) == expected


@pytest.mark.parametrize("signal, background", [
    ([1, 2, 3], [0, 0, 0]), ])
def test_threshold__find_num_pm(ThresholdObj_21, signal, background):
    resp = ThresholdObj_21._find_num_pm(signal)
    assert type(resp) == tuple and len(resp) == 2 and type(resp[0]) == int and type(resp[1]) == int


# ----------------- test threshold _find_indices -----------------
@pytest.mark.parametrize("values, func, expected", [
    ([0, 1, 0], lambda x: x == 1, [1]),
    ([0, 1, 0, 1], lambda x: x == 1, [1, 3]),
    ([0, 1, 1, 3], lambda x: x == 1, [1, 2]),
    ([0, 1, "1", 3], lambda x: x == 1, [1]),
])
def test_threshold__find_indices(ThresholdObj_21, values, func, expected):
    resp = ThresholdObj_21._find_indices(values, func)
    assert resp == expected


# ----------------- test threshold _find_binary_centers -----------------
@pytest.mark.parametrize("bin_signal, min_width, expected", [
    ([0, 1, 1, 1, 0], 1, [0, 0, 1, 0, 0]),
    ([0, 1, 1, 1, 0, 0, 1, 1, 1, 0], 1, [0, 0, 1, 0, 0, 0, 0, 1, 0, 0]),
])
def test_threshold__find_binary_centers(ThresholdObj_21, bin_signal, min_width, expected):
    resp = ThresholdObj_21._find_binary_centers(bin_signal, min_width)
    assert np.array_equal(resp, expected)


# ----------------- test threshold _find_rising_edges -----------------
@pytest.mark.parametrize("bin_signal, expected", [
    ([0, 1, 1, 1, 0], [1]),
    ([0, 1, 1, 1, 0, 0, 1, 1, 1, 0], [1, 6]),
])
def test_threshold__find_rising_edges(ThresholdObj_21, bin_signal, expected):
    resp = ThresholdObj_21._find_rising_edges(bin_signal)
    assert np.array_equal(resp, expected)


# ----------------- test threshold _find_falling_edges -----------------
@pytest.mark.parametrize("bin_signal, expected", [
    ([0, 1, 1, 1, 0], [3]),
    ([0, 1, 1, 1, 0, 0, 1, 1, 1, 0], [3, 8]),
])
def test_threshold__find_falling_edges(ThresholdObj_21, bin_signal, expected):
    resp = ThresholdObj_21._find_falling_edges(bin_signal)
    assert np.array_equal(resp, expected)


# ----------------- test threshold _confirm_binary -----------------
@pytest.mark.parametrize("signal, expected", [
    ([0, 1, 1, 1, 0], True),
    ([0, 1, 1, 1, 0, 0, 1, 1, 1, 0], True),
    (np.array([0, 1, 1, 1, 0]), True),
    (np.array([0, 1, 1, 1, 0, 0, 1, 1, 1, 0]), True),
    ([0, 1, 3, 1, 0], False),
    ([0, 1, 3, 1, "0"], False),
])
def test_threshold__confirm_binary(ThresholdObj_21, signal, expected):
    resp = ThresholdObj_21._confirm_binary(signal)
    assert resp == expected
