import pytest
import numpy as np

from file_handler import FileHandler
from filtered_signal import FilteredSignal


# useful, keep in mind
# https://github.com/pytest-dev/pytest/issues/349

# test variables
@pytest.fixture()
def filtered_signal_obj():
    filename = "tests/test_data/test_data{}.csv".format(1)
    file = FileHandler(filename)
    filtered_signal_1 = FilteredSignal(file.time, file.signal)
    return filtered_signal_1


@pytest.fixture()
def test_variables():
    time_list = np.linspace(0, 10, 300)
    signal_line = np.ones(len(time_list))
    freq = 1  # hz
    fs = 300  # sampling freq

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
    3.4,
])
def test_constructor_with_bad_high_pass(test_variables, high_pass_cutoff):
    with pytest.raises(TypeError):
        time_list = test_variables["time"]
        signal_line = test_variables["line"]
        _ = FilteredSignal(time_list, signal_line,
                           high_pass_cutoff=high_pass_cutoff)


@pytest.mark.parametrize("high_pass_cutoff, low_pass_cutoff", [
    (3, 50),
    (5, 10),
])
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
    (10, 5),
])
def test_constructor_with_bad_high_low_pass(
        test_variables, high_pass_cutoff, low_pass_cutoff):
    with pytest.raises(ValueError):
        time_list = test_variables["time"]
        signal_line = test_variables["line"]
        _ = FilteredSignal(time_list, signal_line,
                           high_pass_cutoff=high_pass_cutoff,
                           low_pass_cutoff=low_pass_cutoff)


# --------------- check time -----------------------
@pytest.mark.parametrize("time_list", [
    (["1", "2", "3"]),
    (np.array(["1", "2", "3"]))
])
def test_constructor_bad_time_list_values(test_variables, time_list):
    with pytest.raises(ValueError):
        signal_line = test_variables["line"]
        _ = FilteredSignal(time_list, signal_line)


@pytest.mark.parametrize("time_list", [
    ("1", "2", "3"),
])
def test_constructor_bad_time_list_type(test_variables, time_list):
    with pytest.raises(TypeError):
        signal_line = test_variables["line"]
        _ = FilteredSignal(time_list, signal_line)
        assert True
