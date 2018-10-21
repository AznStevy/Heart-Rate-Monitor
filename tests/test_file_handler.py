import json
import pytest
from file_handler import FileHandler


@pytest.mark.parametrize("filename, expected", [
    ("test1.json", ""),
    ("tests/test1.csv", "/tests"),
    ("tests/test_data/test_data1.csv", "/tests/test_data")])
def test_get_folder_path(filename, expected):
    ecg_file = FileHandler(filename, initialize=False)
    assert ecg_file.get_folder_path() == expected


@pytest.mark.parametrize("filename, expected", [
    ("test1.json", "test1"),
    ("tests/test1.json", "test1"),
    ("tests/test_data/test_data1.csv", "test_data1")])
def test_get_basename(filename, expected):
    ecg_file = FileHandler("tests/test_data/test1.csv", initialize=False)
    assert ecg_file.get_basename(filename) == expected


@pytest.mark.parametrize("filename, expected", [
    ("test1.json", ".json"),
    ("tests/test1.json", ".json"),
    ("tests/test_data/test_data1.csv", ".csv")])
def test_get_ext(filename, expected):
    ecg_file = FileHandler(filename, initialize=False)
    assert ecg_file.get_ext() == expected


def _get_test_data_files():
    files = []
    for i in range(31):
        filename = "tests/test_data/test_data{}.csv".format(i + 1)
        files.append(filename)
    return files


@pytest.mark.parametrize("filename", _get_test_data_files())
def test_read_data(filename):
    ecg_file = FileHandler(filename, initialize=False)
    time, signal = ecg_file.read_data()
    assert len(time) == len(signal)


@pytest.mark.parametrize("filename", _get_test_data_files())
def test_read_bad_data(filename):
    ecg_file = FileHandler(filename, initialize=False)
    time, signal = ecg_file.read_data()
    assert len(time) == len(signal)


@pytest.mark.parametrize("filename", [
    ("tests/thisdoesntexist.csv"),
    ("tests/test1.dat")])
def test_read_data_no_file(filename):
    ecg_file = FileHandler(filename, initialize=False)
    with pytest.raises(FileNotFoundError):
        _, _ = ecg_file.read_data()


@pytest.mark.parametrize("filename", [
    ("tests/test_data/test_data1.html")])
def test_read_data_bad_ext(filename):
    ecg_file = FileHandler(filename, initialize=False)
    with pytest.raises(TypeError):
        _, _ = ecg_file.read_data()


@pytest.mark.parametrize("data", [
    ({}),
    ({"name": "Bob"}),
    ({"_id": "42", "favorite_nums": [2, 3, 4]})])
def test_write_data(data):
    test_file_path = "tests/test_data/test_json.json"
    ecg_file = FileHandler("/test_data/test1.csv", initialize=False)
    ecg_file.write_data(data, test_file_path)

    with open(test_file_path) as json_data:
        file_data = json.load(json_data)
        assert file_data == data


@pytest.mark.parametrize("data", [
    (str("This is a string"))])
def test_write_bad_data(data):
    test_file_path = "tests/test_data/test_json.json"
    ecg_file = FileHandler("/test_data/test1.csv", initialize=False)
    with pytest.raises(TypeError):
        ecg_file.write_data(data, test_file_path)
