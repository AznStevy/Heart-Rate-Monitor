import json
import pytest
from file_handler import FileHandler


@pytest.mark.parametrize("filename, expected", [
    ("test1.json", ""),
    ("tests/test1.csv", "/tests"),
    ("test_folder/test_sub_folder/test1.csv", "/test_folder/test_sub_folder")])
def test_get_folder_path(filename, expected):
    try:
        ecg_file = FileHandler(filename, initialize=False)
        assert ecg_file.get_folder_path() == expected
    except FileNotFoundError:
        assert True


@pytest.mark.parametrize("filename, expected", [
    ("test1.json", "test1"),
    ("tests/test1.json", "test1"),
    ("tests/test_data/test1.csv", "test1")])
def test_get_basename(filename, expected):
    ecg_file = FileHandler("tests/test_data/test1.csv", initialize=False)
    assert ecg_file.get_basename(filename) == expected


@pytest.mark.parametrize("filename, expected", [
    ("test1.json", ".json"),
    ("tests/test1.json", ".json"),
    ("test_folder/test_sub_folder/test1.csv", ".csv")])
def test_get_ext(filename, expected):
    ecg_file = FileHandler(filename, initialize=False)
    assert ecg_file.get_ext() == expected


@pytest.mark.parametrize("filename", [
    ("tests/test1.csv"),
    ("tests/thisdoesntexist.csv"),
    ("tests/test1.dat")])
def test_read_data(filename):
    ecg_file = FileHandler(filename, initialize=False)
    try:
        time, signal = ecg_file.read_data()
        assert len(time) == len(signal)
    except FileNotFoundError:
        assert True


@pytest.mark.parametrize("data", [
    ({}),
    ({"name": "Bob"}),
    ({"_id": "42", "favorite_nums": [2, 3, 4]}),
    (str("This is a string"))])
def test_write_data(data):
    test_file_path = "tests/test_data/test_json.json"
    ecg_file = FileHandler("/test_data/test1.csv", initialize=False)
    try:
        ecg_file.write_data(data, test_file_path)
    except TypeError:
        assert True

    with open(test_file_path) as json_data:
        file_data = json.load(json_data)
        assert file_data == data
