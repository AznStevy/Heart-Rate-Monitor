import os
import json
import pytest
import numpy as np
from file_handler import FileHandler


@pytest.mark.parametrize("filename, expected", [
    ("test1.json", ""),
    ("test/test1.json", "/test"),
    ("test_folder/test_sub_folder/test1.csv", "/test_folder/test_sub_folder")])
def test_get_folder_path(filename, expected):
    try:
        ecg_file = FileHandler(filename, initialize=False)
        assert ecg_file.get_folder_path() == expected
    except FileNotFoundError:
        assert True

@pytest.mark.parametrize("filename, expected", [
    ("test1.json", "test1"),
    ("test/test1.json", "test1"),
    ("test_folder/test_sub_folder/test1.csv", "test1")])
def test_get_basename(filename, expected):
    ecg_file = FileHandler(filename, initialize=False)
    assert ecg_file.get_basename() == expected

@pytest.mark.parametrize("filename", [
    ("tests/test1.csv"),
    ("test/thisdoesntexist.csv"),
    ("tests/test1.dat")])
def test_read_data(filename):
    ecg_file = FileHandler(filename, initialize=False)
    try:
        time, signal = ecg_file.read_data()
        assert len(time) == len(signal)
    except FileNotFoundError:
        assert True
