import os
import json
import logging
import numpy as np


class FileHandler(object):
    def __init__(self, filename, **kwargs):
        self.filename = filename
        self.basename = self.get_basename()
        self.folder_path = self.get_folder_path()
        self.time = None
        self.signal = None

        initialize = kwargs.get('initialize', True)
        if initialize:
            self.read_data()

    def get_folder_path(self, filename: str = None) -> object:
        """
        Gets file path without file from project root directory.
        Args:
            filename: Name of the file

        Returns:
            str:  Folder path of the file
        """
        if not filename and not self.filename:
            raise FileNotFoundError("No file provided.")
        if not filename:
            filename = self.filename

        current_directory = os.getcwd()
        folder_path = os.path.dirname(os.path.abspath(filename))
        return folder_path.replace(current_directory, "").replace("\\", "/")

    def get_basename(self, filename: str = None):
        """
        Gets filename without extension.
        Args:
            filename: Name of the file

        Returns:
            str:  Filename without extension
        """
        if not filename and not self.filename:
            raise FileNotFoundError("No file provided.")
        if not filename:
            filename = self.filename

        base = os.path.basename(filename)
        return str(os.path.splitext(base)[0])

    def get_ext(self, filename: str = None):
        """
        Gets file extension
        Args:
            filename: Name of the file

        Returns: Extension of the file.

        """
        if not filename and not self.filename:
            raise FileNotFoundError("No file provided.")
        if not filename:
            filename = self.filename

        _, file_extension = os.path.splitext(filename)
        return file_extension

    def read_data(self, filename: str = None):
        """
        Reads in data from specified file.

        Args:
            filename: File to read from
        """
        if not filename and not self.filename:
            raise FileNotFoundError("No file provided.")
        if not filename:
            filename = self.filename

        exists = os.path.isfile(filename)
        if exists:
            if self.get_ext(filename) != ".csv":
                raise TypeError("Incorrect file type.")

            verified_data = self._verify_data(filename)
            self.time = verified_data[:, 0]
            self.signal = verified_data[:, 1]
            return self.time, self.signal
        else:
            raise FileNotFoundError("That file does not exist.")

    def _verify_data(self, filename):
        """
        Verifies data by ensuring dimensions and data is numeric
        Args:
            filename: Name of the file to verify data for

        Returns:
            numpy.array: data from given file
        """
        try:
            # as numpy array
            csv_data = np.genfromtxt(filename, delimiter=",")
            csv_data = self._verify_dimensions(csv_data)
            csv_data = self._verify_values(csv_data)

        except ValueError:
            print("Something wrong with file read.")
            return None
        return csv_data

    def _verify_values(self, data):
        """
        Gets rid of elements that are NaN.
        Args:
            data: Nx2 data numpy array

        Returns: Matrix without any NaN

        """
        data = np.array(data)
        nan_indices = ~np.isnan(data).any(axis=1)
        return data[nan_indices]

    def _verify_dimensions(self, data):
        """
        Determines if the data is in the correct dimension.
        Args:
            data: Data from the file as numpy array.

        Returns:
            data: Data in the correct orientation
        """
        data = np.array(data)
        if len(data.shape) != 2:
            raise ValueError("Data dimensions is not correct")
        elif data.shape[0] != 2 and data.shape[1] != 2:
            raise ValueError("Data must have 2 columns.")

        # correct if not
        if data.shape[0] > 2 and data.shape[1] == 2:
            return data
        else:
            return data.transpose()

    def write_data(self, json_object, filename):
        """
        Writes json object to specified filename.
        Args:
            json_object: dictionary object with data
            filename: name of file to save to
        """
        if type(json_object) is dict:
            with open(filename, 'w') as outfile:
                json.dump(json_object, outfile)
        else:
            raise TypeError("Data type must be dict.")
