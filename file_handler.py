import os
import json
import logging
import numpy as np

logging.basicConfig(filename='heart_rate_monitor.log', level=logging.DEBUG)


class FileHandler(object):
    def __init__(self, filename, **kwargs):
        self.time = None
        self.signal = None
        self.filename = filename
        self.basename = self.get_basename()
        self.folder_path = self.get_folder_path()
        initialize = kwargs.get('initialize', True)
        if initialize:
            self.read_data()

    def get_folder_path(self, filename: str = None) -> str:
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

    def get_basename(self, filename: str = None) -> str:
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

    def get_ext(self, filename: str = None) -> str:
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

    def read_data(self, filename: str = None) -> tuple:
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
            if verified_data is None:
                raise ValueError("File could not be read.")
            self.time = verified_data[:, 0]
            self.signal = verified_data[:, 1]
            return self.time, self.signal
        else:
            raise FileNotFoundError("That file does not exist.")

    def _verify_data(self, filename) -> np.ndarray:
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
        except ValueError as e:
            logging.exception(e)
            return None

        return csv_data

    def _verify_values(self, data) -> np.ndarray:
        """
        Gets rid of elements that are NaN.
        Args:
            data: Nx2 data numpy array

        Returns: Matrix without any NaN

        """
        og_data = np.array(data)
        nan_indices = ~np.isnan(og_data).any(axis=1)
        fixed_data = data[nan_indices]

        if fixed_data.size != og_data.size:
            logging.warning("NaN elements from numpy.ndarray removed.")
        return fixed_data

    def _verify_dimensions(self, data) -> np.ndarray:
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
        Writes json object to specified filename. Converts numpy arrays to lists.
        Args:
            json_object: dictionary object with data
            filename: name of file to save to
        """
        if type(json_object) != dict:
            raise TypeError("Data type must be dict.")

        # convert to list if np array
        for attr in json_object.keys():
            if type(json_object[attr]) == np.ndarray:
                json_object[attr] = json_object[attr].tolist()

        with open(filename, 'w') as outfile:
            json.dump(json_object, outfile)
