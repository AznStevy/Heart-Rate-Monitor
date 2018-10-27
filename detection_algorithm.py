from filtered_signal import FilteredSignal

import logging
import numpy as np
import scipy.signal as sp
from abc import abstractmethod
import matplotlib.pyplot as plt

logging.basicConfig(filename='heart_rate_monitor.log', level=logging.DEBUG)


class ECGDetectionAlgorithm(object):
    def __init__(self, time, signal, **kwargs):
        self.name = kwargs.get('name', "None")
        if type(time) != list and type(time) != np.ndarray:
            raise TypeError("time must be numpy.ndarray.")
        self.time = np.array(time)
        if type(signal) != list and type(signal) != np.ndarray:
            raise TypeError("signal must be numpy.ndarray.")
        self.raw_signal = np.array(signal)
        if len(self.raw_signal) == 0 or len(self.time) == 0:
            raise ValueError("signal and time must contain elements.")
        if len(self.raw_signal) != len(self.time):
            raise ValueError("signal and time must be same length.")

        # define properties
        self.beats = None
        self.duration = None
        self.num_beats = None
        self.mean_hr_bpm = None
        self.voltage_extremes = None

    def start_analysis(self, time_interval=None):
        """
        Begins the analysis to get necessary return parameters

        Args:
            time_interval(tuple): Time interval to analyze for mean_hr_bpm in minutes.
        """
        logging.info("Beginning heartbeat analysis.")
        self.duration = self.find_duration()
        self.voltage_extremes = self.find_voltage_extremes()

        self.beats = self.find_beats()
        self.num_beats = self.find_num_beats()
        try:
            self.mean_hr_bpm = self.find_mean_hr_bpm(time_interval=time_interval)
        except ValueError as e:
            self.mean_hr_bpm = None
            logging.exception(e)
        except TypeError as e:
            self.mean_hr_bpm = None
            logging.exception(e)
        logging.info("Heartbeat analysis completed.")

    def find_voltage_extremes(self) -> tuple:
        """
        Finds the voltage extremes from the original signal.

        Returns:
            tuple: Returns min and max as floats of raw signal in form (min, max).

        """
        logging.info("find_voltage_extremes called")
        try:
            return self._find_voltage_extremes(self.raw_signal)
        except TypeError as e:
            logging.exception(e)

    def _find_voltage_extremes(self, signal) -> tuple:
        """
        Finds the voltage extremes from a given signal.
        Args:
            signal (numpy.ndarray): Finds extremes of that signal

        Returns:
            tuple: Minimum and maximum values of the signal.

        """
        if type(signal) != list and type(signal) != np.ndarray:
            raise TypeError("signal must be numpy.ndarray.")

        signal = np.array(signal)
        min_sig = np.min(signal)
        max_sig = np.max(signal)
        if max_sig >= .3:
            logging.warning("Voltage is too high to be biologically relevant.")
        return min_sig, max_sig

    def find_duration(self) -> float:
        """
        Finds total duration of the signal.

        Returns:
            float: Duration in seconds.

        """
        logging.info("find_duration called")
        duration = float(self.time[-1]) - float((self.time[0]))  # in seconds
        return duration

    def find_num_beats(self) -> int:
        """
        Finds number of beats in the signal.

        Returns:
            int: number of beats

        """
        logging.info("find_num_beats called")
        if self.beats is None:
            self.beats = self.find_beats()
        return len(self.beats)

    def _find_nearest_index(self, array, value) -> int:
        """
        Finds the index which is nearest to the value.
        Args:
            array (numpy.ndarray): The array in which to find the value
            value (float): Value in question.

        Returns:
            int: Nearest index to the value.

        """
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    @abstractmethod
    def find_mean_hr_bpm(self, time_interval=None) -> float:
        """
        Abstract method that finds mean hr bpm which must be implemented.
        Args:
            time_interval (tuple): time interval to analyze in minutes.

        Returns:
            float: The mean hr bpm in that minute time-frame.

        """
        pass

    @abstractmethod
    def find_beats(self) -> np.ndarray:
        """
        Abstract method that finds beat times which must be implemented.
        Returns:
            numpy.ndarray: Beat times.

        """
        pass

    @abstractmethod
    def plot_graph(self):
        """
        Abstract method that provides visualization of the analysis which must be implemented.
        """
        pass


class Threshold(ECGDetectionAlgorithm):
    def __init__(self, time, signal, **kwargs):
        # confirms they are good inputs
        super().__init__(time, signal, **kwargs)

        self.high_cutoff = kwargs.get('high_pass_cutoff', 1)
        if type(self.high_cutoff) != int:
            raise TypeError("high_cutoff must be type int.")
        self.low_cutoff = kwargs.get('low_pass_cutoff', 30)
        if type(self.low_cutoff) != int:
            raise TypeError("low_cutoff must be type int.")

        self.threshold_frac = kwargs.get('threshold_frac', 1)
        if type(self.threshold_frac) != float and type(self.threshold_frac) != int:
            raise TypeError("threshold_frac must be type int.")
        elif self.threshold_frac > 1 or self.threshold_frac < 0:
            raise ValueError("threshold_frac must be between [0,1].")

        # this does not need to be here for this class... will iron this out later
        try:
            self.filtered_signal_obj = FilteredSignal(
                time=self.time, signal=self.raw_signal,
                high_pass_cutoff=self.high_cutoff, low_pass_cutoff=self.low_cutoff)
            self.filtered_signal = self.filtered_signal_obj.filtered_signal
            self.background = self.filtered_signal_obj.bg_sub_signal
            self.fs = self.filtered_signal_obj.fs
        except ValueError or TypeError as e:
            logging.exception(e)
            self.filtered_signal_obj = None
            self.filtered_signal = self.raw_signal
            self.background = np.zeros(len(self.raw_signal))
            self.fs = 0
        # print(self.filtered_signal_obj.get_properties())

        # processing parameters
        self.threshold = None
        self.binary_signal = None
        self.binary_centers = None
        self.rising_edges = None
        self.falling_edges = None
        self.signal_period = None

    def find_beats(self) -> np.ndarray:
        """
        Finds the beats from the signal.

        Returns:
            numpy.ndarray: Times at which the beats occur.

        """
        self.binary_signal = self.apply_threshold(self.filtered_signal, self.background)
        self.binary_centers = self._find_binary_centers(self.binary_signal)

        # find the indices where it equals 1
        beat_ind = self._find_indices(self.binary_centers, lambda x: x == 1)

        if not self.duration:
            self.duration = self.find_duration()

        test_bpm = len(beat_ind) / (self.duration / 60)
        if test_bpm < 40:  # reasonable, but still abnormal bpm
            binary_signal_rev = self.apply_threshold(
                self.filtered_signal, self.background, reverse_threshold=True)

            binary_centers_rev = self._find_binary_centers(binary_signal_rev)
            beat_ind_rev = self._find_indices(binary_centers_rev, lambda x: x == 1)
            test_bpm_rev = len(beat_ind_rev) / (self.duration / 60)

            if test_bpm_rev >= 40:
                self.binary_signal = binary_signal_rev
                self.binary_centers = binary_centers_rev
                beat_ind = beat_ind_rev

        beat_time_list = np.take(self.time, tuple(beat_ind))
        return np.array(beat_time_list)

    def _find_indices(self, values, func) -> list:
        """
        Finds indices of an array given parameters.
        Args:
            values (numpy.ndarray): list of values
            func: lambda function

        Returns:
            list: list of indices that fit the lambda function.

        """
        return [i for (i, val) in enumerate(values) if func(val)]

    def apply_threshold(self, signal=None, background=None,
                        abs_signal=False, reverse_threshold=False) -> np.ndarray:
        """
        Applies a threshold of a certain percentage.
        Args:
            reverse_threshold (bool): Reverse threshold from what it should be.
            background (numpy.ndarray): Supply a background signal to consider.
            abs_signal (bool): Whether or not to threshold with absolute values.
            signal (numpy.ndarray): Filtered signal in numpy array.

        Returns:
            numpy.ndarray: list of binary values based on threshold.

        """
        logging.info("THRESHOLD apply_threshold called")

        if signal is None:
            signal = self.raw_signal
        if type(signal) != np.ndarray:
            raise TypeError("signal must be type numpy.ndarray")
        if type(background) != np.ndarray and background is not None:
            raise TypeError("background must be type numpy.ndarray")
        if type(abs_signal) != bool:
            raise TypeError("abs_signal must be type bool")
        if type(reverse_threshold) != bool:
            raise TypeError("reverse_threshold must be type bool")

        self.threshold, is_negative = self._find_threshold(signal, background,
                                                           reverse_threshold=reverse_threshold)
        if abs_signal:
            signal = abs(signal)

        if is_negative:
            bin_sig = signal <= self.threshold
        else:
            bin_sig = signal >= self.threshold
        return bin_sig

    def _find_threshold(self, signal, background=None,
                        filter_bg: bool = True, reverse_threshold=False) -> tuple:
        """
        Determines threshold based on a absolute-value-filtered/zeroed signal and proportion.
        Threshold is padded by one period. Note: abs value isn't used because of double/triple counting.
        Args:
            reverse_threshold (bool): Reverse threshold of what it should be in terms of positive or negative.
            filter_bg (bool): Whether or not to filter the background.
            background (numpy.ndarray): background for the signal.
            signal (numpy.ndarray): Heart beat signal.

        Returns:
            tuple: First is a numpy.ndarray threshold array and second is bool if threshold is negative.

        """
        if filter_bg and background is not None:
            background = self.filtered_signal_obj.apply_noise_reduction(
                background, self.low_cutoff + 10, max(0, self.high_cutoff - 5))

        try:
            padding = self.filtered_signal_obj.period
            if padding:
                start_ind = padding
                end_ind = len(self.filtered_signal) - padding
                padded_signal = signal[start_ind:end_ind]
                min_v, max_v = self._find_voltage_extremes(padded_signal)
            else:
                min_v, max_v = self._find_voltage_extremes(signal)
        except ValueError as e:
            logging.exception(e)
            min_v, max_v = self._find_voltage_extremes(signal)

        if reverse_threshold:
            if abs(min_v) < abs(max_v):
                is_negative = True
                threshold_value = min_v * self.threshold_frac
            else:
                is_negative = False
                threshold_value = max_v * self.threshold_frac
        else:
            if abs(min_v) > abs(max_v):
                is_negative = True
                threshold_value = min_v * self.threshold_frac
            else:
                is_negative = False
                threshold_value = max_v * self.threshold_frac

        # determine if spikes tend to be positive or negative
        threshold_array = []
        if background is None:
            threshold_array = np.ones(len(self.filtered_signal)) * threshold_value
        else:
            for bg_val in background:
                threshold_array.append(threshold_value - bg_val)

        return threshold_array, is_negative

    def _find_num_pm(self, signal) -> tuple:
        """
        Finds the number of values above and below axis.
        Args:
            signal: Signal in question.

        Returns:
            tuple: First is number of positive, second is number of negative elements.

        """
        signal = np.array(signal)
        # strictly above or below 0
        pos = signal[np.where(signal > 0)]
        neg = signal[np.where(signal < 0)]
        return len(pos), len(neg)

    def find_mean_hr_bpm(self, time_interval=None) -> float:
        """
        Finds the mean heart rate beats per minute for signal.
        Args:
            time_interval (tuple): Interval in minutes of the signal to find mean hr bpm.

        Returns:
            float: mean heart rate bpm within the designated time interval.

        """
        logging.info("find_mean_hr_bpm called")

        # get necessary information if not already calculated
        if self.duration is None:
            self.duration = self.find_duration()
        if self.beats is None:
            self.beats = self.find_beats()

        if time_interval is None:
            time_interval = (min(self.time) / 60, max(self.time) / 60)
        elif type(time_interval) != tuple:
            raise TypeError("interval must be type tuple.")
        elif len(time_interval) != 2:
            raise ValueError("interval tuple must have two elements.")
        elif (type(time_interval[0]) != float and type(time_interval[0]) != int) or \
                (type(time_interval[1]) != float and type(time_interval[1]) != int):
            raise TypeError("tuple elements must be type float or int.")
        elif time_interval[0] * 60 < min(self.time) or \
                                time_interval[1] * 60 > max(self.time):
            raise ValueError("interval tuple must have proper range.")
        elif time_interval[0] >= time_interval[1]:
            raise ValueError("interval tuple must have proper range.")
        elif (time_interval[1] - time_interval[0]) * 60 > self.duration:
            # check if they are within range
            raise ValueError("interval must be less than signal duration.")

        # find proper signal and time intervals
        duration_oi_sec = (time_interval[0] * 60, time_interval[1] * 60)
        duration_indices = (self._find_nearest_index(self.time, duration_oi_sec[0]),
                            self._find_nearest_index(self.time, duration_oi_sec[1]))
        duration_indices = np.array(duration_indices)

        if self.binary_centers is None:
            self.find_beats()

        bin_center_oi = self.binary_centers[duration_indices[0]: duration_indices[1]]
        num_beats_oi = self._find_indices(bin_center_oi, lambda x: x == 1)

        duration_oi = time_interval[1] - time_interval[0]  # minutes

        bpm = float(len(num_beats_oi) / float(duration_oi))
        return bpm

    def plot_graph(self, file_path: str = None):
        """
        Plots a graph of thresholding and frequency information for the threshold algorithm.
        Args:
            file_path: The path of the file to output.

        """
        logging.info("THRESHOLD plot_graph called")
        fig = plt.figure(figsize=(10, 6))
        plt.title("{}".format(self.name))
        plt.rcParams['text.antialiased'] = True
        plt.style.use('ggplot')
        ax1 = fig.add_subplot(211)
        ax1.grid(True)
        ax1.plot(self.time, self.raw_signal,
                 label='Raw Signal', linewidth=1, antialiased=True)
        ax1.plot(self.time, self.filtered_signal,
                 label='Filtered Signal', linewidth=1, antialiased=True)
        ax1.plot(self.time, np.ones(len(self.time)) * self.threshold,
                 label='Threshold', linewidth=1, antialiased=True)

        # scale the signals
        _, max_val = self._find_voltage_extremes(self.filtered_signal)
        ax1.plot(self.time, self.binary_signal * max_val,
                 label='Binary Signal', linewidth=5, antialiased=True)
        ax1.plot(self.time, self.binary_centers * max_val,
                 label='Binary Centers', linewidth=5, antialiased=True)
        ax1.legend(loc='best')

        ax2 = fig.add_subplot(212)
        freq_raw, fft_out_raw = self.filtered_signal_obj.get_fft(is_filtered=False)
        ax2.plot(freq_raw, abs(fft_out_raw),
                 label='Raw Signal', linewidth=1)  # plotting the spectrum
        freq_filtered, fft_out_filtered = self.filtered_signal_obj.get_fft(is_filtered=True)
        ax2.plot(freq_filtered, abs(fft_out_filtered),
                 label='Filtered Signal', linewidth=1)  # plotting the spectrum
        ax2.set_xlabel('Freq (Hz)')
        ax2.set_ylabel('|Y(freq)|')
        ax2.legend(loc='best')
        fig.tight_layout()
        if file_path:
            fig.savefig(file_path)
        plt.show()
        plt.close()

    def _find_binary_centers(self, bin_signal, min_width: int = 1) -> np.ndarray:
        # first make sure that this is a binary signal
        """
        Finds the centers of the thresholded binary signal.
        Args:
            min_width (int): Minimum width for binary signal.
            bin_signal (numpy.ndarray): binary signal

        Returns:
            numpy.ndarray: List of binary values representing the centers of the binary steps.

        """
        if min_width < 1:
            raise ValueError("min_width must be int greater than 0.")

        try:
            self.rising_edges = self._find_rising_edges(bin_signal)
            self.falling_edges = self._find_falling_edges(bin_signal)
        except ValueError as e:
            logging.exception(e)
            return np.zeros(len(bin_signal))

        # puts falling edge at end if there's a incomplete peak at end (test_data1)
        if len(self.rising_edges) > len(self.falling_edges):
            temp_falling_edges = self.falling_edges.tolist()
            temp_falling_edges.append(len(bin_signal))
            self.falling_edges = np.array(temp_falling_edges)

        max_len = min(len(self.rising_edges), len(self.falling_edges))

        centers = []  # gets the centers only
        for i in range(max_len):
            if (self.falling_edges[i] - self.rising_edges[i]) >= min_width:
                centers.append(round((self.rising_edges[i] + self.falling_edges[i]) / 2))

        # generate actual binary for centers
        ecg_center_peaks = []
        for i in range(len(bin_signal)):
            if i in centers:
                ecg_center_peaks.append(1)
            else:
                ecg_center_peaks.append(0)
        return np.array(ecg_center_peaks)

    def _find_rising_edges(self, bin_signal) -> np.ndarray:
        """
        Finds the rising edge of a binary signal.
        Args:
            bin_signal (numpy.ndarray): binary signal

        Returns:
            numpy.array: Indices at which a rising edge occurs.

        """
        is_binary = self._confirm_binary(bin_signal)
        if not is_binary:
            raise ValueError("Signal is not binary")

        rising_edges = []
        previous_val = 0
        for i, val in enumerate(bin_signal):
            if i == 0:
                if val == 0:
                    previous_val = 0
                elif val == 1:
                    previous_val = 1
            elif previous_val == 1 and val == 1:
                previous_val = 1
            elif previous_val == 0 and val == 1:
                previous_val = 1
                rising_edges.append(i)
            elif val == 0:
                previous_val = 0

        return np.array(rising_edges)

    def _find_falling_edges(self, bin_signal) -> np.ndarray:
        """
        Finds the falling edge of a binary signal.
        Args:
            bin_signal (numpy.ndarray): binary signal

        Returns:
            numpy.array: Indices at which a falling edge occurs.

        """
        is_binary = self._confirm_binary(bin_signal)
        if not is_binary:
            raise ValueError("Signal is not binary")

        falling_edges = []
        previous_val = 0
        for i, val in enumerate(bin_signal):
            if i == 0:
                if val == 0:
                    previous_val = 0
                elif val == 1:
                    previous_val = 1
            elif previous_val == 1 and val == 0:
                previous_val = 0
                falling_edges.append(i - 1)
            elif val == 1:
                previous_val = 1

        return np.array(falling_edges)

    def _confirm_binary(self, signal) -> bool:
        """
        Tests of the signal is a binary signal of 0s and 1s
        Args:
            signal (numpy.ndarray): signal to test

        Returns:
            bool: Whether or not signal is a binary signal

        """
        signal = np.array(signal)
        return np.array_equal(signal, signal.astype(bool))


class Wavelet(Threshold):
    def __init__(self, time, signal, **kwargs):
        super().__init__(time, signal, **kwargs)

        self.signal_cwt = None
        self.threshold_frac = kwargs.get('threshold_frac', .5)

    def find_beats(self, reverse_threshold: bool = False) -> np.ndarray:
        """
        Finds the beats from the signal using a continuous wavelet transform.
        Args:
            reverse_threshold (bool): Whether or not to reverse the automatic threshold.

        Returns:
            numpy.ndarray: Times at which the beats occur.
        """

        if type(reverse_threshold) != bool:
            raise TypeError("reverse_threshold must be type bool.")

        self.signal_cwt = self._wavelet_transform()
        self.binary_signal = self.apply_threshold(self.signal_cwt, self.signal_cwt)
        self.binary_centers = self._find_binary_centers(self.binary_signal)

        # find the indices where it equals 1
        beat_ind = self._find_indices(self.binary_centers, lambda x: x == 1)

        if not self.duration:
            self.duration = self.find_duration()

        test_bpm = len(beat_ind) / (self.duration / 60)
        if test_bpm < 40:  # reasonable, but still abnormal bpm
            binary_signal_rev = self.apply_threshold(
                self.signal_cwt, self.signal_cwt, reverse_threshold=True)

            binary_centers_rev = self._find_binary_centers(binary_signal_rev)
            beat_ind_rev = self._find_indices(binary_centers_rev, lambda x: x == 1)
            test_bpm_rev = len(beat_ind_rev) / (self.duration / 60)

            if test_bpm_rev >= 40:
                self.binary_signal = binary_signal_rev
                self.binary_centers = binary_centers_rev
                beat_ind = beat_ind_rev

        beat_time_list = np.take(self.time, tuple(beat_ind))
        return np.array(beat_time_list)

    def _wavelet_transform(self) -> np.ndarray:
        # limit to the average detected period the signal
        """
        Takes a wavelet transform of the object's signal and then averages along the width dimension.
        Returns:
            numpy.ndarray: Averaged wavelet transform.

        """
        self.widths_cwt = np.arange(1, 6)
        self.signal_cwt_img = sp.cwt(self.raw_signal, sp.ricker, self.widths_cwt)
        return np.average(self.signal_cwt_img, axis=0)

    def plot_graph(self, file_path: str = None):
        """
        Plots a graph of relevant information for the wavelet algorithm.
        Args:
            file_path (str): The path of the file to output.

        """

        fig = plt.figure(figsize=(10, 6))
        plt.title("{}".format(self.name))
        plt.rcParams['text.antialiased'] = True
        plt.style.use('ggplot')
        ax1 = fig.add_subplot(211)
        ax1.grid(True)
        ax1.set_xlim([min(self.time), max(self.time)])
        ax1.plot(self.time, self.raw_signal,
                 label='Raw Signal', linewidth=1, antialiased=True)
        ax1.plot(self.time, self.threshold,
                 label='Threshold', linewidth=1, antialiased=True)
        ax1.plot(self.time, self.signal_cwt,
                 label='Averaged Wavelet Transform', linewidth=1, antialiased=True)
        _, max_val = self._find_voltage_extremes(self.filtered_signal)
        ax1.plot(self.time, self.binary_signal * max_val,
                 label='Binary Signal', linewidth=5, antialiased=True)
        ax1.plot(self.time, self.binary_centers * max_val,
                 label='Binary Centers', linewidth=5, antialiased=True)
        ax1.legend(loc='best')

        ax2 = fig.add_subplot(212)
        ax2.imshow(self.signal_cwt_img, cmap='magma', aspect='auto',
                   vmax=abs(self.signal_cwt).max(), vmin=-abs(self.signal_cwt).max())
        plt.show()
        plt.close()
