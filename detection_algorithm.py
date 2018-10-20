from filtered_signal import FilteredSignal

import numpy as np
from abc import abstractmethod
import matplotlib.pyplot as plt


class ECGDetectionAlgorithm(object):
    def __init__(self, time, signal, **kwargs):
        self.name = kwargs.get('name', "None")
        self.time = time
        self.raw_signal = signal

        # define properties
        self.beats = []
        self.duration = 0
        self.num_beats = 0
        self.mean_hr_bpm = 0
        self.voltage_extremes = (0, 0)

    def start_analysis(self):
        """
        Begins the analysis to get necessary return parameters
        """
        self.beats = self.find_beats()
        self.duration = self.find_duration()
        self.num_beats = self.find_num_beats()
        self.mean_hr_bpm = self.find_mean_hr_bpm()
        self.voltage_extremes = self.find_voltage_extremes()

    def find_voltage_extremes(self):
        """
        Finds the voltage extremes from the original signal.

        """
        return self._find_voltage_extremes(self.raw_signal)

    def _find_voltage_extremes(self, signal):
        """
        Finds the voltage extremes from a given signal.
        Args:
            signal: Finds extremes of that signal

        Returns: Minimum and maximum values of the signal

        """
        signal = np.array(signal)
        min_sig = np.min(signal)
        max_sig = np.max(signal)
        return min_sig, max_sig

    def find_duration(self):
        """
        Finds total duration of the signal.
        Returns: Duration in seconds.

        """
        duration = float(self.time[-1]) - float((self.time[0]))  # in seconds
        return duration

    def find_num_beats(self):
        """
        Finds number of beats in the signal.
        Returns: number of beats

        """
        if not self.beats:
            self.find_beats()
        return len(self.beats)

    def find_mean_hr_bpm(self):
        """
        Finds the mean heartrate beats per minute for signal.
        Returns: mean heartrate bpm.

        """
        if not self.duration:
            self.duration = self.find_duration()
        if not self.beats:
            self.beats = self.find_beats()

        bpm = float(len(self.beats) / float(self.duration / 60))
        return bpm

    @abstractmethod
    def find_beats(self):
        pass

    @abstractmethod
    def plot_graph(self):
        pass


class Threshold(ECGDetectionAlgorithm):
    def __init__(self, time, signal, **kwargs):
        super().__init__(time, signal)
        self.high_cutoff = kwargs.get('high_cutoff', 1)
        self.low_cutoff = kwargs.get('low_cutoff', 30)

        self.filtered_signal_obj = FilteredSignal(
            time=self.time, signal=self.raw_signal,
            high_pass_cutoff=self.high_cutoff, low_pass_cutoff=self.low_cutoff)
        self.filtered_signal = self.filtered_signal_obj.filtered_signal
        self.background = self.filtered_signal_obj.bg_sub_signal
        self.fs = self.filtered_signal_obj.fs

        # processing parameters
        self.threshold_frac = kwargs.get('threshold_frac', 1)
        self.threshold = None
        self.binary_signal = None
        self.binary_centers = None
        self.rising_edges = None
        self.falling_edges = None

    def find_beats(self):
        """
        Finds the beats from the signal.
        Returns: Times at which the beats occur.

        """
        self.binary_signal = self._apply_threshold(
            self.filtered_signal, self.background)
        self.binary_centers = self._find_binary_centers(self.binary_signal)
        # find the indices where it equals 1
        beat_ind = self._find_indices(self.binary_centers, lambda x: x == 1)
        beat_time_list = np.take(self.time, tuple(beat_ind))
        return beat_time_list.tolist()

    def _apply_threshold(self, signal, background):
        """
        Applies a threshold of a certain percentage.
        Args:
            signal: Filtered signal in numpy array

        Returns: list of binary values

        """
        self.threshold = self._find_threshold(signal, background)
        bin_sig = abs(signal) >= self.threshold
        return bin_sig

    def _find_threshold(self, signal, background, filter_bg: bool = True):
        """
        Determines threshold based on a absolute-value-filtered/zeroed signal and proportion.
        Threshold is padded by one period.
        Args:
            filter_bg (bool): Whether or not to filter the background.
            background (object): background for the signal
            signal: heart beat signal

        Returns: Threshold array

        """
        if filter_bg:
            background = self.filtered_signal_obj.apply_noise_reduction(
                background, self.low_cutoff + 10, max(0, self.high_cutoff - 5))

        padding = self.filtered_signal_obj.period
        start_ind = padding
        end_ind = len(background) - padding
        padded_signal = signal[start_ind:end_ind]
        min_v, max_v = self._find_voltage_extremes(padded_signal)

        # determine if spikes tend to be positive or negative
        threshold_array = []
        threshold_value = self.threshold_frac * max(abs(max_v), abs(min_v))
        for bg_val in background:
            threshold_array.append(threshold_value - abs(bg_val))
            # threshold_array.append(threshold_value)

        return threshold_array

    def plot_graph(self, file_path=None):
        """
        Plots a graph of relevant information for the threshold algorithm.
        Args:
            file_path: The path of the file to output.
        """
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

    def _find_indices(self, values, func):
        """
        Finds indices of an array given parameters.
        Args:
            values: list of values
            func: lambda function

        Returns: list of indices

        """
        return [i for (i, val) in enumerate(values) if func(val)]

    def _find_binary_centers(self, bin_signal, min_width=1):
        # first make sure that this is a binary signal
        """
        Finds the centers of the thresholded binary signal.
        Args:
            min_width (int): Minimum width for binary signal.
            bin_signal: binary signal

        Returns: list of binary values representing the centers of the binary steps.

        """
        is_binary = self._confirm_binary(bin_signal)
        if not is_binary:
            raise ValueError("Signal is not binary")

        self.rising_edges = self._find_rising_edges(bin_signal)
        self.falling_edges = self._find_falling_edges(bin_signal)

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

    def _find_rising_edges(self, bin_signal):
        """
        Finds the rising edge of a binary signal.
        Args:
            bin_signal: binary signal

        Returns: list of binary values representing the rising edge

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

    def _find_falling_edges(self, bin_signal):
        """
        Finds the falling edge of a binary signal.
        Args:
            bin_signal: binary signal

        Returns: list of binary values representing the rising edge

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
                falling_edges.append(i)
            elif val == 1:
                previous_val = 1

        return np.array(falling_edges)

    def _confirm_binary(self, signal):
        """
        Tests of the signal is a binary signal of 0s and 1s
        Args:
            signal: signal to test

        Returns: boolean of if it is a binary signal

        """
        return np.array_equal(signal, signal.astype(bool))


class Convolution(Threshold):
    def __init__(self, time, signal):
        super().__init__(time, signal)

    def find_beats(self):
        """
        Finds the beats from the signal using convolution
        Returns: Times at which the beats occur.
        """
        pass


class Wavelet(ECGDetectionAlgorithm):
    def __init__(self, time, signal):
        super().__init__(time, signal)

    def find_beats(self):
        """
        Finds the beats from the signal using a continuous wavelet transform.
        Returns: Times at which the beats occur.
        """
        pass
