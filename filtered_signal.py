import numpy as np
import scipy.signal as sp
import scipy.stats as stats


class FilteredSignal(object):
    def __init__(self, time, signal, **kwargs):
        # honestly, raising these in here seems really clunky,
        # but idk if there's any other way...
        self.time = self._check_list_input(time)
        self.raw_signal = self._check_list_input(signal)
        if len(self.time) != len(self.raw_signal):
            raise ValueError("time and signal array lengths much match.")

        # http://www.ems12lead.com/wp-content/uploads/sites/42/2014/03/ecg-component-frequencies.jpg
        self.high_pass_cutoff = kwargs.get('high_pass_cutoff', 1)
        if type(self.high_pass_cutoff) != int:
            raise TypeError("high_pass_cutoff must be type int.")
        self.low_pass_cutoff = kwargs.get('low_pass_cutoff', 30)
        if type(self.low_pass_cutoff) != int:
            raise TypeError("low_pass_cutoff must be type int.")
        if self.low_pass_cutoff <= self.high_pass_cutoff:
            raise ValueError("high_pass_cutoff must be less than low_pass_cutoff.")

        filter_sig = kwargs.get('filter', True)
        if type(filter_sig) != bool:
            raise TypeError("filter_sig must be type bool.")

        # other class attributes
        self.period = 0  # for moving average
        self.bg_sub_signal = None
        self.fs = self._determine_frequency(self.time)
        self.filtered_signal = self.clean_signal(filter_sig=filter_sig)

    def _check_list_input(self, test_list):
        """
        Checks if all elements are valid.
        Args:
            test_list: list to check

        Returns:
            numpy.ndarray: original list.

        """
        if type(test_list) != list and type(test_list) != np.ndarray:
            raise TypeError("Must be numpy.array.")
        # see if anything is not numeric
        """
        if not self._is_integer(test_list).all():
            raise ValueError("All elements must be numeric.")"""

        return np.array(test_list)

    def clean_signal(self, filter_sig: bool = True):
        """
        Applies a moving average subtraction to get rid of global drift and noise reduction filters.

        Args:
            filter_sig (bool): Whether or not to filter the signal.

        Returns:
            numpy.array: Filtered signal
        """
        self.bg_sub_signal = self.apply_moving_average_sub(self.raw_signal)
        if filter_sig:
            signal_noise_removed = self.apply_noise_reduction(
                self.bg_sub_signal, self.low_pass_cutoff, self.high_pass_cutoff)
            return signal_noise_removed
        else:
            return self.bg_sub_signal

    def apply_moving_average_sub(self, signal):
        """
        Applies a moving average filter and subtracts it from the signal.

        Args:
            signal: Signal to apply moving average background subtraction.
        """
        if type(signal) != list and type(signal) != np.ndarray:
            raise TypeError("signal must be numpy.array.")

        signal = np.array(signal)
        self.period = round(len(signal) / 100)
        periods = round(len(signal) / self.period)
        weights = np.ones(periods) / periods
        mov_avg = np.convolve(signal, weights, mode='same')

        return signal - mov_avg

    def apply_noise_reduction(self, signal, low_pass_cutoff: int, high_pass_cutoff: int):
        """
        Applies a bandpass filter determined by some frequency analysis.

        Args:
            signal: Signal to apply noise reduction to.
            low_pass_cutoff (int): Low-pass filter cut-off
            high_pass_cutoff (int): High-pass filter cut-off

        Returns:
            numpy.array: Background subtracted and filtered signal.
        """
        if type(signal) != list and type(signal) != np.ndarray:
            raise TypeError("signal must be numpy.array.")
        signal = np.array(signal)

        if low_pass_cutoff <= high_pass_cutoff:
            raise ValueError("low_pass_cutoff must be greater than high_pass_cutoff.")

        try:
            low_passed = self._apply_low_pass(signal, low_pass_cutoff)
            high_passed = self._apply_high_pass(low_passed, high_pass_cutoff)
            return high_passed
        except ValueError:
            return signal

    def _apply_high_pass(self, signal, high_cutoff: int, order: int = 1):
        """
        Applies a high-pass filter.
        Args:
            signal: Signal to high-pass
            high_cutoff (int): Cutoff frequency in Hz
            order: Order of the filter

        Returns:
            numpy.array: High-passed signal.
        """
        if type(signal) != list and type(signal) != np.ndarray:
            raise TypeError("signal must be numpy.array.")
        signal = np.array(signal)

        nyq = 0.5 * self.fs
        high = high_cutoff / nyq
        b, a = sp.butter(order, high, btype="highpass")
        filtered_signal = sp.filtfilt(b, a, signal)
        if np.isnan(filtered_signal).any():
            raise ValueError("Failed to high-pass filter.")
        return filtered_signal

    def _apply_low_pass(self, signal, low_cutoff: int, order: int = 1):
        """
        Applies a low-pass filter.
        Args:
            signal: Signal to low-pass
            low_cutoff: Cutoff frequency in Hz
            order: Order of the filter

        Returns:
            numpy.array: Low-passed signal.
        """
        if type(signal) != list and type(signal) != np.ndarray:
            raise TypeError("signal must be numpy.array.")
        signal = np.array(signal)

        nyq = 0.5 * self.fs
        high = low_cutoff / nyq
        b, a = sp.butter(order, high, btype="lowpass")
        filtered_signal = sp.filtfilt(b, a, signal)
        if np.isnan(filtered_signal).any():
            raise ValueError("Failed to low-pass filter.")
        return filtered_signal

    def _determine_frequency(self, time=None):
        """
        Determines the frequencies with a time array.
        Args:
            time: an array of times which correspond to the signal

        Returns: frequency of the signal
        """
        if time is None:
            time = self.time

        if type(time) != list and type(time) != np.ndarray:
            raise TypeError("time must be numpy.array.")

        time = np.array(time)
        periods = np.diff(time)
        return float(1 / stats.mode(periods, axis=None)[0])

    def get_fft(self, signal=None, is_filtered: bool = False):
        """
        Gets FFT of a signal
        Args:
            signal: Signal to fft. Defaults to signal that was read in at instantiation.
            is_filtered: Use the filtered signal or raw signal.

        Returns:
            tuple: frequency and fft.

        """
        if signal is not None:
            if type(signal) != list and type(signal) != np.ndarray:
                raise TypeError("signal must be numpy.array.")

        if signal is None:
            if not is_filtered:
                signal = self.raw_signal
            else:
                signal = self.filtered_signal

        n = len(signal)
        k = np.arange(n)
        t = n / self.fs
        frq = k / t  # two sides frequency range
        frq = frq[range(int(n / 2))]  # one side frequency range

        fft_out = np.fft.fft(signal) / n  # fft computing and normalization
        fft_out = fft_out[range(int(n / 2))]
        return frq, fft_out
