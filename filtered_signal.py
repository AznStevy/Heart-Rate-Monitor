import logging
import numpy as np
import scipy.signal as sp
import scipy.stats as stats

logging.basicConfig(filename='heart_rate_monitor.log', level=logging.DEBUG)


class FilteredSignal(object):
    def __init__(self, time, signal, **kwargs):
        # honestly, raising these in here seems really clunky,
        # but idk if there's any other way...
        self.time = self._check_list_input(time)
        self.raw_signal = self._check_list_input(signal)
        if len(self.time) != len(self.raw_signal):
            raise ValueError("time and signal array lengths much match.")

        # --------------- other properties --------------------

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
        # don't need to raise exceptions here because handled above.
        self.fs = self.determine_frequency(self.time)
        self.filtered_signal = self.clean_signal(filter_sig=filter_sig)

    def get_properties(self) -> dict:
        """
        Gets some signal properties. Nothing directly related to the signal is returned.
        Returns:
            dict: properties of the filtered signal.

        """
        info = {
            "high_pass_cutoff": self.high_pass_cutoff,
            "low_pass_cutoff": self.low_pass_cutoff,
            "period": self.period,
            "fs": self.fs,
        }
        return info

    def _check_list_input(self, test_list) -> np.ndarray:
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

        if not self._is_numeric_list(test_list):
            raise ValueError("All elements must be numeric.")

        return np.array(test_list)

    def _is_numeric_list(self, list) -> bool:
        """
        Tests if the list contains all numeric values.
        Args:
            list (numpy.ndarray): list or numpy array.

        Returns:
            bool: Whether all values are numeric.

        """
        bool_list = np.isreal(list)
        return np.all(bool_list)

    def clean_signal(self, filter_sig: bool = True) -> np.ndarray:
        """
        Applies a moving average subtraction to get rid of global drift and noise reduction filters.

        Args:
            filter_sig (bool): Whether or not to filter the signal.

        Returns:
            numpy.ndarray: Filtered signal
        """
        try:
            self.bg_sub_signal = self.apply_moving_average_sub(self.raw_signal)
        except ZeroDivisionError as e:
            logging.exception(e)
            self.bg_sub_signal = self.raw_signal
        except TypeError as e:
            logging.exception(e)
            self.bg_sub_signal = self.raw_signal
        except ValueError as e:
            logging.exception(e)
            self.bg_sub_signal = self.raw_signal

        if filter_sig:
            try:
                signal_noise_removed = self.apply_noise_reduction(
                    self.bg_sub_signal, self.low_pass_cutoff, self.high_pass_cutoff)
            except TypeError as e:
                logging.exception(e)
                return self.bg_sub_signal
            except ValueError as e:
                logging.exception(e)
                return self.bg_sub_signal
            return signal_noise_removed
        else:
            return self.bg_sub_signal

    def apply_moving_average_sub(self, signal=None) -> np.ndarray:
        """
        Applies a moving average filter and subtracts it from the signal.

        Args:
            signal: Signal to apply moving average background subtraction.

        Returns:
            np.ndarray: Signal with moving average subtracted.
        """
        if signal is None:
            signal = self.raw_signal
        if type(signal) != list and type(signal) != np.ndarray:
            raise TypeError("signal must be numpy.array.")
        if not self._is_numeric_list(signal):
            raise ValueError("All elements must be numeric.")

        signal = np.array(signal)
        self.period = round(len(signal) / 100)

        if self.period < 1:
            raise ZeroDivisionError("Period was not detected.")

        periods = round(len(signal) / self.period)
        weights = np.ones(periods) / periods
        mov_avg = np.convolve(signal, weights, mode='same')

        return signal - mov_avg

    def apply_noise_reduction(self, signal=None, low_pass_cutoff=None, high_pass_cutoff=None) -> np.ndarray:
        """
        Applies a bandpass filter determined by some frequency analysis.

        Args:
            signal: Signal to apply noise reduction to.
            low_pass_cutoff (int): Low-pass filter cut-off.
            high_pass_cutoff (int): High-pass filter cut-off.

        Returns:
            numpy.array: Background subtracted and filtered signal.
        """
        if signal is None:
            signal = self.raw_signal
        if type(signal) != list and type(signal) != np.ndarray:
            raise TypeError("signal must be numpy.array.")
        signal = np.array(signal)
        if not low_pass_cutoff:
            low_pass_cutoff = self.low_pass_cutoff
        if not high_pass_cutoff:
            high_pass_cutoff = self.high_pass_cutoff
        if low_pass_cutoff <= high_pass_cutoff:
            raise ValueError("low_pass_cutoff must be greater than high_pass_cutoff.")

        # if one fails, they will most likely both fail.
        try:
            low_passed = self.apply_low_pass(signal, low_pass_cutoff)
            high_passed = self.apply_high_pass(low_passed, high_pass_cutoff)
            return np.array(high_passed)
        except ValueError as e:
            logging.exception(e)
            return signal

    def apply_high_pass(self, signal=None, high_cutoff: int = None, order: int = 1, fs=None) -> np.ndarray:
        """
        Applies a high-pass filter.
        Args:
            fs (float): Sampling frequency
            signal (numpy.ndarray): Signal to high-pass
            high_cutoff (int): Cutoff frequency in Hz
            order (int): Order of the filter

        Returns:
            numpy.ndarray: High-passed signal.
        """

        if signal is not None:
            if type(signal) != list and type(signal) != np.ndarray:
                raise TypeError("signal must be numpy.array.")
            if not self._is_numeric_list(signal):
                raise ValueError("All elements must be numeric.")
            signal = np.array(signal)
        else:
            signal = self.raw_signal

        if not high_cutoff:
            logging.warning("High-pass order not valid. Using order=1.")
            high_cutoff = self.high_pass_cutoff
        elif high_cutoff < 1:
            raise ValueError("High-pass cutoff must be >= 1.")

        if order < 1:
            raise ValueError("High-pass order must be >= 1.")

        if not fs:
            fs = self.fs
        else:
            if type(fs) != float and type(fs) != int:
                raise TypeError("fs must type float.")

        nyq = 0.5 * fs
        high = high_cutoff / nyq
        b, a = sp.butter(order, high, btype="highpass")
        filtered_signal = sp.filtfilt(b, a, signal)
        if np.isnan(filtered_signal).any():
            raise ValueError("Failed to high-pass filter.")
        return filtered_signal

    def apply_low_pass(self, signal=None, low_cutoff: int = None, order: int = 1, fs=None) -> np.ndarray:
        """
        Applies a low-pass filter.
        Args:
            fs (float): Sampling frequency
            signal: Signal to low-pass
            low_cutoff: Cutoff frequency in Hz
            order: Order of the filter

        Returns:
            numpy.array: Low-passed signal.
        """
        if signal is not None:
            if type(signal) != list and type(signal) != np.ndarray:
                raise TypeError("signal must be numpy.array.")
            if not self._is_numeric_list(signal):
                raise ValueError("All elements must be numeric.")
            signal = np.array(signal)
        else:
            signal = self.raw_signal

        if not low_cutoff:
            logging.warning("Low-pass order not valid. Using order=1.")
            low_cutoff = self.high_pass_cutoff
        elif low_cutoff < 1:
            raise ValueError("Low-pass cutoff must be >= 1.")

        if order < 1:
            raise ValueError("Low-pass order must be >= 1.")

        if not fs:
            fs = self.fs
        else:
            if type(fs) != float and type(fs) != int:
                raise TypeError("fs must type float.")

        nyq = 0.5 * fs
        high = low_cutoff / nyq
        b, a = sp.butter(order, high, btype="lowpass")
        filtered_signal = sp.filtfilt(b, a, signal)
        if np.isnan(filtered_signal).any():
            raise ValueError("Failed to low-pass filter.")
        return filtered_signal

    def determine_frequency(self, time=None) -> float:
        """
        Determines the frequencies with a time array.
        Args:
            time: an array of times which correspond to the signal

        Returns:
            float: frequency of the signal

        """
        if time is None:
            time = self.time
        if time.size < 2:
            raise ValueError("time must have 2> elements.")
        if type(time) != list and type(time) != np.ndarray:
            raise TypeError("time must be numpy.array.")

        time = np.array(time)  # ensure numpy
        periods = np.diff(time)

        logging.info("Determining frequency using mode of time deltas.")
        return float(1 / stats.mode(periods, axis=None)[0])

    def get_fft(self, signal=None, is_filtered: bool = False) -> tuple:
        """
        Gets FFT of a signal
        Args:
            signal (numpy.ndarray): Signal to fft. Defaults to signal that was read in at instantiation.
            is_filtered (bool): Use the filtered signal or raw signal.

        Returns:
            tuple: first element is frequency numpy.ndarray and second is fft numpy.ndarray .

        """
        if signal is not None:
            if type(signal) != list and type(signal) != np.ndarray:
                raise TypeError("signal must be numpy.array.")
        else:
            if not is_filtered:
                signal = self.raw_signal
            else:
                signal = self.filtered_signal

        if type(is_filtered) != bool:
            raise TypeError("is_filtered must be type bool.")

        n = len(signal)
        k = np.arange(n)
        t = n / self.fs
        frq = k / t  # two sides frequency range
        frq = frq[range(int(n / 2))]  # one side frequency range

        fft_out = np.fft.fft(signal) / n  # fft computing and normalization
        fft_out = fft_out[range(int(n / 2))]
        return frq, fft_out
