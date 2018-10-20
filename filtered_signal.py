import numpy as np
import scipy.signal as sp
import scipy.stats as stats


class FilteredSignal(object):
    def __init__(self, time, signal, **kwargs):
        # kwargs
        self.time = time
        self.raw_signal = signal
        # http://www.ems12lead.com/wp-content/uploads/sites/42/2014/03/ecg-component-frequencies.jpg
        self.high_pass_cutoff = kwargs.get('high_pass_cutoff', 2)
        self.low_pass_cutoff = kwargs.get('low_pass_cutoff', 30)

        # other attributes
        self.period = 0  # for moving average
        self.bg_sub_signal = None
        self.fs = self._determine_frequency(self.time)
        self.filtered_signal = self.clean_signal(self.raw_signal)

    def clean_signal(self, raw_signal):
        """
        Applies a moving average subtraction to get rid of global drift and noise reduction filters.
        """
        bg_drift_subbed = self.apply_moving_average_sub(raw_signal)
        self.bg_sub_signal = bg_drift_subbed
        signal_noise_removed = self.apply_noise_reduction(
            bg_drift_subbed, self.low_pass_cutoff, self.high_pass_cutoff)
        return signal_noise_removed

    def apply_moving_average_sub(self, signal):
        """
        Applies a moving average filter and subtracts it from the signal
        """
        self.period = round(len(signal) / 100)
        periods = round(len(signal) / self.period)
        weights = np.ones(periods) / periods
        mov_avg = np.convolve(signal, weights, mode='same')

        return signal - mov_avg

    def apply_noise_reduction(self, signal, lowpass_cutoff, highpass_cutoff):
        """
        Applies a bandpass filter determined by some frequency analysis.
        """
        try:
            low_passed = self._apply_low_pass(signal, lowpass_cutoff)
            high_passed = self._apply_high_pass(low_passed, highpass_cutoff)
            return high_passed
        except ValueError:
            return signal

    def _apply_high_pass(self, signal, high_cutoff, order=1):
        """
        Applies a high pass filter.
        Args:
            high_cutoff: Cutoff frequency in Hz
            order: Order of the filter
        """
        nyq = 0.5 * self.fs
        high = high_cutoff / nyq
        b, a = sp.butter(order, high, btype="highpass")
        filtered_signal = sp.filtfilt(b, a, signal)
        if np.isnan(filtered_signal).any():
            raise ValueError("Failed to high-pass filter.")
        return filtered_signal

    def _apply_low_pass(self, signal, low_cutoff, order=1):
        """
        Applies a low pass filter.
        Args:
            low_cutoff: Cutoff frequency in Hz
            order: Order of the filter
        """
        nyq = 0.5 * self.fs
        high = low_cutoff / nyq
        b, a = sp.butter(order, high, btype="lowpass")
        filtered_signal = sp.filtfilt(b, a, signal)
        if np.isnan(filtered_signal).any():
            raise ValueError("Failed to low-pass filter.")
        return filtered_signal

    def _determine_frequency(self, time):
        """
        Determines the frequencies with a time array.
        Args:
            time: an array of times which correspond to the signal

        Returns: frequency of the signal
        """
        time = np.array(time)
        periods = np.diff(time)
        return float(1 / stats.mode(periods, axis=None)[0])

    def get_fft(self, is_filtered=False):
        """
        Gets FFT of a signal
        Args:
            is_filtered: Use the filtered signal or raw signal.

        Returns: Tuple with frequency and fft.

        """
        if not is_filtered:
            signal = self.raw_signal
        else:
            signal = self.filtered_signal

        n = len(signal)
        k = np.arange(n)
        T = n / self.fs
        frq = k / T  # two sides frequency range
        frq = frq[range(int(n / 2))]  # one side frequency range

        fft_out = np.fft.fft(signal) / n  # fft computing and normalization
        fft_out = fft_out[range(int(n / 2))]
        return frq, fft_out
