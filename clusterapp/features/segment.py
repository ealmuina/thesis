import numpy as np
from scipy.signal import spectrogram

from .utils import apply_mean_filter, apply_median_filter


class Signal:
    """docstring for Signal"""

    def __init__(self, data, samplingRate, filter=None):
        self.data = np.array(data, np.float64) / np.max(data)
        self.samplingRate = samplingRate
        # Spectrogram filter: 'mean', 'median'
        self.filter = filter
        self.window = 'boxcar'

        # Global spectrogram information
        self.freqs = None
        self.times = None
        self.spec = None
        self.db_reference = None

        # Peaks indexes and amplitudes of peak frequency by frame
        self.peaks_indexes = None
        self.peaks_values = None

    def set_window(self, window):
        """Any of the scipy available windows"""
        self.window = window
        return True

    def compute_spectrogram(self, nperseg=256, percent_overlap=0.75):
        self.freqs, self.times, self.spec = spectrogram(self.data, fs=self.samplingRate,
                                                        window=(self.window), nperseg=256,
                                                        noverlap=int(nperseg * percent_overlap),
                                                        nfft=None, detrend='constant', return_onesided=True,
                                                        scaling='density', axis=-1, mode='psd')
        if self.filter == 'mean':
            self.spec = apply_mean_filter(self.spec)
        elif self.filter == 'median':
            self.spec = apply_median_filter(self.spec)
        self.db_reference = np.max(self.spec)
        return True

    def compute_peaks(self):
        if self.spec is None:
            self.compute_spectrogram()

        indexes, values = [], []
        for i in range(self.spec.shape[1]):
            indexes.append(np.argmax(self.spec[:, i]))
            values.append(np.max(self.spec[:, i]))

        self.peaks_indexes = np.array(indexes)
        self.peaks_values = np.array(values)
        return True


class Segment:
    """docstring for Segment"""

    def __init__(self, signal, a, b):
        self.signal = signal

        # Segment time signal information
        self.data = self.signal.data[a:b + 1]
        self.IndexFrom = a
        self.IndexTo = b
        self.samplingRate = self.signal.samplingRate

        if self.signal.freqs is None:
            self.signal.compute_spectrogram()

        # Segment spectrogram information
        self.FrameFrom = self.signal.spec.shape[1] * a // len(self.signal.data)
        self.FrameTo = self.signal.spec.shape[1] * b // len(self.signal.data)

        self.freqs = self.signal.freqs
        self.times = self.signal.times[self.FrameFrom:self.FrameTo + 1]
        self.spec = self.signal.spec[:, self.FrameFrom:self.FrameTo + 1]

        if self.signal.peaks_indexes is None:
            self.signal.compute_peaks()

        self.peaks_indexes = self.signal.peaks_indexes[self.FrameFrom:self.FrameTo + 1]
        self.peaks_values = self.signal.peaks_values[self.FrameFrom:self.FrameTo + 1]

        # Dictionary of segment measures "name":value
        self.measures_dict = {}
