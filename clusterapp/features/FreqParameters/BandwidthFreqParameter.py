import numpy as np

from clusterapp.features.utils import apply_threshold, get_location
from .FreqParameter import FreqParameter
from .__init__ import *


class BandwidthFreqParameter(FreqParameter):
    name = 'BandwidthFreq'

    """docstring for BandwidthFreqParameter"""

    def __init__(self, total=True):
        super(FreqParameter, self).__init__()
        self.total = total

    def measure(self, segment, threshold=-20, location='center'):
        if segment.peaks_values is None:
            segment.compute_peaks()

        if location is None:
            return self.__measure_spectrum(segment, threshold)
        j = get_location(segment, location)
        i = segment.peaks_indexes[j]

        value = apply_threshold(segment.spec[i, j], threshold)

        if self.total:
            min_freq = np.argwhere(segment.spec[:, j] >= value).min()
            max_freq = np.argwhere(segment.spec[:, j] >= value).max()
        else:
            below = segment.spec[:, j] < value
            below[i:] = False
            min_freq = np.argwhere(below).max()

            below = segment.spec[:, j] < value
            below[:i] = False
            max_freq = np.argwhere(below).min()

        segment.measures_dict[self.name + '(' + location + ')'] = np.round(
            segment.freqs[max_freq] - segment.freqs[min_freq],
            DECIMAL_PLACES)
        return True

    def __measure_spectrum(self, segment, threshold):
        if segment.peaks_values is None:
            segment.compute_peaks()

        i = np.argmax(segment.spectrum)
        value = apply_threshold(segment.spectrum[i], threshold)

        if self.total:
            min_freq = np.argwhere(segment.spectrum >= value).min()
            max_freq = np.argwhere(segment.spectrum >= value).max()
        else:
            below = segment.spectrum < value
            below[i:] = False
            min_freq = np.argwhere(below).max()

            below = segment.spectrum < value
            below[:i] = False
            max_freq = np.argwhere(below).min()

        segment.measures_dict[self.name + '(total)'] = np.round(
            segment.spectrum_freqs[max_freq] - segment.spectrum_freqs[min_freq],
            DECIMAL_PLACES)
        return True
