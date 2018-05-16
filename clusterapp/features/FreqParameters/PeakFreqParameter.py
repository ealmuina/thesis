import numpy as np

from clusterapp.features.utils import get_location
from .FreqParameter import FreqParameter
from .__init__ import *


class PeakFreqParameter(FreqParameter):
    name = 'PeakFreq'

    """docstring for PeakFreqParameter"""

    def __init__(self):
        super(FreqParameter, self).__init__()

    def measure(self, segment, location='center'):
        if location is None:
            return self.__measure_spectrum(segment)

        j = get_location(segment, location)
        i = segment.peaks_indexes[j]

        segment.measures_dict[self.name + '(' + location + ')'] = np.round(segment.freqs[i], DECIMAL_PLACES)
        return True

    def __measure_spectrum(self, segment):
        i = np.argmax(segment.spectrum)

        segment.measures_dict[self.name + '(total)'] = np.round(segment.spectrum_freqs[i], DECIMAL_PLACES)
        return True
