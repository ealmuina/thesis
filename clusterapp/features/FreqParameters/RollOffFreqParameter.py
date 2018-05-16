import numpy as np

from clusterapp.features.utils import get_location
from .FreqParameter import FreqParameter
from .__init__ import *


class RollOffFreqParameter(FreqParameter):
    name = 'Roll Off Freq'

    """docstring for RollOffFreqParameter"""

    def __init__(self):
        super(FreqParameter, self).__init__()

    def measure(self, segment, location='center', cutoff=0.95):
        if location is None:
            return self.__measure_spectrum(segment, cutoff)

        j = get_location(segment, location)
        data = segment.spec[:, j]

        energy = np.sum(np.square(data))
        cutoff *= energy

        energy_sum = 0.
        roll_off = None

        for i in range(len(data)):
            energy_sum += data[i] ** 2
            if energy_sum >= cutoff:
                roll_off = i
                break

        segment.measures_dict[self.name + '(' + location + ')'] = np.round(segment.freqs[roll_off], DECIMAL_PLACES)
        return True

    def __measure_spectrum(self, segment, cutoff=0.95):
        data = segment.spectrum

        energy = np.sum(np.square(data))
        cutoff *= energy

        energy_sum = 0.
        roll_off = None

        for i in range(len(data)):
            energy_sum += data[i] ** 2
            if energy_sum >= cutoff:
                roll_off = i
                break

        segment.measures_dict[self.name + '(total)'] = np.round(segment.spectrum_freqs[roll_off], DECIMAL_PLACES)
        return True
