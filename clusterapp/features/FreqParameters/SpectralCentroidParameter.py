import numpy as np

from clusterapp.features.utils import get_location
from .FreqParameter import FreqParameter
from .__init__ import *


class SpectralCentroidParameter(FreqParameter):
    name = 'Spectral Centroid'

    """docstring for SpectralCentroidParameter"""

    def __init__(self):
        super(FreqParameter, self).__init__()

    def measure(self, segment, location='center'):
        if location is None:
            return self.__measure_spectrum(segment)

        j = get_location(segment, location)
        data = segment.spec[:, j]

        indexes = np.array(range(1, len(data) + 1))
        weights = np.sum(data)
        if weights != 0.0:
            value = ((np.sum(indexes * data) + segment.IndexFrom) / weights)
        else:
            value = 0.0
        value *= segment.freqs[-1] / len(segment.freqs)

        segment.measures_dict[self.name + '(' + location + ')'] = np.round(value, DECIMAL_PLACES)
        return True

    def __measure_spectrum(self, segment):
        data = segment.spectrum

        indexes = np.array(range(1, len(data) + 1))
        weights = np.sum(data)
        if weights != 0.0:
            value = ((np.sum(indexes * data) + segment.IndexFrom) / weights)
        else:
            value = 0.0
        value *= segment.spectrum_freqs[-1] / len(segment.spectrum_freqs)

        segment.measures_dict[self.name + '(total)'] = np.round(value, DECIMAL_PLACES)
        return True
