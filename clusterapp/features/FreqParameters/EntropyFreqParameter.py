import numpy as np

from clusterapp.features.utils import get_location, geometric_mean
from .FreqParameter import FreqParameter


class EntropyFreqParameter(FreqParameter):
    name = 'EntropyFreq'

    """docstring for EntropyFreqParameter"""

    def __init__(self):
        super(FreqParameter, self).__init__()

    def measure(self, segment, location='center'):
        if location is None:
            return self.__measure_spectrum(segment)

        j = get_location(segment, location)

        data = np.array(segment.spec[:, j], np.float64)
        g_mean = geometric_mean(data)
        a_mean = np.mean(data)
        value = g_mean / a_mean

        segment.measures_dict[self.name + '(' + location + ')'] = value
        return True

    def __measure_spectrum(self, segment):
        data = np.array(segment.spectrum, np.float64)
        g_mean = geometric_mean(data)
        a_mean = np.mean(data)
        value = g_mean / a_mean

        segment.measures_dict[self.name + '(total)'] = value
        return True