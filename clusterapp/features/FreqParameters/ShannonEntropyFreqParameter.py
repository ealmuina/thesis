import numpy as np

from clusterapp.features.utils import get_location
from .FreqParameter import FreqParameter


class ShannonEntropyFreqParameter(FreqParameter):
    name = 'Shannon Entropy'

    """docstring for ShannonEntropyFreqParameter"""

    def __init__(self):
        super(FreqParameter, self).__init__()

    def measure(self, segment, location='center'):
        if location is None:
            return self.__measure_spectrum(segment)

        j = get_location(segment, location)

        data = np.array(segment.spec[:, j], np.float64)
        entropy = 0.0
        for i in range(len(data)):
            if data[i] == 0.0:
                entropy -= np.log2(1)
            else:
                entropy -= np.log2(data[i]) * data[i]

        segment.measures_dict[self.name + '(' + location + ')'] = entropy
        return True

    def __measure_spectrum(self, segment):
        data = np.array(segment.spectrum, np.float64)
        entropy = 0.0
        for i in range(len(data)):
            if data[i] == 0.0:
                entropy -= np.log2(1)
            else:
                entropy -= np.log2(data[i]) * data[i]

        segment.measures_dict[self.name + '(total)'] = entropy
        return True
