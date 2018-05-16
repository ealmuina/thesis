import numpy as np

from clusterapp.features.utils import get_location, energy
from .FreqParameter import FreqParameter


class EnergyFreqParameter(FreqParameter):
    name = 'Spectral Energy'

    """docstring for EnergyFreqParameter"""

    def __init__(self):
        super(FreqParameter, self).__init__()

    def measure(self, segment, location='center'):
        if location is None:
            return self.__measure_spectrum(segment)

        j = get_location(segment, location)

        data = np.array(segment.spec[:, j])
        value = energy(data)

        segment.measures_dict[self.name + '(' + location + ')'] = value
        return True

    def __measure_spectrum(self, segment):
        data = np.array(segment.spectrum)
        value = energy(data)

        segment.measures_dict[self.name + '(total)'] = value
        return True
