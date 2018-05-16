import numpy as np

from clusterapp.features.utils import get_location
from .FreqParameter import FreqParameter


class FluxFreqParameter(FreqParameter):
    name = 'Flux'

    """docstring for FluxFreqParameter"""

    def __init__(self):
        super(FreqParameter, self).__init__()

    def measure(self, segment, location='center'):
        j = get_location(segment, location)

        flux = 0.
        for i in range(segment.spec.shape[0]):
            flux += (segment.spec[i, j] - segment.spec[i, j - 1]) ** 2

        value = np.sqrt(flux)

        segment.measures_dict[self.name + '(' + location + ')'] = value
        return True
