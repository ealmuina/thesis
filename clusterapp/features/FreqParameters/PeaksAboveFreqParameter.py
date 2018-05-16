import numpy as np
from scipy.ndimage import label

from clusterapp.features.utils import get_location, apply_threshold
from .FreqParameter import FreqParameter


class PeaksAboveFreqParameter(FreqParameter):
    name = 'PeaksAboveFreq'

    """docstring for PeaksAboveFreqParameter"""

    def __init__(self):
        super(FreqParameter, self).__init__()

    def measure(self, segment, threshold=-20, location='center'):
        if location is None:
            return self.__measure_spectrum(segment, threshold)

        j = get_location(segment, location)
        i = segment.peaks_indexes[j]

        value = apply_threshold(segment.spec[i, j], threshold)
        _, cnt_regions = label(segment.spec[:, j] >= value)

        segment.measures_dict[self.name + '(' + location + ')'] = int(cnt_regions)
        return True

    def __measure_spectrum(self, segment, threshold=-20):
        i = np.argmax(segment.spectrum)

        value = apply_threshold(segment.spectrum[i], threshold)
        _, cnt_regions = label(segment.spectrum >= value)

        segment.measures_dict[self.name + '(total)'] = int(cnt_regions)
        return True
