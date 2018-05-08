import numpy as np

from .FreqParameter import FreqParameter
from .__init__ import *
from ..utils import get_location


class EntropyFreqParameter(FreqParameter):
    name = 'EntropyFreq'

    """docstring for EntropyFreqParameter"""

    def __init__(self):
        super(FreqParameter, self).__init__()

    def measure(self, segment, location='center'):
        j = get_location(segment, location)
        i = segment.peaks_indexes[j]

        data = np.array(segment.spec[:, j], np.float64)
        gmean = np.exp(np.sum(np.log(data[np.nonzero(data)])) / len(data))
        amean = np.mean(data)
        value = gmean / amean

        segment.measures_dict[self.name + '-' + location] = np.round(value, DECIMAL_PLACES)
        return True
