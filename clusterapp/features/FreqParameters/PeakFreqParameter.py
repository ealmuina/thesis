import numpy as np

from .FreqParameter import FreqParameter
from .__init__ import *
from ..utils import get_location


class PeakFreqParameter(FreqParameter):
    name = 'PeakFreq'

    """docstring for PeakFreqParameter"""

    def __init__(self):
        super(FreqParameter, self).__init__()

    def measure(self, segment, location='center'):
        j = get_location(segment, location)
        i = segment.peaks_indexes[j]

        segment.measures_dict[self.name + '-' + location] = np.round(segment.freqs[i], DECIMAL_PLACES)
        return True
