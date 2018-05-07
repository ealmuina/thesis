import numpy as np

from .FreqParameter import FreqParameter
from .__init__ import *
from ..utils import get_location, toDB


class PeakAmpFreqParameter(FreqParameter):
    name = 'PeakAmpFreq'

    """docstring for MaxFreqParameter"""

    def __init__(self):
        super(FreqParameter, self).__init__()

    def measure(self, segment, location='center'):
        if segment.peaks_values is None:
            segment.compute_peaks()

        index_frame = get_location(segment, location)
        j = segment.peaks_values[index_frame]
        db_reference = segment.signal.db_reference
        segment.measures_dict[self.name + '-' + location] = np.round(toDB(j, db_reference), DECIMAL_PLACES)
        return True
