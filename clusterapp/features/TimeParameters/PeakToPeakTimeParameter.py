import numpy as np

from .TimeParameter import TimeParameter
from .__init__ import *


class PeakToPeakTimeParameter(TimeParameter):
    name = 'PeakToPeakTime'

    """docstring for PeakToPeakTimeParameter"""

    def __init__(self):
        super(TimeParameter, self).__init__()

    def measure(self, segment):
        data = segment.data
        data /= np.max(np.abs(segment.signal.data))
        value = np.ptp(data) / 2.

        segment.measures_dict[self.name] = np.round(value, DECIMAL_PLACES)
        return True
