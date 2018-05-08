import numpy as np

from .TimeParameter import TimeParameter
from .__init__ import *


class DurationTimeParameter(TimeParameter):
    name = 'DurationTime'

    """docstring for DurationTimeParameter"""

    def __init__(self):
        super(TimeParameter, self).__init__()

    def measure(self, segment):
        value = (segment.IndexTo - segment.IndexFrom) / segment.samplingRate

        segment.measures_dict[self.name] = np.round(value, DECIMAL_PLACES)
        return True
