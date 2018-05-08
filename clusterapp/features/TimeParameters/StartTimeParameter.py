import numpy as np

from .TimeParameter import TimeParameter
from .__init__ import *


class StartTimeParameter(TimeParameter):
    name = 'StartTime'

    """docstring for StartTimeParameter"""

    def __init__(self):
        super(TimeParameter, self).__init__()

    def measure(self, segment):
        value = segment.IndexFrom / segment.samplingRate

        segment.measures_dict[self.name] = np.round(value, DECIMAL_PLACES)
        return True
