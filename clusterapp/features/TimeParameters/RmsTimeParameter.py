import numpy as np

from .TimeParameter import TimeParameter
from .__init__ import *


class RmsTimeParameter(TimeParameter):
    name = 'RmsTime'

    """docstring for RmsTimeParameter"""

    def __init__(self):
        super(TimeParameter, self).__init__()

    def measure(self, segment):
        value = np.sqrt(np.sum(segment.data ** 2) / len(segment.data))
        segment.measures_dict[self.name] = np.round(value, DECIMAL_PLACES)
        return True
