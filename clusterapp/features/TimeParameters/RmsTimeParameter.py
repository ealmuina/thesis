import numpy as np

from clusterapp.features.utils import energy
from .TimeParameter import TimeParameter


class RmsTimeParameter(TimeParameter):
    name = 'RmsTime'

    """docstring for RmsTimeParameter"""

    def __init__(self):
        super(TimeParameter, self).__init__()

    def measure(self, segment):
        value = np.sqrt(energy(segment.data) / len(segment.data))
        segment.measures_dict[self.name] = value
        return True
