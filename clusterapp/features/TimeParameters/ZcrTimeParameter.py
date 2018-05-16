import numpy as np

from .TimeParameter import TimeParameter


class ZcrTimeParameter(TimeParameter):
    name = 'ZeroCrossingRate'

    """docstring for ZcrTimeParameter"""

    def __init__(self):
        super(TimeParameter, self).__init__()

    def measure(self, segment):
        data = segment.data[segment.data != 0]
        value = np.sum(data[:-1] * data[1:] < 0) / len(segment.data)

        segment.measures_dict[self.name] = value
        return True
