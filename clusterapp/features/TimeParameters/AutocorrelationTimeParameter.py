import numpy as np

from .TimeParameter import TimeParameter


class AutocorrelationTimeParameter(TimeParameter):
    name = 'Autocorrelation'

    """docstring for AutocorrelationTimeParameter"""

    def __init__(self):
        super(TimeParameter, self).__init__()

    def measure(self, segment, offset):
        offset = int(offset * segment.samplingRate)
        a = segment.data[:len(segment.data) - offset]
        b = segment.data[offset:]
        value = np.sum(a * b)

        segment.measures_dict[self.name] = value
        return True
