import numpy as np

from .TimeParameter import TimeParameter


class DistanceToMaxTimeParameter(TimeParameter):
    name = 'DistanceToMaxTime'

    """
    Distance to the max amplitude of the time 
    signal relative to the start time
    """

    def __init__(self):
        super(TimeParameter, self).__init__()

    def measure(self, segment):
        max_index = np.argmax(segment.data)
        value = max_index / segment.samplingRate
        segment.measures_dict[self.name] = value
        return True
