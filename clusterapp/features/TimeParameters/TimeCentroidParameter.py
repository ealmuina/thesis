from .TimeParameter import TimeParameter
from .__init__ import *
from ..utils import *


class TimeCentroidParameter(TimeParameter):
    name = 'TimeCentroid'

    """docstring for TimeCentroidParameter"""

    def __init__(self):
        super(TimeParameter, self).__init__()

    def measure(self, segment, envelope='hilbert'):
        envelope = hilbertEnvelope(segment.data)
        indexes = np.array(range(1, len(envelope) + 1))
        value = (np.sum(indexes * envelope) + segment.IndexFrom) / np.sum(envelope) / segment.samplingRate

        segment.measures_dict[self.name] = np.round(value, DECIMAL_PLACES)
        return True
