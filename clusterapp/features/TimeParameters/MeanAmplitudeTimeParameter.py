from .TimeParameter import TimeParameter
from .__init__ import *
from ..utils import *


class MeanAmplitudeTimeParameter(TimeParameter):
    name = 'MeanAmplitudeTime'

    """docstring for MeanAmplitudeTimeParameter"""

    def __init__(self):
        super(TimeParameter, self).__init__()

    def measure(self, segment, envelope='hilbert'):
        envelope = hilbertEnvelope(segment.data)
        value = np.mean(envelope)

        segment.measures_dict[self.name] = np.round(value, DECIMAL_PLACES)
        return True
