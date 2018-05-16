from clusterapp.features.utils import *
from .TimeParameter import TimeParameter
from .__init__ import *


class TimeCentroidParameter(TimeParameter):
    name = 'TimeCentroid'

    """docstring for TimeCentroidParameter"""

    def __init__(self):
        super(TimeParameter, self).__init__()

    def measure(self, segment, envelope='hilbert'):
        if segment.envelope is None or envelope != segment.envelope_type:
            if envelope == 'hilbert':
                segment.envelope = hilbert_envelope(segment.data)
            elif envelope == 'three_step':
                segment.envelope = three_step_envelope(segment.data)
            else:
                segment.envelope = segment.data
            segment.envelope_type = envelope

        indexes = np.array(range(1, len(segment.envelope) + 1))
        weights = np.sum(segment.envelope)
        if weights != 0.0:
            value = ((np.sum(indexes * segment.envelope) + segment.IndexFrom)
                     / weights) / segment.samplingRate
        else:
            value = 0.0

        segment.measures_dict[self.name] = np.round(value, DECIMAL_PLACES)
        return True
