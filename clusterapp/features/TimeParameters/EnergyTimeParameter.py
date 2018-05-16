from clusterapp.features.utils import *
from .TimeParameter import TimeParameter


class EnergyTimeParameter(TimeParameter):
    name = 'Time Energy'

    """docstring for EnergyTimeParameter"""

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

        value = np.sum(np.square(segment.envelope))

        segment.measures_dict[self.name] = value
        return True
