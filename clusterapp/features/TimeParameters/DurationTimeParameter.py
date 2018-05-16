from .TimeParameter import TimeParameter


class DurationTimeParameter(TimeParameter):
    name = 'DurationTime'

    """docstring for DurationTimeParameter"""

    def __init__(self):
        super(TimeParameter, self).__init__()

    def measure(self, segment):
        value = (segment.IndexTo - segment.IndexFrom) / segment.samplingRate

        segment.measures_dict[self.name] = value
        return True
