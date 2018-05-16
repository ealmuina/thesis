import numpy as np

from clusterapp.features.utils import get_location, to_db
from .FreqParameter import FreqParameter


class PeakAmpFreqParameter(FreqParameter):
    name = 'PeakAmpFreq'

    """docstring for MaxFreqParameter"""

    def __init__(self):
        super(FreqParameter, self).__init__()

    def measure(self, segment, location='center'):
        if segment.peaks_values is None:
            segment.compute_peaks()

        if location is None:
            return self.__measure_spectrum(segment)

        index_frame = get_location(segment, location)
        j = segment.peaks_values[index_frame]
        db_reference = segment.signal.db_reference
        segment.measures_dict[self.name + '(' + location + ')'] = to_db(j, db_reference)
        return True

    def __measure_spectrum(self, segment):
        if segment.peaks_values is None:
            segment.compute_peaks()

        value = np.max(segment.spectrum)
        segment.measures_dict[self.name + '(total)'] = to_db(value)
        return True
