from scipy.ndimage import label

from .FreqParameter import FreqParameter
from ..utils import get_location, apply_threshold


class PeaksAboveFreqParameter(FreqParameter):
    name = 'PeaksAboveFreq'

    """docstring for PeaksAboveFreqParameter"""

    def __init__(self):
        super(FreqParameter, self).__init__()

    def measure(self, segment, threshold=-20, location='center'):
        j = get_location(segment, location)
        i = segment.peaks_indexes[j]

        value = apply_threshold(segment.spec[i, j], threshold)
        _, cnt_regions = label(segment.spec[:, j] >= value)

        segment.measures_dict[self.name + '-' + location] = int(cnt_regions)
        return True
