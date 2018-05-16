from librosa.feature import mfcc

from .FreqParameters.BandwidthFreqParameter import BandwidthFreqParameter
from .FreqParameters.EnergyFreqParameter import EnergyFreqParameter
from .FreqParameters.EntropyFreqParameter import EntropyFreqParameter
from .FreqParameters.FluxFreqParameter import FluxFreqParameter
from .FreqParameters.MaxFreqParameter import MaxFreqParameter
from .FreqParameters.MinFreqParameter import MinFreqParameter
from .FreqParameters.PeakAmpFreqParameter import PeakAmpFreqParameter
from .FreqParameters.PeakFreqParameter import PeakFreqParameter
from .FreqParameters.PeaksAboveFreqParameter import PeaksAboveFreqParameter
from .FreqParameters.RmsFreqParameter import RmsFreqParameter
from .FreqParameters.RollOffFreqParameter import RollOffFreqParameter
from .FreqParameters.ShannonEntropyFreqParameter import ShannonEntropyFreqParameter
from .FreqParameters.SpectralCentroidParameter import SpectralCentroidParameter
from .TimeParameters.AutocorrelationTimeParameter import AutocorrelationTimeParameter
from .TimeParameters.DistanceToMaxTimeParameter import DistanceToMaxTimeParameter
from .TimeParameters.DurationTimeParameter import DurationTimeParameter
from .TimeParameters.EndTimeParameter import EndTimeParameter
from .TimeParameters.EnergyTimeParameter import EnergyTimeParameter
from .TimeParameters.MeanAmplitudeTimeParameter import MeanAmplitudeTimeParameter
from .TimeParameters.PeakToPeakTimeParameter import PeakToPeakTimeParameter
from .TimeParameters.RmsTimeParameter import RmsTimeParameter
from .TimeParameters.StartTimeParameter import StartTimeParameter
from .TimeParameters.StdAmplitudeTimeParameter import StdAmplitudeTimeParameter
from .TimeParameters.TimeCentroidParameter import TimeCentroidParameter
from .TimeParameters.VarianceAmplitudeTimeParameter import VarianceAmplitudeTimeParameter
from .TimeParameters.ZcrTimeParameter import ZcrTimeParameter
from .segment import Segment
from .signal import Signal
from .utils import *

"""Time Parameters Instances"""
std_t = StdAmplitudeTimeParameter()
var_t = VarianceAmplitudeTimeParameter()
mean_t = MeanAmplitudeTimeParameter()
cent_t = TimeCentroidParameter()
energy_t = EnergyTimeParameter()

corr_t = AutocorrelationTimeParameter()

disttomax_t = DistanceToMaxTimeParameter()
start_t = StartTimeParameter()
end_t = EndTimeParameter()
dur_t = DurationTimeParameter()
rms_t = RmsTimeParameter()
ptp_t = PeakToPeakTimeParameter()
zcr_t = ZcrTimeParameter()

"""Spectral Parameters Instances"""
max_f = MaxFreqParameter(total=True)
min_f = MinFreqParameter(total=True)
bandwidth_f = BandwidthFreqParameter(total=True)
peaks_above_f = PeaksAboveFreqParameter()

energy_f = EnergyFreqParameter()
rms_f = RmsFreqParameter()
s_centroid = SpectralCentroidParameter()
peak_f = PeakFreqParameter()
peak_amp_f = PeakAmpFreqParameter()
s_entropy = ShannonEntropyFreqParameter()
entropy_f = EntropyFreqParameter()
roll_off_f = RollOffFreqParameter()
flux_f = FluxFreqParameter()


class Audio:
    def __init__(self, path, string_path=False):
        file, self.fs = load_file(str(path) if string_path else path)
        signal = Signal(file, self.fs)
        signal.set_window('boxcar')

        self.segment = Segment(signal, 0, len(signal.data) - 1)
        self.name = path.name

        self._build_temporal_features()
        self._build_spectral_features()
        self._build_mfcc()

    def __getattr__(self, item):
        if item in self.segment.measures_dict:
            return self.segment.measures_dict[item]
        raise AttributeError(item)

    def _build_mfcc(self):
        coeffs = mfcc(self.segment.data, self.fs, n_mfcc=13)[1:]
        self.segment.measures_dict['MFCC'] = coeffs

    def _build_spectral_features(self):
        locations = ['start', 'end', 'centre', 'max', 'max_amp']

        for l in locations:
            max_f.measure(self.segment, threshold=-20, location=l)
            min_f.measure(self.segment, threshold=-20, location=l)
            bandwidth_f.measure(self.segment, threshold=-20, location=l)
            peaks_above_f.measure(self.segment, threshold=-20, location=l)

            energy_f.measure(self.segment, location=l)
            rms_f.measure(self.segment, location=l)
            s_centroid.measure(self.segment, location=l)
            s_entropy.measure(self.segment, location=l)
            entropy_f.measure(self.segment, location=l)
            peak_f.measure(self.segment, location=l)
            peak_amp_f.measure(self.segment, location=l)
            roll_off_f.measure(self.segment, location=l, cutoff=.95)
            flux_f.measure(self.segment, location=l)

    def _build_temporal_features(self):
        corr_t.measure(self.segment, offset=0)

        std_t.measure(self.segment, envelope='hilbert')
        var_t.measure(self.segment, envelope='hilbert')
        mean_t.measure(self.segment, envelope='hilbert')
        cent_t.measure(self.segment, envelope='hilbert')
        energy_t.measure(self.segment, envelope='hilbert')

        zcr_t.measure(self.segment)
        dur_t.measure(self.segment)
        rms_t.measure(self.segment)
        ptp_t.measure(self.segment)
        start_t.measure(self.segment)
        end_t.measure(self.segment)
        disttomax_t.measure(self.segment)
