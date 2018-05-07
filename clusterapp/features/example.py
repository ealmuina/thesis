from .FreqParameters.BandwidthFreqParameter import BandwidthFreqParameter
from .FreqParameters.EntropyFreqParameter import EntropyFreqParameter
from .FreqParameters.MaxFreqParameter import MaxFreqParameter
from .FreqParameters.MinFreqParameter import MinFreqParameter
from .FreqParameters.PeakAmpFreqParameter import PeakAmpFreqParameter
from .FreqParameters.PeakFreqParameter import PeakFreqParameter
from .FreqParameters.PeaksAboveFreqParameter import PeaksAboveFreqParameter
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
from .segment import Segment, Signal
from .utils import *

"""Load wav file"""
filename = 'whale.wav'
x, fs = load_file(filename)

# Signal instance
signal = Signal(x, fs, filter=None)
# Setting spectrogram windows
signal.set_window('flattop')
# Segment instance 
a = int(fs * 0.136)
b = int(fs * 1.459)
segment = Segment(signal, a, b)

print(filename)
print(fs)
print()

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

peak_f = PeakFreqParameter()
peak_amp_f = PeakAmpFreqParameter()
entropy_f = EntropyFreqParameter()

"""Time Parameters Measures"""
corr_t.measure(segment, offset=0.1)

std_t.measure(segment, envelope='hilbert')
var_t.measure(segment, envelope='hilbert')
mean_t.measure(segment, envelope='hilbert')
cent_t.measure(segment, envelope='hilbert')
energy_t.measure(segment, envelope='hilbert')

zcr_t.measure(segment)
dur_t.measure(segment)
rms_t.measure(segment)
ptp_t.measure(segment)
start_t.measure(segment)
end_t.measure(segment)
disttomax_t.measure(segment)

"""Spectral Parameters Measures"""
locations = ['start', 'end', 'center', 'max', 'max_amp']

for l in locations:
    max_f.measure(segment, threshold=-20, location=l)
    min_f.measure(segment, threshold=-20, location=l)
    bandwidth_f.measure(segment, threshold=-20, location=l)
    peaks_above_f.measure(segment, threshold=-20, location=l)

    entropy_f.measure(segment, location=l)
    peak_f.measure(segment, location=l)
    peak_amp_f.measure(segment, location=l)

# Print measures dictionary
for i in segment.measures_dict:
    print(str(i) + ': ', segment.measures_dict[i])
