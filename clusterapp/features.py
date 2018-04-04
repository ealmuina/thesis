import essentia
import essentia.standard as es
import numpy as np
import scipy.signal

FS = 44100

w = es.Windowing(type='hann')
spectrum = es.Spectrum()
centroid = es.Centroid()
moments = es.CentralMoments()

# Temporal descriptors
power = es.InstantPower()
log_attack_time = es.LogAttackTime()
effective_duration = es.EffectiveDuration()
auto_correlation = es.AutoCorrelation()
zero_crossing_rate = es.ZeroCrossingRate()

# Spectral descriptors
peak_freq = es.MaxMagFreq()
roll_off = es.RollOff()
flux = es.Flux()
flatness = es.Flatness()

# Harmonic descriptors
pitch = es.PitchYin(frameSize=1024)
spectral_peaks = es.SpectralPeaks(minFrequency=1e-5)
harmonic_peaks = es.HarmonicPeaks()
inharmonicity = es.Inharmonicity()
oer = es.OddToEvenHarmonicEnergyRatio()
tristimulus = es.Tristimulus()

# MFCC
mfcc = es.MFCC(inputSize=513)


class Audio:
    def __init__(self, path):
        self.audio = es.MonoLoader(filename=str(path))()
        self.name = path.name
        self.pool = essentia.Pool()

        self._build_temporal_features()
        self._build_spectral_features()
        self._build_harmonic_features()
        self._build_mfcc()

        self._features = {
            'audio_correlation': 'AC',
            'audio_power': 'AP',
            'audio_waveform': 'AWF',
            'bandwidth': 'SB',
            'effective_duration': 'ED',
            'fundamental_freq': 'F0',
            'inharmonicity': 'INH',
            'log_attack_time': 'LAT',
            'max_freq': 'FMax',
            'mfcc': 'MFCC',
            'min_freq': 'FMin',
            'oer': 'OER',
            'peak_ampl': 'PA',
            'peak_freq': 'PF',
            'spectral_centroid': 'SC',
            'spectral_flatness': 'SF',
            'spectral_flux': 'SFX',
            'spectral_roll_off': 'SRO',
            'spectral_spread': 'SS',
            'temporal_centroid': 'TC',
            'tristimulus': 'T',
            'zcr': 'ZCR'
        }

    def __getattr__(self, item):
        if item in self._features:
            key = self._features[item]
            return self.pool[key]
        raise AttributeError(item)

    def _build_harmonic_features(self):
        for frame in es.FrameGenerator(self.audio, frameSize=1024, hopSize=512, startFromZero=True):
            spec = spectrum(w(frame))
            f0, _ = pitch(frame)
            freq_ep, amp_ep = spectral_peaks(spec)
            freq_hp, amp_hp = harmonic_peaks(freq_ep, amp_ep, f0)

            self.pool.add('F0', f0)
            self.pool.add('INH', inharmonicity(freq_hp, amp_hp))
            self.pool.add('OER', oer(freq_hp, amp_hp))
            self.pool.add('T', tristimulus(freq_hp, amp_hp))

    def _build_mfcc(self):
        for frame in es.FrameGenerator(self.audio, frameSize=1024, hopSize=512, startFromZero=True):
            spec = spectrum(w(frame))
            _, mfcc_coeffs = mfcc(spec)
            self.pool.add('MFCC', mfcc_coeffs)

    def _build_spectral_features(self):
        for frame in es.FrameGenerator(self.audio, frameSize=1024, hopSize=512, startFromZero=True):
            spec = spectrum(w(frame))
            fmin = _first_over_threshold(spec)
            fmax = _first_over_threshold(np.flip(spec, 0))

            self.pool.add('PF', peak_freq(spec))
            self.pool.add('PA', spec.max())
            self.pool.add('FMin', fmin)
            self.pool.add('FMax', fmax)
            self.pool.add('SB', fmax - fmin)

            self.pool.add('SC', centroid(spec))
            self.pool.add('SS', moments(spec)[2])
            self.pool.add('SRO', roll_off(spec))
            self.pool.add('SFX', flux(spec))
            self.pool.add('SF', flatness(spec))

    def _build_temporal_features(self):
        self.pool.add('LAT', log_attack_time(self.audio)[0])
        self.pool.add('TC', centroid(self.audio))
        self.pool.add('ED', effective_duration(self.audio))
        self.pool.add('AC', auto_correlation(self.audio))

        for i, frame in enumerate(es.FrameGenerator(self.audio, frameSize=1024, hopSize=1024, startFromZero=True)):
            self.pool.add('AP', power(frame))
            self.pool.add('AWF', essentia.array([frame.min(), frame.max()]))
            self.pool.add('ZCR', zero_crossing_rate(frame))


def _decibels(a1, a2):
    return 10 * np.log10(a1 / a2 + 1e-8)


def _delta(data, width=9, order=1, axis=-1, mode='interp', **kwargs):
    data = np.atleast_1d(data)

    if mode == 'interp' and width > data.shape[axis]:
        raise ValueError("when mode='interp', width={} "
                         "cannot exceed data.shape[axis]={}".format(width, data.shape[axis]))

    if width < 3 or np.mod(width, 2) != 1:
        raise ValueError('width must be an odd integer >= 3')

    if order <= 0 or not isinstance(order, int):
        raise ValueError('order must be a positive integer')

    kwargs.pop('deriv', None)
    kwargs.setdefault('polyorder', order)
    return scipy.signal.savgol_filter(data, width, deriv=order, axis=axis, mode=mode, **kwargs)


def _first_over_threshold(spectrum, threshold=-20):
    peak_ampl = spectrum.max()
    for k in range(spectrum.shape[0]):
        d = _decibels(spectrum[k], peak_ampl)
        if d >= threshold:
            return k * (FS / 1024)
