import essentia
import essentia.standard as es
import numpy as np
import scipy.signal

FS = 44100


class Audio:
    def __init__(self, path):
        self.audio = es.MonoLoader(filename=str(path))()
        self.name = path.name
        self.memo = {}

        w = es.Windowing(type='hann')
        spectrum = es.Spectrum()
        pitch = es.PitchYin(frameSize=1024)
        spectral_peaks = es.SpectralPeaks(minFrequency=1e-5)
        harmonic_peaks = es.HarmonicPeaks()

        pool = essentia.Pool()

        for frame in es.FrameGenerator(self.audio, frameSize=1024, hopSize=512, startFromZero=True):
            spec = spectrum(w(frame))
            pool.add('spectrum', spec)
            f0, _ = pitch(frame)
            freq_ep, amp_ep = spectral_peaks(spec)
            freq_hp, amp_hp = harmonic_peaks(freq_ep, amp_ep, f0)
            pool.add('harmonic_peaks_freq', freq_hp)
            pool.add('harmonic_peaks_amp', amp_hp)

        self.spectrum = pool['spectrum'].T
        self.harmonic_peaks_freq = pool['harmonic_peaks_freq']
        self.harmonic_peaks_amp = pool['harmonic_peaks_amp']

    def _get_harmonic_feature(self, name, func):
        if name not in self.memo:
            pool = essentia.Pool()
            for hp_freq, hp_amp in zip(self.harmonic_peaks_freq, self.harmonic_peaks_amp):
                pool.add('result', func(hp_freq, hp_amp))
            self.memo[name] = pool['result']
        return self.memo[name]

    def _get_spectral_feature(self, name, func):
        if name not in self.memo:
            self.memo[name] = np.apply_along_axis(func, 0, self.spectrum)
        return self.memo[name]

    def _get_temporal_feature(self, name, func):
        if name not in self.memo:
            pool = essentia.Pool()
            for frame in es.FrameGenerator(self.audio, frameSize=1024, hopSize=1024, startFromZero=True):
                pool.add('result', func(frame))
            self.memo['name'] = pool['result']
        return self.memo['name']

    @property
    def audio_power(self):
        return self._get_temporal_feature('audio_power', es.InstantPower())

    @property
    def audio_waveform(self):
        return self._get_temporal_feature(
            'audio_waveform',
            lambda frame: essentia.array([frame.min(), frame.max()])
        )

    @property
    def bandwidth(self):
        if 'bandwidth' not in self.memo:
            self.memo['bandwidth'] = np.array([self.max_freq - self.min_freq])
        return self.memo['bandwidth']

    @property
    def fundamental_freq(self):
        return self._get_spectral_feature(
            'fundamental_freq',
            lambda spec: es.PitchYin(frameSize=1024)(spec)
        )

    @property
    def inharmonicity(self):
        return self._get_harmonic_feature(
            'inharmonicity',
            es.Inharmonicity()
        )

    @property
    def max_freq(self):
        return self._get_spectral_feature(
            'max_freq',
            lambda spec: _first_over_threshold(np.flip(spec, 0)),
        )

    @property
    def mfcc(self):
        if 'mfcc' not in self.memo:
            mfcc = es.MFCC(inputSize=513)
            self.memo['mfcc'] = np.apply_along_axis(lambda spec: mfcc(spec)[1], 0, self.spectrum)
        return self.memo['mfcc']

    @property
    def min_freq(self):
        return self._get_spectral_feature('min_freq', _first_over_threshold)

    @property
    def oer(self):
        return self._get_harmonic_feature(
            'OER',
            es.OddToEvenHarmonicEnergyRatio()
        )

    @property
    def peak_ampl(self):
        if 'peak_ampl' not in self.memo:
            peak_ampl = self.spectrum.max(0)
            self.memo['peak_ampl'] = np.array([peak_ampl.mean()])
        return self.memo['peak_ampl']

    @property
    def peak_freq(self):
        return self._get_spectral_feature('peak_freq', es.MaxMagFreq())

    @property
    def spectral_centroid(self):
        return self._get_spectral_feature('spectral_centroid', es.Centroid())

    @property
    def spectral_flatness(self):
        return self._get_spectral_feature('spectral_flatness', es.Flatness())

    @property
    def spectral_flux(self):
        return self._get_spectral_feature('spectral_flux', es.Flux())

    @property
    def spectral_roll_off(self):
        return self._get_spectral_feature('spectral_roll_off', es.RollOff())

    @property
    def spectral_spread(self):
        return self._get_spectral_feature(
            'spectral_spread',
            lambda spec: es.CentralMoments()(spec)[2]
        )

    @property
    def tristimulus(self):
        return self._get_harmonic_feature(
            'tristimulus',
            es.Tristimulus()
        )

    @property
    def zcr(self):
        return self._get_temporal_feature('audio_power', es.ZeroCrossingRate())


def _decibels(a1, a2):
    return 10 * np.log10(a1 / a2 + 1e-8)


def _delta(data, width=9, order=1, axis=-1, mode='interp', **kwargs):
    r"""Compute delta features: local estimate of the derivative
    of the input data along the selected axis.

    Delta features are computed Savitsky-Golay filtering.

    Parameters
    ----------
    data      : np.ndarray
        the input data matrix (eg, spectrogram)

    width     : int, positive, odd [scalar]
        Number of frames over which to compute the delta features.
        Cannot exceed the length of `data` along the specified axis.
        If `mode='interp'`, then `width` must be at least `data.shape[axis]`.

    order     : int > 0 [scalar]
        the order of the difference operator.
        1 for first derivative, 2 for second, etc.

    axis      : int [scalar]
        the axis along which to compute deltas.
        Default is -1 (columns).

    mode : str, {'interp', 'nearest', 'mirror', 'constant', 'wrap'}
        Padding mode for estimating differences at the boundaries.

    kwargs : additional keyword arguments
        See `scipy.signal.savgol_filter`

    Returns
    -------
    delta_data   : np.ndarray [shape=(d, t)]
        delta matrix of `data` at specified order

    See Also
    --------
    scipy.signal.savgol_filter

    """
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
