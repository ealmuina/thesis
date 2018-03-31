import essentia
import essentia.standard as es
import numpy as np
import pylab as pl
import scipy.signal
import seaborn as sns
from matplotlib import collections as mc
from matplotlib.ticker import MaxNLocator, FuncFormatter

FS = 44100


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


class FeaturesExtractor:
    def __init__(self):
        self.w = es.Windowing(type='hann')
        self.spectrum = es.Spectrum()
        self.centroid = es.Centroid()
        self.moments = es.CentralMoments()

        # Temporal descriptors
        self.power = es.InstantPower()
        self.log_attack_time = es.LogAttackTime()
        self.effective_duration = es.EffectiveDuration()
        self.auto_correlation = es.AutoCorrelation()
        self.zero_crossing_rate = es.ZeroCrossingRate()

        # Spectral descriptors
        self.peak_freq = es.MaxMagFreq()
        self.roll_off = es.RollOff()
        self.flux = es.Flux()
        self.flatness = es.Flatness()

        # Harmonic descriptors
        self.pitch = es.PitchYin(frameSize=1024)
        self.spectral_peaks = es.SpectralPeaks(minFrequency=1e-5)
        self.harmonic_peaks = es.HarmonicPeaks()
        self.inharmonicity = es.Inharmonicity()
        self.oer = es.OddToEvenHarmonicEnergyRatio()
        self.tristimulus = es.Tristimulus()

        # MFCC
        self.mfcc = es.MFCC(inputSize=513)

    @staticmethod
    def _min_freq(spec, pa, frame_size=1024, threshold=-20):
        for k in range(spec.shape[0]):
            d = _decibels(spec[k], pa)
            if d >= threshold:
                return k * (FS / frame_size)

    @staticmethod
    def _max_freq(spec, pa, frame_size=1024, threshold=-20):
        for k in range(spec.shape[0] - 1, -1, -1):
            d = _decibels(spec[k], pa)
            if d >= threshold:
                return k * (FS / frame_size)

    def full_features(self, audio):
        pool_temp = self.temporal_descriptors(audio)
        pool_spec = self.spectral_descriptors(audio)
        pool_harm = self.harmonic_descriptors(audio)
        mfccs, d_mfccs, dd_mfccs = self.mfcc_descriptors(audio)

        return np.array([
            # Temporal features [0-7]
            pool_temp['LAT'],
            *pool_temp['AWF'].mean(0),  # 2 values
            pool_temp['AP'].mean(),
            pool_temp['TC'],
            pool_temp['ED'],
            pool_temp['AC'].mean(),
            pool_temp['ZCR'].mean(),

            # Spectral features [8-17]
            pool_spec['PF'].mean(),
            pool_spec['PA'].mean(),
            pool_spec['FMin'].mean(),
            pool_spec['FMax'].mean(),
            pool_spec['SB'].mean(),
            pool_spec['SC'].mean(),
            pool_spec['SS'].mean(),
            pool_spec['SRO'].mean(),
            pool_spec['SFX'].mean(),
            pool_spec['SF'].mean(),

            # Harmonic features [18-23]
            pool_harm['F0'].mean(),
            pool_harm['INH'].mean(),
            pool_harm['OER'].mean(),
            *pool_harm['T'].mean(0),  # 3 values

            # MFFCCs [24-62]
            *mfccs.mean(1),  # 13 values
            *d_mfccs.mean(1),  # 13 values
            *dd_mfccs.mean(1)  # 13 values
        ])

    def harmonic_descriptors(self, audio):
        pool = essentia.Pool()

        for frame in es.FrameGenerator(audio, frameSize=1024, hopSize=512, startFromZero=True):
            spec = self.spectrum(self.w(frame))
            f0, _ = self.pitch(frame)
            freq_ep, amp_ep = self.spectral_peaks(spec)
            freq_hp, amp_hp = self.harmonic_peaks(freq_ep, amp_ep, f0)

            pool.add('F0', f0)
            pool.add('INH', self.inharmonicity(freq_hp, amp_hp))
            pool.add('OER', self.oer(freq_hp, amp_hp))
            pool.add('T', self.tristimulus(freq_hp, amp_hp))

        return pool

    def mfcc_descriptors(self, audio):
        mfccs = []

        for frame in es.FrameGenerator(audio, frameSize=1024, hopSize=512, startFromZero=True):
            spec = self.spectrum(self.w(frame))
            _, mfcc_coeffs = self.mfcc(spec)
            mfccs.append(mfcc_coeffs)

        mfccs = essentia.array(mfccs).T
        d_mfccs = _delta(mfccs, mode='nearest')
        dd_mfccs = _delta(d_mfccs, mode='nearest')

        return mfccs, d_mfccs, dd_mfccs

    def spectral_descriptors(self, audio):
        pool = essentia.Pool()

        for frame in es.FrameGenerator(audio, frameSize=1024, hopSize=512, startFromZero=True):
            spec = self.spectrum(self.w(frame))
            pa = spec.max()
            fmin = self._min_freq(spec, pa)
            fmax = self._max_freq(spec, pa)

            pool.add('PF', self.peak_freq(spec))
            pool.add('PA', pa)
            pool.add('FMin', fmin)
            pool.add('FMax', fmax)
            pool.add('SB', fmax - fmin)

            pool.add('SC', self.centroid(spec))
            pool.add('SS', self.moments(spec)[2])
            pool.add('SRO', self.roll_off(spec))
            pool.add('SFX', self.flux(spec))
            pool.add('SF', self.flatness(spec))

        return pool

    def temporal_descriptors(self, audio):
        pool = essentia.Pool()

        pool.add('LAT', self.log_attack_time(audio)[0])
        pool.add('TC', self.centroid(audio))
        pool.add('ED', self.effective_duration(audio))
        pool.add('AC', self.auto_correlation(audio))

        for i, frame in enumerate(es.FrameGenerator(audio, frameSize=1024, hopSize=1024, startFromZero=True)):
            pool.add('AP', self.power(frame))
            pool.add('AWF', essentia.array([frame.min(), frame.max()]))
            pool.add('ZCR', self.zero_crossing_rate(frame))

        return pool


def plot_temporal_descriptors(audio):
    extractor = FeaturesExtractor()
    pool = extractor.temporal_descriptors(audio)

    awf = [[(i, f[0]), (i, f[1])] for i, f in enumerate(pool['AWF'])]

    fig, (ax1, ax2) = pl.subplots(1, 2, figsize=(12, 4))
    fig.subplots_adjust(left=0.07, right=0.97)

    lc = mc.LineCollection(awf, linewidths=2)
    ax1.add_collection(lc)
    ax1.autoscale()
    ax1.set_title('Audio Waveform (AWF)')

    ax2.plot(pool['AP'])
    ax2.set_title('Audio Power (AP)')

    fig.show()


def plot_spectral_descriptors(audio):
    extractor = FeaturesExtractor()
    pool = extractor.spectral_descriptors(audio)

    fig, ax = pl.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(left=0.07, right=0.97)

    ax[0, 0].plot(pool['SC'], label='SC')
    ax[0, 0].plot(pool['SS'], label='SS')
    ax[0, 0].legend()
    ax[0, 0].set_title('Spectral Centroid (SC) y Spectral Spread (SS)')

    ax[0, 1].plot(pool['SRO'])
    ax[0, 1].set_title('Spectral Roll-off (SRO)')

    ax[1, 0].plot(pool['SFX'])
    ax[1, 0].set_title('Spectral Flux (SFX)')

    ax[1, 1].plot(pool['SF'])
    ax[1, 1].set_title('Spectral Flatness (SF)')

    fig.show()


def plot_harmonic_descriptors(audio):
    extractor = FeaturesExtractor()
    pool = extractor.harmonic_descriptors(audio)

    fig, (ax1, ax2, ax3) = pl.subplots(1, 3, figsize=(12, 4))
    fig.subplots_adjust(left=0.055, right=0.98)

    ax1.plot(pool['INH'])
    ax1.set_title('Inharmonicity (INH)')

    ax2.plot(pool['OER'])
    ax2.set_title('Odd to Even Harmonic Energy Ratio (OER)')

    ax3.imshow(pool['T'].T[:, :], aspect='auto', origin='lower', interpolation='none')
    ax3.set_title('Tristimulus')
    ax3.yaxis.set_major_locator(MaxNLocator(integer=True))

    fig.show()


def plot_mfccs(audio):
    extractor = FeaturesExtractor()
    mfccs, _, _ = extractor.mfcc_descriptors(audio)

    fig, ax = pl.subplots(1, 1, figsize=(12, 4))
    fig.subplots_adjust(left=0.05, right=0.97)

    ax.imshow(mfccs[1:, :], aspect='auto', origin='lower', interpolation='none')
    ax.set_title('MFCC')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: int(x + 1)))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    fig.show()


def main():
    sns.set()
    sns.set_style('white')

    audio = es.MonoLoader(filename='../sounds/sheep.wav')()

    plot_temporal_descriptors(audio)
    plot_spectral_descriptors(audio)
    plot_harmonic_descriptors(audio)
    plot_mfccs(audio)


if __name__ == '__main__':
    main()
