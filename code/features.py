import essentia
import essentia.standard as es
import numpy as np
import pylab as pl
from matplotlib import collections as mc
from matplotlib.ticker import MaxNLocator, FuncFormatter


def delta(c, step=2):
    result = []
    for t in range(step, c.shape[1] - step):
        num = np.zeros(c.shape[0])
        den = 0
        for n in range(1, step + 1):
            num += n * (c[:, t + n] - c[:, t - n])
            den += 2 * n ** 2
        result.append(num / den)
    return np.array(result).T


def plot_temporal_descriptors(audio):
    awf = []
    pool = essentia.Pool()
    power = es.InstantPower()

    for i, frame in enumerate(es.FrameGenerator(audio, frameSize=1024, hopSize=1024, startFromZero=True)):
        awf.append([(i, frame.min()), (i, frame.max())])
        pool.add('AP', power(frame))

    awf = essentia.array(awf)

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
    pool = essentia.Pool()
    w = es.Windowing(type='hann')
    spectrum = es.Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum

    centroid = es.Centroid()
    moments = es.CentralMoments()
    roll_off = es.RollOff()
    flux = es.Flux()
    flatness = es.Flatness()

    for frame in es.FrameGenerator(audio, frameSize=1024, hopSize=512, startFromZero=True):
        spec = spectrum(w(frame))
        pool.add('SC', centroid(spec))
        pool.add('SS', moments(spec)[2])
        pool.add('SRO', roll_off(spec))
        pool.add('SFX', flux(spec))
        pool.add('SF', flatness(spec))

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
    pool = essentia.Pool()
    w = es.Windowing(type='hann')
    spectrum = es.Spectrum()

    pitch = es.PitchYin(frameSize=1024)
    spectral_peaks = es.SpectralPeaks(minFrequency=1e-5)
    harmonic_peaks = es.HarmonicPeaks()
    inharmonicity = es.Inharmonicity()
    oer = es.OddToEvenHarmonicEnergyRatio()
    tristimulus = es.Tristimulus()

    for frame in es.FrameGenerator(audio, frameSize=1024, hopSize=512, startFromZero=True):
        spec = spectrum(w(frame))
        f0, _ = pitch(frame)
        freq_ep, amp_ep = spectral_peaks(spec)
        freq_hp, amp_hp = harmonic_peaks(freq_ep, amp_ep, f0)

        pool.add('INH', inharmonicity(freq_hp, amp_hp))
        pool.add('OER', oer(freq_hp, amp_hp))
        pool.add('T', tristimulus(freq_hp, amp_hp))

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
    specs = []
    mfccs = []
    w = es.Windowing(type='hann')
    spectrum = es.Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum
    mfcc = es.MFCC(inputSize=513)

    for frame in es.FrameGenerator(audio, frameSize=1024, hopSize=512, startFromZero=True):
        spec = spectrum(w(frame))
        _, mfcc_coeffs = mfcc(spec)
        specs.append(spec)
        mfccs.append(mfcc_coeffs)

    mfccs = essentia.array(mfccs).T
    # d_mfccs = delta(mfccs)
    # dd_mfccs = delta(d_mfccs)

    fig, ax = pl.subplots(1, 1, figsize=(12, 4))
    fig.subplots_adjust(left=0.05, right=0.97)

    ax.imshow(mfccs[1:, :], aspect='auto', origin='lower', interpolation='none')
    ax.set_title('MFCC')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: int(x + 1)))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    fig.show()


def main():
    audio = es.MonoLoader(filename='../sounds/sheep.wav')()

    plot_temporal_descriptors(audio)
    plot_spectral_descriptors(audio)
    plot_harmonic_descriptors(audio)
    plot_mfccs(audio)


if __name__ == '__main__':
    main()
