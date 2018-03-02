from math import ceil

import essentia
import pylab as pl
from essentia.standard import MonoLoader, Windowing, Spectrum, FrameGenerator, InstantPower, FrequencyBands, Centroid, \
    CentralMoments, Flatness
from matplotlib import collections as mc


def get_freq_bands(lo_edge, hi_edge, band_resolution, overlap=0.0):
    freq_bands = [0]
    current = lo_edge
    b = 1
    while current <= hi_edge:
        freq_bands.append(current)
        current = (1 + overlap) * lo_edge * 2 ** (b * band_resolution)
        b += 1
    freq_bands.append(22050)
    return freq_bands


def bands_flatness(x):
    flatness = Flatness()
    freq_bands = get_freq_bands(250, 16000, 0.25, overlap=0.05)

    result = []
    kl = 0
    for kh in sorted(set(map(lambda f: ceil(1024 * f / 44100 + 1e-6), freq_bands[1:]))):
        result.append(flatness(x[kl:kh]))
        kl = kh
    return result


def main():
    audio = MonoLoader(filename='../sounds/sheep.wav')()

    awf = []
    pool = essentia.Pool()

    # BASIC DESCRIPTORS
    power = InstantPower()

    for i, frame in enumerate(FrameGenerator(audio, frameSize=1024, hopSize=1024, startFromZero=True)):
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

    # BASIC SPECTRAL DESCRIPTORS

    freq_bands = get_freq_bands(62.5, 16000, 2 ** -2)

    w = Windowing(type='hann')
    spectrum = Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum
    ase = FrequencyBands(frequencyBands=freq_bands)
    centroid = Centroid(range=len(freq_bands))
    moments = CentralMoments()

    for frame in FrameGenerator(audio, frameSize=1024, hopSize=512, startFromZero=True):
        spec = spectrum(w(frame))
        log_spec = ase(spec)
        pool.add('ASE', log_spec)
        pool.add('ASC', centroid(log_spec))
        pool.add('ASS', moments(log_spec)[2])

    for frame in FrameGenerator(audio, frameSize=1024, hopSize=1024, startFromZero=True):
        spec = spectrum(w(frame))
        pool.add('ASF', bands_flatness(spec))

    fig, ax = pl.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.06, top=0.93)

    ax[0, 0].imshow(pool['ASE'].T[:, :], aspect='auto', origin='lower', interpolation='none')
    ax[0, 0].set_title('Audio Spectrum Envelope (ASE)')

    ax[0, 1].plot(pool['ASC'])
    ax[0, 1].set_title('Audio Spectrum Centroid (ASC)')

    ax[1, 0].plot(pool['ASS'])
    ax[1, 0].set_title('Audio Spectrum Spread (ASS)')

    ax[1, 1].imshow(pool['ASF'].T[:, :], aspect='auto', origin='lower', interpolation='none')
    ax[1, 1].set_title('Audio Spectrum Flatness (ASF)')

    fig.show()


if __name__ == '__main__':
    main()
