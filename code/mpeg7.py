import math

import essentia
import numpy as np
import pylab as pl
from essentia.standard import MonoLoader, Windowing, Spectrum, FrameGenerator, InstantPower, FrequencyBands, Centroid, \
    CentralMoments, Flatness
from matplotlib import collections as mc


def get_log_freq_bands(lo_edge, hi_edge, band_resolution):
    freq_bands = [0]
    current = lo_edge
    b = 1
    while current <= hi_edge:
        freq_bands.append(current)
        current = lo_edge * 2 ** (b * band_resolution)
        b += 1
    freq_bands.append(22050)
    return freq_bands


def bands_flatness(x, lo_edge=250, hi_edge=16000, fs=44100, n=1024):
    flatness = Flatness()
    numbands = int(math.floor(4 * math.log2(hi_edge / lo_edge)))
    firstband = round(math.log2(lo_edge / 1000) * 4)
    overlap = 0.5
    gm, am = [], []

    grpsize = 1
    for k in range(1, numbands + 1):
        f_lo = lo_edge * (2 ** ((k - 1) / 4)) * (1 - overlap)
        f_hi = lo_edge * (2 ** (k / 4)) * (1 + overlap)
        i_lo = round(f_lo / (fs / n)) + 1
        i_hi = round(f_hi / (fs / n)) + 1

        # Rounding of upper index according due to coefficient grouping
        if k + firstband - 1 >= 0:  # start grouping at 1 kHz
            grpsize = 2 ** math.ceil((k + firstband) / 4)
            i_hi = round((i_hi - i_lo + 1) / grpsize) * grpsize + i_lo - 1
        else:
            grpsize = 1
        tmp = x[i_lo - 1:i_hi, :]  # ** 2  # PSD coefficients
        ncoeffs = i_hi - i_lo + 1

        if k + firstband - 1 >= 0:  # Coefficient grouping
            tmp2 = tmp[:grpsize:ncoeffs, :]
            for g in range(2, grpsize + 1):
                tmp2 += tmp[g - 1:grpsize:ncoeffs, :]
            tmp = tmp2

        # Actual calculation
        ncoeffs /= grpsize
        tmp += 1e-50  # avoid underflow for zero signals
        gm_k = np.exp(np.sum(np.log(tmp), 1) / ncoeffs)  # log processing avoids overflow
        gm.append(np.resize(gm_k, numbands))
        am_k = np.sum(tmp, 1) / ncoeffs
        am.append(np.resize(am_k, numbands))

    return np.array(gm) / np.array(am)


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

    freq_bands = get_log_freq_bands(62.5, 16000, 2 ** -2)

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
        pool.add('Spec', spec)

    asf = bands_flatness(pool['Spec'].T)

    # with open('../sounds/sheep_ASF.txt') as file:
    #     content = file.read().split()
    #     asf = np.array(list(map(float, content)))
    #     asf = asf.reshape((68, 24))

    fig, ax = pl.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.06, top=0.93)

    ax[0, 0].imshow(pool['ASE'].T, aspect='auto', origin='lower', interpolation='none')
    ax[0, 0].set_title('Audio Spectrum Envelope (ASE)')

    ax[0, 1].plot(pool['ASC'])
    ax[0, 1].set_title('Audio Spectrum Centroid (ASC)')

    ax[1, 0].plot(pool['ASS'])
    ax[1, 0].set_title('Audio Spectrum Spread (ASS)')

    ax[1, 1].imshow(asf, aspect='auto', origin='lower', interpolation='none')
    ax[1, 1].set_title('Audio Spectrum Flatness (ASF)')

    fig.show()


if __name__ == '__main__':
    main()
