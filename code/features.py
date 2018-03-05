import essentia
import essentia.standard as es
import pylab as pl
from matplotlib import collections as mc


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


def main():
    audio = es.MonoLoader(filename='../sounds/sheep.wav')()

    plot_temporal_descriptors(audio)
    plot_spectral_descriptors(audio)


if __name__ == '__main__':
    main()
