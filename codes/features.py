import pathlib

import pylab as pl
import seaborn as sns
from matplotlib import collections as mc
from matplotlib.ticker import MaxNLocator, FuncFormatter

from clusterapp.features import Audio

FS = 44100


def plot_temporal_descriptors(audio):
    awf = [[(i, f[0]), (i, f[1])] for i, f in enumerate(audio.audio_waveform)]

    fig, (ax1, ax2) = pl.subplots(1, 2, figsize=(12, 4))
    fig.subplots_adjust(left=0.07, right=0.97)

    lc = mc.LineCollection(awf, linewidths=2)
    ax1.add_collection(lc)
    ax1.autoscale()
    ax1.set_title('Audio Waveform (AWF)')

    ax2.plot(audio.audio_power)
    ax2.set_title('Audio Power (AP)')

    fig.show()


def plot_spectral_descriptors(audio):
    fig, ax = pl.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(left=0.07, right=0.97)

    ax[0, 0].plot(audio.spectral_centroid, label='SC')
    ax[0, 0].plot(audio.spectral_spread, label='SS')
    ax[0, 0].legend()
    ax[0, 0].set_title('Spectral Centroid (SC) y Spectral Spread (SS)')

    ax[0, 1].plot(audio.spectral_roll_off)
    ax[0, 1].set_title('Spectral Roll-off (SRO)')

    ax[1, 0].plot(audio.spectral_flux)
    ax[1, 0].set_title('Spectral Flux (SFX)')

    ax[1, 1].plot(audio.spectral_flatness)
    ax[1, 1].set_title('Spectral Flatness (SF)')

    fig.show()


def plot_harmonic_descriptors(audio):
    fig, (ax1, ax2, ax3) = pl.subplots(1, 3, figsize=(12, 4))
    fig.subplots_adjust(left=0.055, right=0.98)

    ax1.plot(audio.inharmonicity)
    ax1.set_title('Inharmonicity (INH)')

    ax2.plot(audio.oer)
    ax2.set_title('Odd to Even Harmonic Energy Ratio (OER)')

    ax3.imshow(audio.tristimulus.T[:, :], aspect='auto', origin='lower', interpolation='none')
    ax3.set_title('Tristimulus')
    ax3.yaxis.set_major_locator(MaxNLocator(integer=True))

    fig.show()


def plot_mfccs(audio):
    fig, ax = pl.subplots(1, 1, figsize=(12, 4))
    fig.subplots_adjust(left=0.05, right=0.97)

    ax.imshow(audio.mfcc.T[1:, :], aspect='auto', origin='lower', interpolation='none')
    ax.set_title('MFCC')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: int(x + 1)))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    fig.show()


def main():
    sns.set()
    sns.set_style('white')

    audio = Audio(pathlib.Path('../sounds/sheep.wav'))

    plot_temporal_descriptors(audio)
    plot_spectral_descriptors(audio)
    plot_harmonic_descriptors(audio)
    plot_mfccs(audio)


if __name__ == '__main__':
    main()
