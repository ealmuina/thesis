import essentia
import essentia.standard as es
import pylab as pl
import seaborn as sns


def main():
    sns.set()
    sns.set_style("white")

    audio = es.MonoLoader(filename='../sounds/sheep.wav')()

    fig, (ax1, ax2, ax3) = pl.subplots(3, 1, figsize=(6, 6))
    fig.subplots_adjust(left=0.06, right=0.97)

    ax1.plot(audio)
    ax1.set_title("Oscilograma")

    w = es.Windowing(type='hamming')
    spectrum = es.Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum

    frame = audio[20000: 20000 + 1024]
    spec = spectrum(w(frame))

    ax2.plot(spec[:250])
    ax2.set_title('Espectro')

    specs = []

    for frame in es.FrameGenerator(audio, frameSize=1024, hopSize=512, startFromZero=True):
        spec = spectrum(w(frame))
        specs.append(spec)

    # transpose to have it in a better shape
    # we need to convert the list to an essentia.array first (== numpy.array of floats)
    specs = essentia.array(specs).T

    ax3.imshow(specs[:80, :], aspect='auto', origin='lower', interpolation='none')
    ax3.set_title('Espectrograma')

    fig.show()


if __name__ == '__main__':
    main()
