import essentia
import essentia.standard as es
import pylab as pl
import seaborn as sns


def main():
    sns.set()

    audio = es.MonoLoader(filename='../sounds/sheep.wav')()

    fig, ax = pl.subplots(1, 1, figsize=(12, 4))
    fig.subplots_adjust(left=0.06, right=0.97)

    ax.plot(audio)
    ax.set_title("Ovis orientalis aries")
    fig.show()

    w = es.Windowing(type='hamming')
    spectrum = es.Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum

    frame = audio[20000: 20000 + 1024]
    spec = spectrum(w(frame))

    sns.set_style("white")
    fig = pl.figure(figsize=(12, 4))
    fig.subplots_adjust(left=0.07, right=0.97)

    ax = fig.add_subplot(1, 2, 1)
    ax.plot(spec[:250])
    ax.set_title('(a)')

    specs = []

    for frame in es.FrameGenerator(audio, frameSize=1024, hopSize=512, startFromZero=True):
        spec = spectrum(w(frame))
        specs.append(spec)

    # transpose to have it in a better shape
    # we need to convert the list to an essentia.array first (== numpy.array of floats)
    specs = essentia.array(specs).T

    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(specs[:80, :], aspect='auto', origin='lower', interpolation='none')
    ax.set_title('(b)')

    fig.show()


if __name__ == '__main__':
    main()
