import essentia
import matplotlib.pyplot as plt
from essentia.standard import MonoLoader, Windowing, Spectrum, MFCC, FrameGenerator
from pylab import plot, show, figure, imshow

plt.rcParams['figure.figsize'] = (15, 6)

audio = MonoLoader(filename='../sounds/sheep.wav')()
plot(audio)
plt.title("Ovis orientalis aries")
show()

w = Windowing(type='hamming')
spectrum = Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum
mfcc = MFCC()

frame = audio[20000: 20000 + 1024]
spec = spectrum(w(frame))

fig = figure(figsize=(15, 6))
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.07, top=0.93)

ax = fig.add_subplot(1, 2, 1)
ax.plot(spec[:250])
ax.set_title('(a)')

specs = []
mfccs = []

for frame in FrameGenerator(audio, frameSize=1024, hopSize=512, startFromZero=True):
    spec = spectrum(w(frame))
    _, mfcc_coeffs = mfcc(spec)
    specs.append(spec)
    mfccs.append(mfcc_coeffs)

# transpose to have it in a better shape
# we need to convert the list to an essentia.array first (== numpy.array of floats)
specs = essentia.array(specs).T
mfccs = essentia.array(mfccs).T

ax = fig.add_subplot(1, 2, 2)
ax.imshow(specs[:160, :], aspect='auto', origin='lower', interpolation='none')
ax.set_title('(b)')

fig.show()

imshow(mfccs[:, :], aspect='auto', origin='lower', interpolation='none')
plt.title('MFCCs in frames')
show()
