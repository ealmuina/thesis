from essentia.standard import *

import matplotlib.pyplot as plt

from pylab import plot, show, figure

plt.rcParams['figure.figsize'] = (15, 6)

audio = MonoLoader(filename='/home/eddy/Downloads/sonidos animales/cougar/cougar4.wav')()
plot(audio)
plt.title("Puma concolor")
show()

w = Windowing(type='hann')
spectrum = Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum

frame = audio[1 * 44100: 1 * 44100 + 1024]
spec = spectrum(w(frame))

fig = figure(figsize=(15, 6))
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.07, top=0.93)

ax = fig.add_subplot(1, 2, 1)
ax.plot(spec[:250])
ax.set_title('(a)')

specs = []

for frame in FrameGenerator(audio, frameSize=1024, hopSize=512, startFromZero=True):
    spec = spectrum(w(frame))
    specs.append(spec)

# transpose to have it in a better shape
# we need to convert the list to an essentia.array first (== numpy.array of floats)
specs = essentia.array(specs).T

ax = fig.add_subplot(1, 2, 2)
ax.imshow(specs[:110, :], aspect='auto', origin='lower', interpolation='none')
ax.set_title('(b)')

fig.show()
