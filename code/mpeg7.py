import essentia
import pylab as pl
from essentia.standard import MonoLoader, Windowing, Spectrum, MFCC, FrameGenerator, InstantPower
from matplotlib import collections as mc

audio = MonoLoader(filename='../sounds/sheep.wav')()

w = Windowing(type='hann')
spectrum = Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum
mfcc = MFCC()
power = InstantPower()

awf = []
pool = essentia.Pool()

# BASIC DESCRIPTORS

for i, frame in enumerate(FrameGenerator(audio, frameSize=512, hopSize=512, startFromZero=True)):
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
ax2.autoscale()
ax2.set_title('Audio Power (AP)')

fig.show()
