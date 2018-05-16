from scipy.signal import welch 

class Segment:
    """docstring for Segment"""

    def __init__(self, signal, a, b):
        self.signal = signal

        # Segment time signal information
        self.data = self.signal.data[a:b + 1]
        self.IndexFrom = a
        self.IndexTo = b
        self.samplingRate = self.signal.samplingRate

        self.envelope = None
        self.envelope_type = None

        self.spectrum = None
        self.spectrum_freqs = None
        self.compute_spectrum()

        if self.signal.freqs is None:
            self.signal.compute_spectrogram()

        # Segment spectrogram information
        self.FrameFrom = self.signal.spec.shape[1] * a // len(self.signal.data)
        self.FrameTo = self.signal.spec.shape[1] * b // len(self.signal.data)

        self.freqs = self.signal.freqs
        self.times = self.signal.times[self.FrameFrom:self.FrameTo + 1]
        self.spec = self.signal.spec[:, self.FrameFrom:self.FrameTo + 1]

        if self.signal.peaks_indexes is None:
            self.signal.compute_peaks()

        self.peaks_indexes = self.signal.peaks_indexes[self.FrameFrom:self.FrameTo + 1]
        self.peaks_values = self.signal.peaks_values[self.FrameFrom:self.FrameTo + 1]

        # Dictionary of segment measures "name":value
        self.measures_dict = {}

    def compute_spectrum(self, nperseg=256, percent_overlap=0.75):
        self.spectrum_freqs, self.spectrum = welch(self.data, fs=self.samplingRate,
                                                   window=self.signal.window, nperseg=256,
                                                   noverlap=int(nperseg * percent_overlap),
                                                   nfft=None, detrend='constant', return_onesided=True,
                                                   scaling='spectrum', axis=-1)
        return True
