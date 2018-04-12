import librosa
import numpy as np
import scipy.signal

FS = 44100


class Audio:
    def __init__(self, path):
        self.audio, self.sr = librosa.load(str(path))
        self.spectrum, _ = librosa.magphase(librosa.stft(
            y=self.audio,
            n_fft=1024,
            hop_length=512,
            window=scipy.signal.hann
        ))
        self.name = path.name
        self.pool = {}

        self._build_temporal_features()
        self._build_spectral_features()
        self._build_mfcc()

        self._features = {
            'audio_correlation': 'AC',
            'bandwidth': 'SB',
            'max_freq': 'FMax',
            'mfcc': 'MFCC',
            'min_freq': 'FMin',
            'peak_ampl': 'PA',
            'spectral_centroid': 'SC',
            'spectral_flatness': 'SF',
            'spectral_roll_off': 'SRO',
            'zcr': 'ZCR'
        }

    def __getattr__(self, item):
        if item in self._features:
            key = self._features[item]
            return self.pool[key]
        raise AttributeError(item)

    def _build_mfcc(self):
        S = librosa.feature.melspectrogram(S=self.spectrum)
        self.pool['MFCC'] = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=13).T

    def _build_spectral_features(self):
        self.pool['PA'] = self.spectrum.max(axis=1)
        self.pool['FMin'] = np.apply_along_axis(_first_over_threshold, 0, self.spectrum)
        self.pool['FMax'] = np.apply_along_axis(_first_over_threshold, 0, np.flip(self.spectrum, 0))
        self.pool['SB'] = self.pool['FMax'] - self.pool['FMin']

        self.pool['SC'] = librosa.feature.spectral_centroid(S=self.spectrum)
        self.pool['SRO'] = librosa.feature.spectral_rolloff(S=self.spectrum)
        self.pool['SF'] = librosa.feature.spectral_flatness(S=self.spectrum)

    def _build_temporal_features(self):
        self.pool['AC'] = librosa.autocorrelate(self.audio)
        self.pool['ZCR'] = librosa.feature.zero_crossing_rate(self.audio)


def _decibels(a1, a2):
    return 10 * np.log10(a1 / a2 + 1e-8)


def _first_over_threshold(spectrum, threshold=-20):
    peak_ampl = spectrum.max()
    for k in range(spectrum.shape[0]):
        d = _decibels(spectrum[k], peak_ampl)
        if d >= threshold:
            return k * (FS / 1024)
