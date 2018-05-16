import numpy as np
from scipy import signal
from scipy.fftpack import hilbert
from scipy.io import wavfile


def load_file(filename):
    fs, x = wavfile.read(filename)
    return x, fs


def get_location(segment, location='centre'):
    if location == 'centre':
        return int(segment.spec.shape[1] / 2)
    elif location == 'start':
        return 0
    elif location == 'end':
        return segment.spec.shape[1] - 1
    elif location == 'max':
        index = np.argmax(segment.spec)
        return index // segment.spec.shape[0]
    elif location == 'max_amp':
        index = np.argmax(segment.data)
        value = int(index * segment.spec.shape[1] / len(segment.data))
        return value
    return None


def to_db(x, reference=1.0):
    return 20 * np.log10(x / reference)


def apply_threshold(value, threshold=-20):
    return value * np.power(10, threshold / 10.0)


def geometric_mean(data):
    g_mean = 0.0
    for i in range(len(data)):
        if data[i] == 0.0:
            continue
        else:
            g_mean += np.log(data[i])

    g_mean /= len(data)
    g_mean = np.exp(g_mean)

    return g_mean


def energy(data):
    return np.sum(np.square(data))

"""Envelopes"""


def three_step_envelope(data, chunk_len=20, filter_order=4, cutoff_frequency=0.1):
    y = abs(data)

    n = len(y)
    k = int(n / chunk_len)
    z = []

    for i in range(k):
        z += [np.max(y[i * chunk_len: (i + 1) * chunk_len - 1]) for j in range(chunk_len)]
    if n % chunk_len != 0:
        z += [np.max(y[(k - 1) * chunk_len: -1]) for j in range(n % chunk_len)]
    z = np.array(z)

    b, a = signal.butter(filter_order, cutoff_frequency, 'low')
    w = signal.filtfilt(b, a, z)

    return w


def hilbert_envelope(data):
    h_data = hilbert(data)
    return np.sqrt(np.square(data) + np.square(h_data))


"""Spectrogram filters"""


def apply_mean_filter(data):
    result = []
    for i in range(data.shape[0]):
        result.append([])
        for j in range(data.shape[1]):
            result[i].append(get_mean(data, i, j))
    return np.array(result)


def apply_median_filter(data):
    result = []
    for i in range(data.shape[0]):
        result.append([])
        for j in range(data.shape[1]):
            result[i].append(get_median(data, i, j))
    return np.array(result)


def get_mean(data, i, j):
    aux = [data[i, j]]
    if is_valid_position(data, i - 1, j):
        aux.append(data[i - 1, j])
    if is_valid_position(data, i + 1, j):
        aux.append(data[i + 1, j])

    return np.mean(np.array(aux))


def get_median(data, i, j):
    aux = [data[i, j]]
    if is_valid_position(data, i - 1, j):
        aux.append(data[i - 1, j])
    if is_valid_position(data, i + 1, j):
        aux.append(data[i + 1, j])

    return np.median(np.array(aux))


def is_valid_position(data, i, j):
    return 0 <= i < data.shape[0] and 0 <= j < data.shape[1]
