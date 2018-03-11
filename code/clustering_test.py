import pathlib
import time

import essentia
import essentia.standard as es
import numpy as np
from hdbscan import HDBSCAN
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder


def print_table(table):
    """
    Print a list of tuples as a pretty tabulated table.
    :param table: List of tuples, each one will be a row of the printed table
    """

    col_width = [max(len(x) for x in col) for col in zip(*table)]
    for line in table:
        print((" " * 3).join("{:{}}".format(x, col_width[i]) for i, x in enumerate(line)))


def main():
    X = []
    y = []
    w = es.Windowing(type='hann')
    spectrum = es.Spectrum()
    mfcc = es.MFCC(inputSize=513)

    for file in pathlib.Path('../sounds/testing').iterdir():
        audio = es.MonoLoader(filename=str(file))()
        y.append(file.name.split('-')[0])
        mfccs = []
        for frame in es.FrameGenerator(audio, frameSize=1024, hopSize=512, startFromZero=True):
            spec = spectrum(w(frame))
            _, mfcc_coeffs = mfcc(spec)
            mfccs.append(mfcc_coeffs)
        mfccs = essentia.array(mfccs).T[1:, :]
        X.append(mfccs.mean(1))

    X = np.array(X)

    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    kmeans = KMeans(n_clusters=6)
    gmm = GaussianMixture(n_components=6)
    hdbscan = HDBSCAN(min_cluster_size=3, gen_min_span_tree=True)

    algorithms = [
        ('KMeans', kmeans),
        ('GaussianMixture', gmm),
        ('HDBSCAN', hdbscan)
    ]

    report = [
        ('ALGORITHM', 'SCORE', 'TIME')
    ]

    for algorithm_name, algorithm in algorithms:
        start = time.time()
        algorithm.fit(X)
        labels = algorithm.labels_ if hasattr(algorithm, 'labels_') else algorithm.predict(X)

        report.append((
            algorithm_name,
            '%.3f' % (metrics.adjusted_rand_score(y, labels)),
            '%.3f' % (time.time() - start)
        ))

    print_table(report)


if __name__ == '__main__':
    main()
