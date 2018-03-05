import pathlib
import time

import essentia
import essentia.standard as es
import numpy as np
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans


def main():
    X = []
    y = []
    w = es.Windowing(type='hann')
    spectrum = es.Spectrum()
    mfcc = es.MFCC()

    for file in pathlib.Path('../sounds/testing').iterdir():
        audio = es.MonoLoader(filename=str(file))()
        y.append(file.name)
        mfccs = []
        for frame in es.FrameGenerator(audio, frameSize=1024, hopSize=512, startFromZero=True):
            spec = spectrum(w(frame))
            _, mfcc_coeffs = mfcc(spec)
            mfccs.append(mfcc_coeffs)
        mfccs = essentia.array(mfccs).T[1:, :]
        X.append(mfccs.mean(1))

    X = np.array(X)

    kmeans = KMeans(n_clusters=4)
    hdbscan = HDBSCAN(min_cluster_size=3, gen_min_span_tree=True)

    algorithms = [
        ('KMeans', kmeans),
        ('HDBSCAN', hdbscan)
    ]

    for algorithm_name, algorithm in algorithms:
        start = time.time()
        algorithm.fit(X)

        results = list(zip(algorithm.labels_, y))
        results.sort()
        for label, segment_name in results:
            print(label, segment_name, sep='\t')

        print('--------------------')
        print('done %s in %d seconds' % (algorithm_name, time.time() - start))
        print('--------------------\n')


if __name__ == '__main__':
    main()
