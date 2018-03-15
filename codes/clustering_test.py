import argparse
import os
import pathlib
import time

import essentia.standard as es
import numpy as np
import pylab as pl
from hdbscan import HDBSCAN
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder

from codes.features import FeaturesExtractor


def export_results(labels, names, path):
    results = list(zip(labels, names))
    results.sort()

    os.makedirs('out', exist_ok=True)
    with open('out/' + path, 'w') as file:
        for label, name in results:
            file.write('%d\t%s\n' % (label, name))


def main(export=False, plot=False):
    X = []
    y = []
    filenames = []
    extractor = FeaturesExtractor()

    start = time.time()
    for file in pathlib.Path('../sounds/testing').iterdir():
        audio = es.MonoLoader(filename=str(file))()
        filenames.append(file.name)
        y.append(file.name.split('-')[0])
        mfccs, _, _ = extractor.mfcc_descriptors(audio)
        X.append(mfccs.mean(1))

    X = np.array(X)
    print('Features computed in %.3f seconds.' % (time.time() - start))

    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    if plot:
        plot_data(X, y)

    kmeans = KMeans(n_clusters=len(le.classes_))
    gmm = GaussianMixture(n_components=len(le.classes_))
    hdbscan = HDBSCAN(min_cluster_size=5)

    algorithms = [
        ('KMeans', kmeans),
        ('GaussianMixture', gmm),
        ('HDBSCAN', hdbscan)
    ]

    report = [
        ('ALGORITHM', 'ARI', 'NMI', 'HOMOGENEITY', 'COMPLETENESS', 'TIME')
    ]

    for algorithm_name, algorithm in algorithms:
        start = time.time()
        algorithm.fit(X)
        labels = algorithm.labels_ if hasattr(algorithm, 'labels_') else algorithm.predict(X)

        report.append((
            algorithm_name,
            '%.3f' % metrics.adjusted_rand_score(y, labels),
            '%.3f' % metrics.normalized_mutual_info_score(y, labels),
            '%.3f' % metrics.homogeneity_score(y, labels),
            '%.3f' % metrics.completeness_score(y, labels),
            '%.3f' % (time.time() - start)
        ))

        if export:
            export_results(labels, filenames, '%s.txt' % algorithm_name)

    print()
    print_table(report)


def plot_data(X, y):
    start = time.time()

    X = X.astype(np.float64)
    tsne = TSNE(n_components=2, random_state=0, init='pca')
    results = tsne.fit(X)
    coords = results.embedding_

    print("Done t-distributed Stochastic Neighbor Embedding in %.3f seconds." % (time.time() - start))

    pl.scatter(coords[:, 0], coords[:, 1], marker='.', c=y)
    pl.show()


def print_table(table):
    """
    Print a list of tuples as a pretty tabulated table.
    :param table: List of tuples, each one will be a row of the printed table
    """

    col_width = [max(len(x) for x in col) for col in zip(*table)]
    for line in table:
        print((" " * 3).join("{:{}}".format(x, col_width[i]) for i, x in enumerate(line)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    main(args.report, args.plot)
