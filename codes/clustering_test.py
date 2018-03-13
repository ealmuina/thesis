import argparse
import os
import pathlib
import time

import essentia.standard as es
import numpy as np
from hdbscan import HDBSCAN
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder

from codes.features import FeaturesExtractor


def print_table(table):
    """
    Print a list of tuples as a pretty tabulated table.
    :param table: List of tuples, each one will be a row of the printed table
    """

    col_width = [max(len(x) for x in col) for col in zip(*table)]
    for line in table:
        print((" " * 3).join("{:{}}".format(x, col_width[i]) for i, x in enumerate(line)))


def export_results(labels, names, path):
    results = list(zip(labels, names))
    results.sort()

    os.makedirs('out', exist_ok=True)
    with open('out/' + path, 'w') as file:
        for label, name in results:
            file.write('%d\t%s\n' % (label, name))


def main(export=False):
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

    print_table(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()

    main(args.report)
