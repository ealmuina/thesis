import argparse
import itertools
import os
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from hdbscan import HDBSCAN
from matplotlib import offsetbox
from sklearn import metrics
from sklearn.cluster import KMeans, SpectralClustering, AffinityPropagation
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder, scale

from clusterapp.features import Audio

TESTING_DIR = '../sounds/testing'


class ClusteringAlgorithm:
    def __init__(self, clusterer, name):
        self.clusterer = clusterer
        self.name = name
        self.labels = None
        self.centroids = None

    def __add__(self, other):
        result = ClusteringAlgorithm(None, '%s+%s' % (self.name, other.name))
        result.labels = np.zeros(len(self.labels), dtype=np.int)
        result.centroids = self.centroids

        order = metrics.pairwise_distances_argmin(self.centroids, other.centroids)
        for l in range(self.labels.max()):
            l1 = self.labels == l
            l2 = other.labels == order[l]
            result.labels += (l1 & l2) * (l + 1)
        result.labels -= 1

        return result

    @staticmethod
    def _centroids(X, labels):
        centroids = []
        for l in range(labels.max()):
            x = X[labels == l]
            centroids.append(x.mean(0))
        return np.array(centroids)

    def fit(self, X):
        if self.clusterer:
            self.clusterer.fit(X)
            labels = self.clusterer.labels_ if hasattr(self.clusterer, 'labels_') else self.clusterer.predict(X)
            self.labels = np.array(labels)
            self.centroids = self._centroids(X, np.array(self.labels))


def export_results(labels, names, path):
    results = list(zip(labels, names))
    results.sort()

    os.makedirs('out', exist_ok=True)
    with open('out/' + path, 'w') as file:
        for label, name in results:
            file.write('%d\t%s\n' % (label, name))


def extract_features(audios, features):
    X = []

    for audio in audios:
        current = []
        for feature in features:
            current.extend(process(feature, getattr(audio, feature)))
        X.append(current)

    X = np.array(X, dtype=np.float64)
    X = scale(X)
    return X


def main(export=False, plot=False):
    sns.set()
    sns.set_style('white')

    single_features = [
        'min_freq', 'max_freq', 'peak_freq', 'peak_ampl', 'fundamental_freq', 'bandwidth'
    ]
    features = [
        ('mfcc',),
        *[(f1, f2) for f1, f2 in itertools.combinations(single_features, 2)]
    ]

    audios, y = [], []
    start = time.time()
    for file in pathlib.Path(TESTING_DIR).iterdir():
        audios.append(Audio(file))
        y.append(file.name.split('-')[0])
    print('Features computed in %.3f seconds.' % (time.time() - start))

    for f in features:
        X = extract_features(audios, f)
        test(X, y, f, export, plot)


def plot_data(X, y, title, show_labels=False):
    if X.shape[1] > 2:  # apply a manifold method
        start = time.time()

        X = X.astype(np.float64)
        tsne = TSNE(n_components=2, random_state=1994)
        results = tsne.fit(X)
        X = results.embedding_

        print("Done t-distributed Stochastic Neighbor Embedding in %.3f seconds." % (time.time() - start))

    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    fig, ax = plt.subplots(1, 1)
    ax.scatter(X[:, 0], X[:, 1], marker='o', c=y)

    if show_labels:
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-4:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.TextArea(y[i], textprops={'size': 5}),
                X[i],
                fontsize=5
            )
            ax.add_artist(imagebox)

    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(title)
    fig.show()
    return X


def print_table(table):
    """
    Print a list of tuples as a pretty tabulated table.
    :param table: List of tuples, each one will be a row of the printed table
    """

    col_width = [max(len(x) for x in col) for col in zip(*table)]
    for line in table:
        print((" " * 3).join("{:{}}".format(x, col_width[i]) for i, x in enumerate(line)))


def process(feature_name, x):
    if feature_name == 'mfcc':
        return x.mean(0)
    return np.array([x.mean()])


def report_algorithm(X, y, algorithm, filenames, plt_X, export=False, plot=False):
    start = time.time()
    algorithm.fit(X)
    labels = algorithm.labels

    if export:
        export_results(labels, filenames, '%s.txt' % algorithm.name)
    if plot:
        plot_data(plt_X, labels, algorithm.name)

    current_y = [y[i] for i in range(len(y)) if labels[i] >= 0]
    labels = [l for l in labels if l >= 0]

    return (
        algorithm.name,
        '%.3f' % metrics.adjusted_rand_score(current_y, labels),
        '%.3f' % metrics.adjusted_mutual_info_score(current_y, labels),
        '%.3f' % metrics.homogeneity_score(current_y, labels),
        '%.3f' % metrics.completeness_score(current_y, labels),
        '%.3f' % (time.time() - start)
    )


def test(X, y, features, export=False, plot=False):
    print('Testing %s' % str(features))

    filenames = [file.name for file in pathlib.Path(TESTING_DIR).iterdir()]

    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    plt_X = None
    if plot:
        plt_X = plot_data(X, y, 'True', True)

    algorithms = [
        ClusteringAlgorithm(KMeans(n_clusters=len(le.classes_)), 'KMeans'),
        ClusteringAlgorithm(GaussianMixture(n_components=len(le.classes_)), 'GaussianMixture'),
        ClusteringAlgorithm(HDBSCAN(min_cluster_size=3), 'HDBSCAN'),
        ClusteringAlgorithm(SpectralClustering(n_clusters=len(le.classes_)), 'SpectralClustering'),
        ClusteringAlgorithm(AffinityPropagation(), 'AffinityPropagation')
    ]

    report = [
        ('ALGORITHM', 'ARI', 'AMI', 'HOMOGENEITY', 'COMPLETENESS', 'TIME')
    ]

    for algorithm in algorithms:
        report.append(report_algorithm(X, y, algorithm, filenames, plt_X, export, plot))

    merge = algorithms[2] + algorithms[0] + algorithms[1]
    report.append(report_algorithm(X, y, merge, filenames, plt_X, export, plot))

    print_table(report)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    main(args.report, args.plot)
