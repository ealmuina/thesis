import argparse
import os
import pathlib
import time

import essentia.standard as es
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from hdbscan import HDBSCAN
from matplotlib import offsetbox
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder, scale

from codes.features import FeaturesExtractor

FEATURE_INDICES = {
    'LAT': (0, 1),
    'AWF': (1, 2),
    'AP': (3, 1),
    'TC': (4, 1),
    'ED': (5, 1),
    'AC': (6, 1),
    'ZCR': (7, 1),

    'SC': (8, 1),
    'SS': (9, 1),
    'SRO': (10, 1),
    'SFX': (11, 1),
    'SF': (12, 1),

    'F0': (13, 1),
    'INH': (14, 1),
    'OER': (15, 1),
    'T': (16, 3),

    'MFCC': (19, 13),
    'D_MFCC': (32, 13),
    'D2_MFCC': (45, 13),
}
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


def load():
    X = []
    y = []
    extractor = FeaturesExtractor()

    start = time.time()
    for file in pathlib.Path(TESTING_DIR).iterdir():
        audio = es.MonoLoader(filename=str(file))()

        y.append(file.name.split('-')[0])

        current = extractor.full_features(audio)
        X.append(current)

    X = np.array(X, dtype=np.float64)
    X = scale(X)
    print('Features computed in %.3f seconds.' % (time.time() - start))
    return X, y


def main(export=False, plot=False):
    sns.set()
    sns.set_style('white')

    X, y = load()
    features = [
        ('MFCC',)
    ]

    for f in features:
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
    print('\nTesting %s' % str(features))
    new_X = []

    for i, x in enumerate(X):
        current = []
        for feature in features:
            a, b = FEATURE_INDICES[feature]
            current.extend(x[a:a + b])
        new_X.append(current)

    X = np.array(new_X)
    filenames = [file.name for file in pathlib.Path(TESTING_DIR).iterdir()]

    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    plt_X = None
    if plot:
        plt_X = plot_data(X, y, 'True', True)

    kmeans = KMeans(n_clusters=len(le.classes_))
    gmm = GaussianMixture(n_components=len(le.classes_))
    hdbscan = HDBSCAN(min_cluster_size=3)

    algorithms = [
        ClusteringAlgorithm(kmeans, 'KMeans'),
        ClusteringAlgorithm(gmm, 'GaussianMixture'),
        ClusteringAlgorithm(hdbscan, 'HDBSCAN')
    ]

    report = [
        ('ALGORITHM', 'ARI', 'AMI', 'HOMOGENEITY', 'COMPLETENESS', 'TIME')
    ]

    for algorithm in algorithms:
        report.append(report_algorithm(X, y, algorithm, filenames, plt_X, export, plot))

    merge = algorithms[2] + algorithms[0] + algorithms[1]
    report.append(report_algorithm(X, y, merge, filenames, plt_X, export, plot))

    print_table(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    main(args.report, args.plot)
