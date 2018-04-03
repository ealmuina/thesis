import pathlib

import numpy as np
from hdbscan import HDBSCAN
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import scale, LabelEncoder

from clusterapp.features import Audio


class IdentityClustering:
    def __init__(self):
        self.labels_ = None

    def fit(self, X, y):
        self.labels_ = y


CLUSTERING = {
    'none': IdentityClustering,
    'kmeans': KMeans,
    'gmm': GaussianMixture,
    'hdbscan': HDBSCAN
}


class Library:
    def __init__(self, path):
        self.segments = {}

        for file in pathlib.Path(path).iterdir():
            audio = Audio(file)
            category = file.name.split('-')[0]
            items = self.segments.get(category, [])
            items.append(audio)
            self.segments[category] = items

        self.categories = set(self.segments.keys())

    @staticmethod
    def _parse_clustering_algo(algorithm, categories):
        if algorithm in ('kmeans',):
            return CLUSTERING[algorithm](n_clusters=len(categories))
        elif algorithm in ('gmm',):
            return CLUSTERING[algorithm](n_components=len(categories))
        elif algorithm in ('hdbscan',):
            return CLUSTERING[algorithm](min_cluster_size=3)
        return CLUSTERING[algorithm]()

    def cluster(self, categories, features, algorithm):
        algorithm = self._parse_clustering_algo(algorithm, categories)

        X, y, names = [], [], []
        for cat in categories:
            for audio in self.segments[cat]:
                X.append([
                    getattr(audio, features[0]).mean(),
                    getattr(audio, features[1]).mean()
                ])
                names.append(audio.name)
                y.append(audio.name.split('-')[0])
        X = np.array(X, dtype=np.float64)

        scaled_X = scale(X) if len(X) else X
        algorithm.fit(scaled_X, y)
        labels = algorithm.labels_ if hasattr(algorithm, 'labels_') else algorithm.predict(scaled_X)

        result = {}
        for i, label in enumerate(labels):
            label = str(label)
            items = result.get(label, [])
            items.append({
                'name': names[i],
                'x': X[i, 0],
                'y': X[i, 1],
            })
            result[label] = items

        return result, evaluate(labels, y)


def evaluate(labels_pred, labels_true):
    le = LabelEncoder()
    le.fit(labels_true)
    labels_true = le.transform(labels_true)

    return {
        'ARI': round(metrics.adjusted_rand_score(labels_true, labels_pred), 2),
        'AMI': round(metrics.adjusted_mutual_info_score(labels_true, labels_pred), 2),
        'Homogeneity': round(metrics.homogeneity_score(labels_true, labels_pred), 2),
        'Completeness': round(metrics.completeness_score(labels_true, labels_pred), 2),
    }


def statistics(clustering):
    result = {}
    for label in clustering.keys():
        x = np.array([item['x'] for item in clustering[label]])
        y = np.array([item['y'] for item in clustering[label]])

        result[label] = {
            'x_mean': x.mean().round(2),
            'x_std': x.std().round(2),
            'y_mean': y.mean().round(2),
            'y_std': y.std().round(2)
        }
    return result
