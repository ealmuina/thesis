import itertools
import pathlib
from collections import Counter

import numpy as np
from hdbscan import HDBSCAN
from sklearn import metrics
from sklearn.cluster import KMeans, SpectralClustering, AffinityPropagation
from sklearn.manifold import MDS
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import scale, LabelEncoder

from .features.audio import Audio


class IdentityClustering:
    def __init__(self):
        self.labels_ = None

    def fit(self, _, y=None):
        self.labels_ = y


class Library:
    @staticmethod
    def _extract_features(audio, features):
        current = []
        for feature in features:
            x = getattr(audio, feature)
            current.append(x)
        return current

    @staticmethod
    def _parse_clustering_algo(algorithm, n_clusters):
        return {
            'kmeans': KMeans(n_clusters=n_clusters),
            'spectral': SpectralClustering(n_clusters=n_clusters),
            'affinity': AffinityPropagation(),
            'gmm': GaussianMixture(n_components=n_clusters),
            'hdbscan': HDBSCAN(min_cluster_size=3),
            'none': IdentityClustering()
        }[algorithm]


class ClassifiedLibrary(Library):
    def __init__(self, path):
        self.segments = {}

        for file in pathlib.Path(path).iterdir():
            audio = Audio(file)
            category = file.name.split('-')[0]
            items = self.segments.get(category, [])
            items.append(audio)
            self.segments[category] = items

        self.categories = set(self.segments.keys())

    def _predict(self, categories, features, algorithm):
        algorithm = self._parse_clustering_algo(algorithm, len(categories))

        X, y, names = [], [], []
        for cat in categories:
            for audio in self.segments[cat]:
                X.append(self._extract_features(audio, features))
                names.append(audio.name)
                y.append(audio.name.split('-')[0])
        X = np.array(X, dtype=np.float64)

        scaled_X = scale(X) if len(X) else X
        algorithm.fit(scaled_X, y)
        labels = algorithm.labels_ if hasattr(algorithm, 'labels_') else algorithm.predict(scaled_X)

        return X, scaled_X, y, names, labels

    def best_features(self, categories, features_set, algorithm):
        best = {
            'AMI': 0
        }
        for r in range(1, 3):
            for features in itertools.combinations(features_set, r):
                X, scaled_X, y, names, labels = self._predict(categories, features, algorithm)
                ami = metrics.adjusted_mutual_info_score(labels, y)

                if ami > best['AMI']:
                    best.update({
                        'AMI': ami,
                        'features': features
                    })

        clustering, scores = self.cluster(categories, best['features'], algorithm)
        return clustering, scores, best['features']

    def cluster(self, categories, features, algorithm):
        X, scaled_X, y, names, labels = self._predict(categories, features, algorithm)

        X_2d = X
        if X.shape[1] != 2:
            mds = MDS(n_components=2, random_state=0)
            X_2d = mds.fit_transform(X)

        result = {}
        for i, label in enumerate(labels):
            label = str(label)
            items = result.get(label, [])
            items.append({
                'name': names[i],
                'label_true': y[i],
                'x': X[i, :],
                'x_2d': X_2d[i, :]
            })
            result[label] = items

        return result, evaluate(scaled_X, labels, y)


class UnclassifiedLibrary(Library):
    def __init__(self, path):
        self.segments = [
            Audio(file) for file in pathlib.Path(path).iterdir()
        ]

    def cluster(self, n_clusters, features, algorithm):
        algorithm = self._parse_clustering_algo(algorithm, n_clusters)

        X, names = [], []
        for audio in self.segments:
            X.append(self._extract_features(audio, features))
            names.append(audio.name)
        X = np.array(X, dtype=np.float64)

        scaled_X = scale(X) if len(X) else X
        algorithm.fit(scaled_X)
        labels = algorithm.labels_ if hasattr(algorithm, 'labels_') else algorithm.predict(scaled_X)

        X_2d = X
        if X.shape[1] != 2:
            mds = MDS(n_components=2, random_state=0)
            X_2d = mds.fit_transform(X)

        result = {}
        for i, label in enumerate(labels):
            label = str(label)
            items = result.get(label, [])
            items.append({
                'name': names[i],
                'x': X[i, :],
                'x_2d': X_2d[i, :]
            })
            result[label] = items

        return result, evaluate(scaled_X, labels)


def evaluate(X, labels_pred, labels_true=None):
    if labels_true:
        le = LabelEncoder()
        le.fit(labels_true)
        labels_true = le.transform(labels_true)

        return {
            'ARI': round(metrics.adjusted_rand_score(labels_true, labels_pred), 2),
            'AMI': round(metrics.adjusted_mutual_info_score(labels_true, labels_pred), 2),
            'Homogeneity': round(metrics.homogeneity_score(labels_true, labels_pred), 2),
            'Completeness': round(metrics.completeness_score(labels_true, labels_pred), 2),
        }
    else:
        return {
            'Silhouette': round(metrics.silhouette_score(X, labels_pred), 2),
            'Calinski-Harabaz': round(metrics.calinski_harabaz_score(X, labels_pred), 2)
        }


def statistics(clustering):
    result = {}

    for label in clustering.keys():
        cluster = clustering[label]

        labels_true = [item.get('label_true') for item in cluster]
        counts = Counter(labels_true)
        label_true, count = counts.most_common(1)[0]

        x = np.array([item['x'] for item in cluster])

        result[label] = {
            'total': len(cluster),
            'mean': x.mean(0).round(2).astype(float).tolist(),
            'std': x.std(0).round(2).astype(float).tolist(),
        }

        if None not in labels_true:  # classified data
            result[label].update({
                'label_true': label_true,
                'label_true_count': count,
            })

    return result
