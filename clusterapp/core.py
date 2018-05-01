import pathlib
from collections import Counter

import numpy as np
from hdbscan import HDBSCAN
from sklearn import metrics
from sklearn.cluster import KMeans, SpectralClustering, AffinityPropagation
from sklearn.manifold import MDS
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import scale, LabelEncoder

from clusterapp.features import Audio


class IdentityClustering:
    def __init__(self):
        self.labels_ = None

    def fit(self, _, y):
        self.labels_ = y


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
    def _extract_features(audio, features):
        current = []
        for feature in features:
            x = getattr(audio, feature)
            if feature in ('mfcc',):
                current.extend(x.mean(0))
            else:
                current.append(x.mean())
        return current

    @staticmethod
    def _parse_clustering_algo(algorithm, categories):
        return {
            'kmeans': KMeans(n_clusters=len(categories)),
            'spectral': SpectralClustering(n_clusters=len(categories)),
            'affinity': AffinityPropagation(),
            'gmm': GaussianMixture(n_components=len(categories)),
            'hdbscan': HDBSCAN(min_cluster_size=3),
            'none': IdentityClustering()
        }[algorithm]

    def cluster(self, categories, features, algorithm):
        algorithm = self._parse_clustering_algo(algorithm, categories)

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

        if X.shape[1] != 2:
            mds = MDS(n_components=2, random_state=0)
            X = mds.fit_transform(X)

        result = {}
        for i, label in enumerate(labels):
            label = str(label)
            items = result.get(label, [])
            items.append({
                'name': names[i],
                'label_true': y[i],
                'x': X[i, :],
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
        cluster = clustering[label]

        labels_true = [item['label_true'] for item in cluster]
        counts = Counter(labels_true)
        label_true, count = counts.most_common(1)[0]

        x = np.array([item['x'] for item in cluster])

        result[label] = {
            'label_true': label_true,
            'label_true_count': count,
            'total': len(cluster),
            'mean': x.mean(0).round(2).astype(float).tolist(),  # TODO Fix this!
            'std': x.std(0).round(2).astype(float).tolist(),
        }

    return result
