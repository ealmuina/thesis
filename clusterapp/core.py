import pathlib

import numpy as np
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import scale

from clusterapp.features import Audio


class IdentityClustering:
    def __init__(self):
        self.labels_ = None

    def fit(self, X, y):
        self.labels_ = [name.split('-')[0] for name in y]


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

    def get_features(self, categories, features, clusterer):
        if clusterer in ('kmeans',):
            clusterer = CLUSTERING[clusterer](n_clusters=len(categories))
        elif clusterer in ('gmm',):
            clusterer = CLUSTERING[clusterer](n_components=len(categories))
        elif clusterer in ('hdbscan',):
            clusterer = CLUSTERING[clusterer](min_cluster_size=3)
        else:
            clusterer = CLUSTERING[clusterer]()

        X, y = [], []
        for cat in categories:
            for audio in self.segments[cat]:
                X.append([
                    getattr(audio, features[0]).mean(),
                    getattr(audio, features[1]).mean()
                ])
                y.append(audio.name)
        X = np.array(X, dtype=np.float64)

        clusterer.fit(scale(X) if len(X) else X, y)
        labels = clusterer.labels_ if hasattr(clusterer, 'labels_') else clusterer.predict(X)

        result = {}
        for i, label in enumerate(labels):
            label = str(label)
            items = result.get(label, [])
            items.append({
                'name': y[i],
                'x': X[i, 0],
                'y': X[i, 1],
            })
            result[label] = items

        return result


def statistics(clustering):
    result = {}
    for label in clustering.keys():
        x = np.array([item['x'] for item in clustering[label]])
        y = np.array([item['y'] for item in clustering[label]])

        result[label] = {
            'x_mean': x.mean().round(2),
            'x_var': x.var().round(2),
            'y_mean': y.mean().round(2),
            'y_var': y.var().round(2)
        }
    return result
