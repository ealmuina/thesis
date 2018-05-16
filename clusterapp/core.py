import itertools
import pathlib
import warnings
from collections import Counter
from math import factorial

import numpy as np
from hdbscan import HDBSCAN
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans, SpectralClustering, AffinityPropagation
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances_argmin
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import scale, LabelEncoder
from tqdm import tqdm, trange

from .features.audio import Audio
from .utils import std_out_err_redirect_tqdm

warnings.filterwarnings("ignore")

RANDOM_STATE = 0


class Classifier:
    def __init__(self, clustering):
        self.centroids = []
        self.labels = []

        for label in clustering.keys():
            cluster = clustering[label]
            X = np.array([item['x'] for item in cluster])

            self.centroids.append(X.mean(0))
            self.labels.append(label)

        self.centroids = np.array(self.centroids)
        self.labels = np.array(self.labels)

    def predict(self, files, features):
        X = []
        for file in files:
            audio = Audio(file.stream)
            x = _extract_features(audio, features)
            X.append(x)

        X = np.array(X)
        X_2d = X
        if X.shape[1] != 2:
            mds = MDS(n_components=2, random_state=RANDOM_STATE)
            X_2d = mds.fit_transform(X)
        centroid = pairwise_distances_argmin(X, self.centroids)

        return [{
            'name': files[i].filename,
            'x': float(X_2d[i, 0].round(2)),
            'y': float(X_2d[i, 1].round(2)),
            'label': self.labels[centroid[i]],
            'centroid': i,
        } for i in range(len(centroid))]


class IdentityClustering:
    def __init__(self):
        self.labels_ = None

    def fit(self, _, y=None):
        self.labels_ = y


class Library:
    def __init__(self, path):
        files = list(pathlib.Path(path).iterdir())
        self.files = []
        with std_out_err_redirect_tqdm() as orig_stdout:
            for file in tqdm(files, desc='Loading audio files', file=orig_stdout, dynamic_ncols=True):
                self.files.append((file, Audio(file, string_path=True)))

    @staticmethod
    def _parse_clustering_algo(algorithm, categories):
        if not isinstance(categories, int):
            categories = len(categories)
        return {
            'kmeans': MiniBatchKMeans(n_clusters=categories, random_state=RANDOM_STATE),
            'spectral': SpectralClustering(n_clusters=categories, n_jobs=-1, random_state=RANDOM_STATE),
            'affinity': AffinityPropagation(),
            'gmm': GaussianMixture(n_components=categories, random_state=RANDOM_STATE),
            'hdbscan': HDBSCAN(min_cluster_size=3),
            'none': IdentityClustering()
        }[algorithm]

    def _best_features(self, best, categories, features_set, algorithm, min_features, max_features):
        n = len(features_set)
        with std_out_err_redirect_tqdm() as orig_stdout:
            sizes = trange(
                min_features,
                max_features + 1,
                desc='Checking subsets of features',
                file=orig_stdout,
                dynamic_ncols=True
            )
            for r in sizes:
                k = factorial(n) / (factorial(r) * factorial(n - r))
                combinations = tqdm(
                    itertools.combinations(features_set, r),
                    total=int(k),
                    desc='Checking subsets of size %d' % r,
                    file=orig_stdout,
                    dynamic_ncols=True,
                    leave=False
                )
                for features in combinations:
                    try:  # TODO remove this try-except
                        _, scaled_X, _, labels_pred, labels_true = self.predict(categories, features, algorithm)
                    except:
                        print('Error extracting features: %s' % str(features))
                        continue
                    self._update_best_features(best, features, scaled_X, labels_true, labels_pred)

        clustering, scores, _ = self.cluster(categories, best['features'], algorithm)
        return clustering, scores, best['features']

    def _parse_data(self, categories, features, algorithm):
        raise NotImplementedError()

    def _update_best_features(self, best, features, scaled_X, labels_true, labels_pred):
        raise NotImplementedError()

    def cluster(self, categories, features, algorithm):
        X, scaled_X, names, labels_pred, labels_true = self.predict(categories, features, algorithm)

        X_2d = X
        if X.shape[1] != 2:
            mds = MDS(n_components=2, random_state=RANDOM_STATE)
            X_2d = mds.fit_transform(X)

        result = {}
        for i, label in enumerate(labels_pred):
            label = str(label)
            items = result.get(label, [])
            entry = {
                'name': names[i],
                'x': X[i, :],
                'x_2d': X_2d[i, :]
            }
            if labels_true:
                entry['label_true'] = labels_true[i]
            items.append(entry)
            result[label] = items

        return result, evaluate(scaled_X, labels_pred, labels_true), Classifier(result)

    def predict(self, categories, features, algorithm):
        algorithm = self._parse_clustering_algo(algorithm, categories)
        X, names, labels_true = self._parse_data(categories, features, algorithm)

        scaled_X = scale(X) if len(X) else X
        algorithm.fit(scaled_X, labels_true)
        labels_pred = algorithm.labels_ if hasattr(algorithm, 'labels_') else algorithm.predict(scaled_X)

        return X, scaled_X, names, labels_pred, labels_true


class ClassifiedLibrary(Library):
    def __init__(self, path):
        super().__init__(path)
        self.segments = {}

        for file, audio in self.files:
            category = file.name.split('-')[0]
            items = self.segments.get(category, [])
            items.append(audio)
            self.segments[category] = items

        self.categories = set(self.segments.keys())

    def _parse_data(self, categories, features, algorithm):
        X, labels_true, names = [], [], []
        for cat in categories:
            for audio in self.segments[cat]:
                X.append(_extract_features(audio, features))
                names.append(audio.name)
                labels_true.append(cat)
        X = np.array(X, dtype=np.float64)
        return X, names, labels_true

    def _update_best_features(self, best, features, scaled_X, labels_true, labels_pred):
        ami = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
        try:
            ch = metrics.calinski_harabaz_score(scaled_X, labels_pred)
        except ValueError:
            ch = -2 ** 31  # Likely because labels_pred had only one category. Just give it a bad score

        if ami > best['AMI'] or (ami == best['AMI'] and ch > best['CH']):
            best.update({
                'AMI': ami,
                'CH': ch,
                'features': list(features)
            })

    def best_features(self, categories, features_set, algorithm, min_features, max_features):
        best = {
            'AMI': 0,
            'CH': -2 ** 31
        }
        return self._best_features(best, categories, features_set, algorithm, min_features, max_features)


class UnclassifiedLibrary(Library):
    def __init__(self, path):
        super().__init__(path)
        self.segments = [
            audio for _, audio in self.files
        ]

    def _parse_data(self, categories, features, algorithm):
        X, names = [], []
        for audio in self.segments:
            X.append(_extract_features(audio, features))
            names.append(audio.name)
        X = np.array(X, dtype=np.float64)
        return X, names, None

    def _update_best_features(self, best, features, scaled_X, labels_true, labels_pred):
        try:
            ch = metrics.calinski_harabaz_score(scaled_X, labels_pred)
        except ValueError:
            ch = -2 ** 31  # Likely because labels_pred had only one category. Just give it a bad score

        if ch > best['CH']:
            best.update({
                'CH': ch,
                'features': list(features)
            })

    def best_features(self, categories, features_set, algorithm, min_features, max_features):
        best = {
            'CH': -2 ** 31
        }
        return self._best_features(best, categories, features_set, algorithm, min_features, max_features)


def _extract_features(audio, features):
    current = []
    for feature in features:
        x = getattr(audio, feature)
        if isinstance(x, np.ndarray):
            current.extend(x.mean(1))
        else:
            current.append(x)
    return current


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
