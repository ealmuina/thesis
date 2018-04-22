import os
import pathlib

import numpy as np
import pylab as pl
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding, TSNE
from sklearn.preprocessing import LabelEncoder, scale

from clusterapp.features import Audio

TESTING_DIR = '../sounds/testing'
CATEGORIES = [
    'bufo bufo A',
    'crex crex A',
    'eumodicogryllus burdigalensis A',
    'jynx torquilla A',
    'nyctalus noctula A'
]


def load_data():
    X, y = [], []

    for c in CATEGORIES:
        for i in range(1, 11):
            path = pathlib.Path(os.path.join(TESTING_DIR, '%s-%d.wav' % (c, i)))
            audio = Audio(path)
            X.append(audio.mfcc.mean(0))
            y.append(c)

    X = np.array(X, dtype=np.float64)

    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    print(le.classes_)

    return scale(X), y


def main():
    sns.set()
    sns.set_style('white')
    np.random.seed(170)

    X, y = load_data()

    fig, ax = pl.subplots(1, 1, figsize=(8, 4))
    fig.subplots_adjust(left=0.05, right=0.97)
    plot(X, y, PCA(n_components=2), ax, 'PCA', True)
    fig.show()

    fig, ax = pl.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(left=0.07, right=0.97)

    algorithms = [
        (MDS, 'MDS', (0, 0)),
        (Isomap, 'Isomap', (0, 1)),
        (LocallyLinearEmbedding, 'LLE', (1, 0)),
        (TSNE, 't-SNE', (1, 1))
    ]

    for algorithm, name, pos in algorithms:
        plot(X, y, algorithm(n_components=2), ax[pos], name)
    fig.show()


def plot(X, y, algorithm, ax, title, show_legend=False):
    X = algorithm.fit_transform(X)

    for label in set(y):
        x = X[y == label, :]
        ax.plot(x[:, 0], x[:, 1], 'o', label=CATEGORIES[label])

    ax.set_title(title)
    if show_legend:
        ax.legend()


if __name__ == '__main__':
    main()
