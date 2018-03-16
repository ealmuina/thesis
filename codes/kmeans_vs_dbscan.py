from itertools import cycle, islice

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets.samples_generator import make_blobs

sns.set()
sns.set_style('white')

# #############################################################################
# Generate sample data
np.random.seed(170)
n_clusters = 3

X, y = make_blobs(n_samples=5000)
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X = np.dot(X, transformation)

# #############################################################################
# Compute clustering with K-Means

k_means = KMeans(n_clusters=n_clusters)
k_means.fit(X)

# #############################################################################
# Compute clustering with DBSCAN

dbscan = DBSCAN(eps=0.15, min_samples=2)
dbscan.fit(X)

# #############################################################################
# Plot result

algorithms = (
    ('K-Means', k_means),
    ('DBSCAN', dbscan)
)

fig = plt.figure(figsize=(8, 3))
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)

for i, (name, algorithm) in enumerate(algorithms):
    y_pred = algorithm.labels_.astype(np.int)

    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00']),
                                  int(len(set(y_pred))))))

    ax = fig.add_subplot(1, 2, i + 1)
    for k, col in zip(set(y_pred), colors):
        my_members = y_pred == k
        ax.scatter(X[:, 0], X[:, 1], s=0.5, color=colors[y_pred])

    ax.set_title(name)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()
