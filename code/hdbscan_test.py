import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as data
import hdbscan

np.random.seed(170)

PLOT_KWDS = {'alpha': 0.5, 's': 80, 'linewidths': 0}
COLORS = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628']

moons, _ = data.make_moons(n_samples=50, noise=0.05)
blobs, _ = data.make_blobs(n_samples=50, centers=[(-0.75, 2.25), (1.0, 2.0)], cluster_std=0.25)
test_data = np.vstack([moons, blobs])

fig = plt.figure(figsize=(8, 3))
fig.subplots_adjust(left=0.02, right=0.97, bottom=0.05, top=0.9)

ax = fig.add_subplot(1, 2, 1)
ax.scatter(test_data.T[0], test_data.T[1], color='b', **PLOT_KWDS)
ax.set_title('(a)')
ax.set_xticks(())
ax.set_yticks(())

clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
clusterer.fit(test_data)

ax = fig.add_subplot(1, 2, 2)
clusterer.minimum_spanning_tree_.plot(axis=ax, edge_cmap='viridis', edge_alpha=0.6, node_size=20, edge_linewidth=1)
ax.set_title('(b)')
ax.set_xticks(())
ax.set_yticks(())

fig.show()

fig = plt.figure(figsize=(8, 4))
fig.subplots_adjust(left=0.08, right=1.06, bottom=0.05, top=0.97)
ax = fig.add_subplot(1, 1, 1)
clusterer.single_linkage_tree_.plot(axis=ax, cmap='viridis', colorbar=True)
fig.show()

fig = plt.figure(figsize=(8, 3))
fig.subplots_adjust(left=0.04, right=0.98, bottom=0.05, top=0.9)

ax = fig.add_subplot(1, 2, 1)
clusterer.condensed_tree_.plot(axis=ax, select_clusters=True, selection_palette=COLORS)
ax.set_title('(a)')
ax.set_xticks(())
ax.set_yticks(())

ax = fig.add_subplot(1, 2, 2)
ax.scatter(test_data.T[0], test_data.T[1], c=np.array(COLORS)[clusterer.labels_], **PLOT_KWDS)
ax.set_title('(b)')
ax.set_xticks(())
ax.set_yticks(())

fig.show()
