import os
import numpy as np
from sklearn.cluster import KMeans
import phate
import matplotlib.pyplot as plt
import seaborn as sns
# https://dburkhardt.github.io/tutorial/visualizing_phate/


path_data = os.path.dirname(os.path.realpath(__file__)) + '/../data/'
path_data =  "/home/kusch/Documents/project/patient_analyse/data/"
avalanches_bin = np.load(path_data + '/avalanches_selected_patient.npy', allow_pickle=True)

phate_operator = phate.PHATE(n_components=2, n_jobs=-2, decay=1.0, n_pca=5, gamma=-1, knn=5, knn_dist='cosine', mds_dist='cosine')
Y_phate = phate_operator.fit_transform(np.concatenate(avalanches_bin))

fig, ax = plt.subplots(1, figsize=(10,10))
sns.kdeplot(Y_phate[:,0], Y_phate[:,1], n_levels=100, shade=True, cmap='inferno', zorder=0, ax=ax)
ax.set_xlabel('PHATE 1', fontsize=18)
ax.set_ylabel('PHATE 2', fontsize=18)
ax.set_title('KDE - T cells', fontsize=20)
plt.show()

# graph = phate_operator.graph # kNNLandmarkGraph see graphtools
# cluster_graph = graph.clusters
# phate.plot.scatter2d(Y_phate, c=graph.clusters)
# plt.show()
#
# clusters = phate.cluster.kmeans(phate_operator, k=5)
# phate.plot.scatter2d(Y_phate, c=clusters, cmap=sns.husl_palette(5), s=1,
#                       figsize=(4.3,4), ticks=None, label_prefix='PHATE',
#                      legend_anchor=(1,1), fontsize=12, title='PHATE clusters')
# plt.show()

silhouette_scores=[phate.cluster.silhouette_score(phate_operator,i) for i in range(2,50)]
fig, ax = plt.subplots(1, figsize=(10,10))
plt.plot(silhouette_scores)
plt.show()