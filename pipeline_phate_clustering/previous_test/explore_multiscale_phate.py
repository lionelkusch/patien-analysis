import matplotlib.pyplot as plt
import multiscale_phate
import phate
import os
import numpy as np

path_data = os.path.dirname(os.path.realpath(__file__)) + '/../data/'
path_data =  "/home/kusch/Documents/project/patient_analyse/data/"
avalanches_bin = np.load(path_data + '/avalanches_selected_patient.npy', allow_pickle=True)
mp_op = multiscale_phate.Multiscale_PHATE( n_jobs=-2, decay=1.0, n_pca=5, gamma=-1, knn=5, knn_dist='cosine', mds_dist='cosine')
mp_embedding, mp_clusters, mp_sizes = mp_op.fit_transform(np.concatenate(avalanches_bin))

# Plot optimal visualization
phate.plot.scatter2d(mp_embedding, s = mp_sizes, c = mp_clusters,
                      fontsize=16, ticks=False,label_prefix="Multiscale PHATE", figsize=(16,12))

plt.figure()
ax = plt.plot(mp_op.gradient)
ax = plt.scatter(mp_op.levels, mp_op.gradient[mp_op.levels], c = 'r', s=100)


tree =  mp_op.build_tree()
plt.figure()
phate.plot.scatter3d(tree, c = tree[:,2], s= 50,
                      fontsize=16, ticks=False, figsize=(10,10))
plt.figure()
phate.plot.scatter3d(tree, c = tree[:,2], s= 50,
                      fontsize=16, ticks=False, figsize=(10,10))


coarse_embedding, coarse_clusters, coarse_sizes = mp_op.transform(visualization_level = mp_op.levels[0],
                                                                      cluster_level = mp_op.levels[-1])
phate.plot.scatter2d(coarse_embedding, s = 100*np.sqrt(coarse_sizes), c = coarse_clusters,
                          fontsize=16, ticks=False,label_prefix="Multiscale PHATE", figsize=(10,8))

plt.show()
