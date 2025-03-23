import numpy as np
import os
import matplotlib.pyplot as plt

titlefont_size = 12.0
tickfont_size = 10.0
labelfont_size = 11.0
letter_font_size = 12


path_root = os.path.dirname(os.path.realpath(__file__)) + '/../../'
path_phate = path_root + "/paper/result/default/"
avalanches_bin = np.load(path_phate + '/avalanches.npy', allow_pickle=True)
histograms_region_phate = np.load(path_phate + "/histograms_region.npy")
cluster_vector_phate = histograms_region_phate / histograms_region_phate.max(axis=0).reshape(1, len(avalanches_bin[0][0]))
order_phate = np.argsort(np.sum(histograms_region_phate, axis=1))
cluster_vector_phate = cluster_vector_phate[order_phate]

path_spherical = path_root + "/paper/result/spectral_cosine/"
histograms_region_spherical = np.load(path_spherical + "/spectral_cosine7/histograms_region.npy")
cluster_vector_spherical = histograms_region_spherical / histograms_region_spherical.max(axis=0).reshape(1, len(avalanches_bin[0][0]))
order_spherical = np.argsort(np.sum(histograms_region_spherical, axis=1))
cluster_vector_spherical = cluster_vector_spherical[order_spherical]


title_1 = 'Phate'
title_2 = 'Spectral clustering'
title_3 = 'Difference'
vmin = 0.0
vmax = 100.0
fontsize = 2
cmap = 'viridis'
cmap_diff = 'seismic'
cluster_1 = cluster_vector_phate.T * 100
order_cluster_1 = order_phate
cluster_2 = cluster_vector_spherical.T * 100
order_cluster_2 = order_spherical

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6.4, 8.4))

im_1 = ax1.imshow(cluster_1, vmin=vmin, vmax=vmax, cmap=cmap)
# plt.colorbar(im, ax=ax1)
ax1.set_xticks(np.arange(0, len(order_cluster_1)))
ax1.set_xticklabels(order_cluster_1)
ax1.set_title(title_1, {"fontsize": titlefont_size})
ax1.tick_params(which='both', labelsize=tickfont_size)
ax1.set_xlabel('# cluster')
ax1.set_ylabel('brain region')

im_2 = ax2.imshow(cluster_2, vmin=vmin, vmax=vmax, cmap=cmap)
ax2.set_xticks(np.arange(0, len(order_cluster_2)))
ax2.set_xticklabels(order_cluster_2)
ax2.set_title(title_2, {"fontsize": titlefont_size})
ax2.tick_params(which='both', labelsize=tickfont_size)
ax2.set_xlabel('# cluster')
ax2.set_ylabel('brain region')
im = ax2.imshow(cluster_2, vmin=vmin, vmax=vmax, cmap=cmap)

if cluster_2.shape[0] == cluster_1.shape[0]:
        im_3 = ax3.imshow(cluster_1-cluster_2, vmin=-vmax, vmax=vmax, cmap=cmap_diff)
elif cluster_2.shape[0] < cluster_1.shape[0]:
        im_3 = ax3.imshow(cluster_1[-cluster_2.shape[0]:]-cluster_2, vmin=-vmax, vmax=vmax, cmap=cmap_diff)
elif cluster_2.shape[0] > cluster_1.shape[0]:
        im_3 = ax3.imshow(cluster_1-cluster_2[-cluster_1.shape[0]:], vmin=-vmax, vmax=vmax, cmap=cmap_diff)
ax3.set_xticks([])
# plt.colorbar(im, ax=ax3)
ax3.set_title(title_3, {"fontsize": titlefont_size})
ax3.set_ylabel('brain region')
ax3.set_xlabel('# cluster')


cax = fig.add_axes([0.18, 0.06, 0.3, 0.01])
colorbar_pattern = fig.colorbar(im_1, cax=cax, orientation='horizontal')
colorbar_pattern.ax.xaxis.set_tick_params(labelsize=tickfont_size)
colorbar_pattern.ax.set_xlabel('% activity in a cluster', {"fontsize": labelfont_size}, labelpad=2)

cax = fig.add_axes([0.75, 0.06, 0.2, 0.01])
colorbar_diff = fig.colorbar(im_3, cax=cax, orientation='horizontal')
colorbar_diff.ax.xaxis.set_tick_params(labelsize=tickfont_size)
colorbar_diff.ax.set_xlabel('difference between %', {"fontsize": labelfont_size}, labelpad=2)


plt.subplots_adjust(top=0.97, right=1., left=0.0, bottom=0.13)
plt.savefig('figure_3/SP_17_compare_with_spectral.png', dpi=600)
plt.show()