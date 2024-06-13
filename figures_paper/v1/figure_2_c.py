import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pipeline_phate_clustering.functions_helper.plot_brain import get_brain_mesh
from pipeline_phate_clustering.functions_helper.plot_brain_test import multiview_pyvista, get_region_select

label_size = 12.0
path = os.path.dirname(os.path.realpath(__file__)) + "/../../paper/result/default/"
kmeans_nb_cluster = 7
kmeans_seed = 123
PHATE_n_pca = 5
avalanches_pattern = np.load(path + "/avalanches.npy", allow_pickle=True)
PCA_fit_data_avalanche_pattern = PCA(n_components=PHATE_n_pca).fit_transform(np.concatenate(avalanches_pattern))
cluster_PCA = KMeans(n_clusters=kmeans_nb_cluster, random_state=kmeans_seed).fit_predict(PCA_fit_data_avalanche_pattern)
Y_phate = np.load(path + "/Phate.npy")
cluster_phate = KMeans(n_clusters=kmeans_nb_cluster, random_state=kmeans_seed).fit_predict(Y_phate)
cluster_vector = []
for i in range(kmeans_nb_cluster):
    cluster_vector.append(np.mean(np.concatenate(avalanches_pattern)[np.where(cluster_phate == i)], axis=0)[:78])
cluster_vector = np.array(cluster_vector)
region_selects = get_region_select()
lhfaces, rhfaces, vertices, region_map = get_brain_mesh()
cmap_brain = plt.cm.coolwarm
cmap_brain.set_bad(color='grey')
transitions = np.load(path + "/transition_all.npy")
colormap = plt.get_cmap('Accent', 7)
shaded = True

fig = plt.figure(figsize=(6.8, 6.8))
gs = GridSpec(12, 13, figure=fig, height_ratios=[0., 1, 1, 0., 1, 1, 0., 1, 1, 0., 1, 1],
              width_ratios=[1, 0.1, 1, 0.1, 1, 0.1, 0.3, 0.3, 0.3, 0.2, 0.3, 0.3, 0.3])
gs_1 = GridSpec(7, 9, figure=fig, height_ratios=[0.3, 1, 0.3, 1, 0.3, 1, 0.4],
                width_ratios=[1, 0.25, 1, 0.25, 1, 0.1, 1, 0.3, 1])
# pipeline
ax1 = fig.add_subplot(gs_1[1, 0])
ax1.scatter(PCA_fit_data_avalanche_pattern[:, 0],
            PCA_fit_data_avalanche_pattern[:, 1],
            color='black', s=0.05)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_ylabel('PCA dim 2', {"fontsize": label_size})
ax1.set_xlabel('PCA dim 1', {"fontsize": label_size})
ax1.annotate('A', xy=(-0.2, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)

ax2 = fig.add_subplot(gs_1[1, 2])
ax2.scatter(Y_phate[:, 0],
            Y_phate[:, 1],
            color='black', s=0.05)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_ylabel('PHATE dim 2', {"fontsize": label_size})
ax2.set_xlabel('PHATE dim 1', {"fontsize": label_size})
ax2.annotate('B', xy=(-0.2, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
ax3 = fig.add_subplot(gs_1[1, 4])
ax3.scatter(Y_phate[:, 0],
            Y_phate[:, 1],
            c=cluster_phate, s=0.05, cmap=colormap, vmin=0,  vmax=7)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_ylabel('PHATE dim 2', {"fontsize": label_size})
ax3.set_xlabel('PHATE dim 1', {"fontsize": label_size})
ax3.annotate('C', xy=(-0.2, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)

# compare PCA clusters
ax4 = fig.add_subplot(gs_1[3, 0])
im = ax4.scatter(PCA_fit_data_avalanche_pattern[:, 0],
            PCA_fit_data_avalanche_pattern[:, 1],
            c=cluster_PCA, s=0.05, cmap=colormap, vmin=0,  vmax=7)
ax4.set_xticks([])
ax4.set_yticks([])
ax4.set_ylabel('PCA dim 2', {"fontsize": label_size})
ax4.set_xlabel('PCA dim 1', {"fontsize": label_size})
ax4.annotate('D', xy=(-0.2, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
ax5 = fig.add_subplot(gs_1[3, 2])
ax5.scatter(PCA_fit_data_avalanche_pattern[:, 1],
            PCA_fit_data_avalanche_pattern[:, 2],
            c=cluster_PCA, s=0.05, cmap=colormap, vmin=0,  vmax=7)
ax5.set_xticks([])
ax5.set_yticks([])
ax5.set_ylabel('PCA dim 2', {"fontsize": label_size})
ax5.set_xlabel('PCA dim 1', {"fontsize": label_size})
ax5.annotate('E', xy=(-0.2, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
ax6 = fig.add_subplot(gs_1[3, 4])
ax6.scatter(PCA_fit_data_avalanche_pattern[:, 0],
            PCA_fit_data_avalanche_pattern[:, 2],
            c=cluster_PCA, s=0.05, cmap=colormap, vmin=0,  vmax=7)
ax6.set_xticks([])
ax6.set_yticks([])
ax6.set_ylabel('PCA dim 2', {"fontsize": label_size})
ax6.set_xlabel('PCA dim 1', {"fontsize": label_size})
ax6.annotate('F', xy=(-0.2, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
# compare Phate clusters
ax7 = fig.add_subplot(gs_1[5, 0])
ax7.scatter(Y_phate[:, 0],
            Y_phate[:, 1],
            c=cluster_phate, s=0.05, cmap=colormap, vmin=0,  vmax=7)
ax7.set_xticks([])
ax7.set_yticks([])
ax7.set_ylabel('PHATE dim 2', {"fontsize": label_size})
ax7.set_xlabel('PHATE dim 1', {"fontsize": label_size})
ax7.annotate('G', xy=(-0.2, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
ax8 = fig.add_subplot(gs_1[5, 2])
ax8.scatter(Y_phate[:, 1],
            Y_phate[:, 2],
            c=cluster_phate, s=0.05, cmap=colormap, vmin=0,  vmax=7)
ax8.set_xticks([])
ax8.set_yticks([])
ax8.set_ylabel('PHATE dim 2', {"fontsize": label_size})
ax8.set_xlabel('PHATE dim 1', {"fontsize": label_size})
ax8.annotate('H', xy=(-0.2, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
ax9 = fig.add_subplot(gs_1[5, 4])
ax9.scatter(Y_phate[:, 0],
            Y_phate[:, 2],
            c=cluster_phate, s=0.05, cmap=colormap, vmin=0,  vmax=7)
ax9.set_xticks([])
ax9.set_yticks([])
ax9.set_ylabel('PHATE dim 2', {"fontsize": label_size})
ax9.set_xlabel('PHATE dim 1', {"fontsize": label_size})
ax9.annotate('I', xy=(-0.2, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)

# brain image
axs10 = [fig.add_subplot(gs[1, 6]), fig.add_subplot(gs[1, 8]),
         fig.add_subplot(gs[2, 6]), fig.add_subplot(gs[2, 8]), fig.add_subplot(gs[1:3, 7])]
multiview_pyvista(axs10, vertices, lhfaces, rhfaces, region_map, cluster_vector[0], region_select=region_selects)
axs10[0].annotate('J', xy=(-0.2, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
axs11 = [fig.add_subplot(gs[1, 10]), fig.add_subplot(gs[1, 12]),
         fig.add_subplot(gs[2, 10]), fig.add_subplot(gs[2, 12]), fig.add_subplot(gs[1:3, 11])]
multiview_pyvista(axs11, vertices, lhfaces, rhfaces, region_map, cluster_vector[1], region_select=region_selects)
axs11[0].annotate('K', xy=(-0.2, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
axs12 = [fig.add_subplot(gs[4, 6]), fig.add_subplot(gs[4, 8]),
         fig.add_subplot(gs[5, 6]), fig.add_subplot(gs[5, 8]), fig.add_subplot(gs[4:6, 7])]
multiview_pyvista(axs12, vertices, lhfaces, rhfaces, region_map, cluster_vector[2], region_select=region_selects)
axs12[0].annotate('L', xy=(-0.2, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
axs13 = [fig.add_subplot(gs[4, 10]), fig.add_subplot(gs[4, 12]),
         fig.add_subplot(gs[5, 10]), fig.add_subplot(gs[5, 12]), fig.add_subplot(gs[4:6, 11])]
multiview_pyvista(axs13, vertices, lhfaces, rhfaces, region_map, cluster_vector[3], region_select=region_selects)
axs13[0].annotate('M', xy=(-0.2, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
axs14 = [fig.add_subplot(gs[7, 6]), fig.add_subplot(gs[7, 8]),
         fig.add_subplot(gs[8, 6]), fig.add_subplot(gs[8, 8]), fig.add_subplot(gs[7:9, 7])]
multiview_pyvista(axs14, vertices, lhfaces, rhfaces, region_map, cluster_vector[4], region_select=region_selects)
axs14[0].annotate('N', xy=(-0.2, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
axs15 = [fig.add_subplot(gs[7, 10]), fig.add_subplot(gs[7, 12]),
         fig.add_subplot(gs[8, 10]), fig.add_subplot(gs[8, 12]), fig.add_subplot(gs[7:9, 11])]
multiview_pyvista(axs15, vertices, lhfaces, rhfaces, region_map, cluster_vector[5], region_select=region_selects)
axs15[0].annotate('O', xy=(-0.2, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
axs16 = [fig.add_subplot(gs[10, 6]), fig.add_subplot(gs[10, 8]),
         fig.add_subplot(gs[11, 6]), fig.add_subplot(gs[11, 8]), fig.add_subplot(gs[10:12, 7])]
multiview_pyvista(axs16, vertices, lhfaces, rhfaces, region_map, cluster_vector[6], region_select=region_selects)
axs16[0].annotate('P', xy=(-0.2, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)

ax17 = fig.add_subplot(gs[10:12, 10:13])
im_transition = ax17.imshow(transitions)
divider = make_axes_locatable(ax17)
cax = divider.append_axes("right", size="5%", pad=0.05)
colorbar_transition = fig.colorbar(im_transition, cax=cax)
ax17.set_xticks([0.1, 3.1, 6.1])
ax17.set_xticklabels([0, 3, 6])
ax17.set_yticks([0.1, 3.1, 6.1])
ax17.set_yticklabels([0, 3, 6])
ax17.annotate('Q', xy=(-0.2, 0.99), xycoords='axes fraction', weight='bold', fontsize=label_size)

ax = fig.add_axes([0.04, 0.03, 0.5, 0.02])
colbar = fig.colorbar(im, cax=ax, orientation='horizontal')
colbar.set_ticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
colbar.set_ticklabels([0, 1, 2, 3, 4, 5, 6])

plt.subplots_adjust(left=0.04, right=0.95, top=1., bottom=0., hspace=0., wspace=0.)
plt.savefig('figure/figure_2_c_pre.png')
plt.show()
