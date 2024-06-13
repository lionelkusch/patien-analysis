import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
transitions = np.load(path + "/transition_all.npy")
colormap = plt.get_cmap('Accent', 7)

fig = plt.figure(figsize=(6.8, 6.8))
# pipeline
ax1 = plt.subplot(3, 5, 1)
ax1.scatter(PCA_fit_data_avalanche_pattern[:, 0],
            PCA_fit_data_avalanche_pattern[:, 1],
            color='black', s=0.05)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_ylabel('PCA dim 2', {"fontsize": label_size})
ax1.set_xlabel('PCA dim 1', {"fontsize": label_size})
ax1.annotate('A', xy=(-0.2, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
ax2 = plt.subplot(3, 5, 2)
ax2.scatter(Y_phate[:, 0],
            Y_phate[:, 1],
            color='black', s=0.05)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_ylabel('PHATE dim 2', {"fontsize": label_size})
ax2.set_xlabel('PHATE dim 1', {"fontsize": label_size})
ax2.annotate('B', xy=(-0.2, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
ax3 = plt.subplot(3, 5, 3)
ax3.scatter(Y_phate[:, 0],
            Y_phate[:, 1],
            c=cluster_phate, s=0.05, cmap=colormap, vmin=0,  vmax=7)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_ylabel('PHATE dim 2', {"fontsize": label_size})
ax3.set_xlabel('PHATE dim 1', {"fontsize": label_size})
ax3.annotate('C', xy=(-0.2, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)

# compare PCA clusters
ax4 = plt.subplot(3, 5, 6)
im = ax4.scatter(PCA_fit_data_avalanche_pattern[:, 0],
            PCA_fit_data_avalanche_pattern[:, 1],
            c=cluster_PCA, s=0.05, cmap=colormap, vmin=0,  vmax=7)
ax4.set_xticks([])
ax4.set_yticks([])
ax4.set_ylabel('PCA dim 2', {"fontsize": label_size})
ax4.set_xlabel('PCA dim 1', {"fontsize": label_size})
ax4.annotate('D', xy=(-0.2, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
ax5 = plt.subplot(3, 5, 7)
ax5.scatter(PCA_fit_data_avalanche_pattern[:, 1],
            PCA_fit_data_avalanche_pattern[:, 2],
            c=cluster_PCA, s=0.05, cmap=colormap, vmin=0,  vmax=7)
ax5.set_xticks([])
ax5.set_yticks([])
ax5.set_ylabel('PCA dim 2', {"fontsize": label_size})
ax5.set_xlabel('PCA dim 1', {"fontsize": label_size})
ax5.annotate('E', xy=(-0.2, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
ax6 = plt.subplot(3, 5, 8)
ax6.scatter(PCA_fit_data_avalanche_pattern[:, 0],
            PCA_fit_data_avalanche_pattern[:, 2],
            c=cluster_PCA, s=0.05, cmap=colormap, vmin=0,  vmax=7)
ax6.set_xticks([])
ax6.set_yticks([])
ax6.set_ylabel('PCA dim 2', {"fontsize": label_size})
ax6.set_xlabel('PCA dim 1', {"fontsize": label_size})
ax6.annotate('F', xy=(-0.2, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
# compare Phate clusters
ax7 = plt.subplot(3, 5, 11)
ax7.scatter(Y_phate[:, 0],
            Y_phate[:, 1],
            c=cluster_phate, s=0.05, cmap=colormap, vmin=0,  vmax=7)
ax7.set_xticks([])
ax7.set_yticks([])
ax7.set_ylabel('PHATE dim 2', {"fontsize": label_size})
ax7.set_xlabel('PHATE dim 1', {"fontsize": label_size})
ax7.annotate('G', xy=(-0.2, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
ax8 = plt.subplot(3, 5, 12)
ax8.scatter(Y_phate[:, 1],
            Y_phate[:, 2],
            c=cluster_phate, s=0.05, cmap=colormap, vmin=0,  vmax=7)
ax8.set_xticks([])
ax8.set_yticks([])
ax8.set_ylabel('PHATE dim 2', {"fontsize": label_size})
ax8.set_xlabel('PHATE dim 1', {"fontsize": label_size})
ax8.annotate('H', xy=(-0.2, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
ax9 = plt.subplot(3, 5, 13)
ax9.scatter(Y_phate[:, 0],
            Y_phate[:, 2],
            c=cluster_phate, s=0.05, cmap=colormap, vmin=0,  vmax=7)
ax9.set_xticks([])
ax9.set_yticks([])
ax9.set_ylabel('PHATE dim 2', {"fontsize": label_size})
ax9.set_xlabel('PHATE dim 1', {"fontsize": label_size})
ax9.annotate('I', xy=(-0.2, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)

# brain image
ax10 = plt.subplot(4, 5, 4)
ax10.imshow(mpimg.imread(path + 'figure/k_7cluster_0.png'))
ax10.axis('off')
ax10.annotate('J', xy=(-0.2, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
ax11 = plt.subplot(4, 5, 5)
ax11.imshow(mpimg.imread(path + 'figure/k_7cluster_1.png'))
ax11.axis('off')
ax11.annotate('K', xy=(-0.2, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
ax12 = plt.subplot(4, 5, 9)
ax12.imshow(mpimg.imread(path + 'figure/k_7cluster_2.png'))
ax12.axis('off')
ax12.annotate('L', xy=(-0.2, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
ax13 = plt.subplot(4, 5, 10)
ax13.imshow(mpimg.imread(path + 'figure/k_7cluster_3.png'))
ax13.axis('off')
ax13.annotate('M', xy=(-0.2, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
ax14 = plt.subplot(4, 5, 14)
ax14.imshow(mpimg.imread(path + 'figure/k_7cluster_4.png'))
ax14.axis('off')
ax14.annotate('N', xy=(-0.2, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
ax15 = plt.subplot(4, 5, 15)
ax15.imshow(mpimg.imread(path + 'figure/k_7cluster_5.png'))
ax15.axis('off')
ax15.annotate('O', xy=(-0.2, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
ax16 = plt.subplot(4, 5, 19)
ax16.imshow(mpimg.imread(path + 'figure/k_7cluster_6.png'))
ax16.axis('off')
ax16.annotate('P', xy=(-0.2, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)

ax17 = plt.subplot(4, 5, 20)
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

plt.subplots_adjust(left=0.04, right=0.95, top=0.995, bottom=0.09, hspace=0.3, wspace=0.3)
plt.savefig('figure/figure_2_pre.png')
# plt.show()
