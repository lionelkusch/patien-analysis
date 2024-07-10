import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib.gridspec import GridSpec
from helper_function import add_arrows


label_size = 12.0
tickfont_size = 10.0
path = os.path.dirname(os.path.realpath(__file__)) + "/../../paper/result/default/"
kmeans_nb_cluster = 7
kmeans_seed = 123
PHATE_n_pca = 5
avalanches_pattern = np.load(path + "/avalanches.npy", allow_pickle=True)
PCA_fit_data_avalanche_pattern = PCA(n_components=PHATE_n_pca).fit_transform(np.concatenate(avalanches_pattern))
cluster_PCA = KMeans(n_clusters=kmeans_nb_cluster, random_state=kmeans_seed).fit_predict(PCA_fit_data_avalanche_pattern)
Y_phate = np.load(path + "/Phate.npy")
cluster_phate = KMeans(n_clusters=kmeans_nb_cluster, random_state=kmeans_seed).fit_predict(Y_phate)
cmap_brain = plt.cm.coolwarm
cmap_brain.set_bad(color='grey')
colormap = plt.get_cmap('Accent', 7)

fig = plt.figure(figsize=(6.8, 3.4))
gs_1 = GridSpec(2, 3, figure=fig)
# compare PCA clusters
ax1 = fig.add_subplot(gs_1[0, 0])
im = ax1.scatter(PCA_fit_data_avalanche_pattern[:, 0],
            PCA_fit_data_avalanche_pattern[:, 1],
            c=cluster_PCA, s=0.05, cmap=colormap, vmin=0,  vmax=7)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.annotate('A', xy=(-0.1, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
add_arrows(PCA_fit_data_avalanche_pattern[:, 0], PCA_fit_data_avalanche_pattern[:, 1], ax1, "PCA 1", "PCA 2")
ax1.axis('off')
ax2 = fig.add_subplot(gs_1[0, 1])
ax2.scatter(PCA_fit_data_avalanche_pattern[:, 1],
            PCA_fit_data_avalanche_pattern[:, 2],
            c=cluster_PCA, s=0.05, cmap=colormap, vmin=0,  vmax=7)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.annotate('B', xy=(-0.1, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
add_arrows(PCA_fit_data_avalanche_pattern[:, 1], PCA_fit_data_avalanche_pattern[:, 2], ax2, "PCA 2", "PCA 3")
ax2.axis('off')
ax3 = fig.add_subplot(gs_1[0, 2])
ax3.scatter(PCA_fit_data_avalanche_pattern[:, 0],
            PCA_fit_data_avalanche_pattern[:, 2],
            c=cluster_PCA, s=0.05, cmap=colormap, vmin=0,  vmax=7)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.annotate('C', xy=(-0.1, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
add_arrows(PCA_fit_data_avalanche_pattern[:, 0], PCA_fit_data_avalanche_pattern[:, 2], ax3, "PCA 1", "PCA 3")
ax3.axis('off')
# compare Phate clusters
ax4 = fig.add_subplot(gs_1[1, 0])
ax4.scatter(Y_phate[:, 0],
            Y_phate[:, 1],
            c=cluster_phate, s=0.05, cmap=colormap, vmin=0,  vmax=7)
ax4.set_xticks([])
ax4.set_yticks([])
ax4.annotate('D', xy=(-0.1, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
add_arrows(Y_phate[:, 0], Y_phate[:, 1], ax4, "PHATE 1", "PHATE 2")
ax4.axis('off')
ax5 = fig.add_subplot(gs_1[1, 1])
ax5.scatter(Y_phate[:, 1],
            Y_phate[:, 2],
            c=cluster_phate, s=0.05, cmap=colormap, vmin=0,  vmax=7)
ax5.set_xticks([])
ax5.set_yticks([])
ax5.annotate('E', xy=(-0.1, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
add_arrows(Y_phate[:, 1], Y_phate[:, 2], ax5, "PHATE 2", "PHATE 3")
ax5.axis('off')
ax6 = fig.add_subplot(gs_1[1, 2])
ax6.scatter(Y_phate[:, 0],
            Y_phate[:, 2],
            c=cluster_phate, s=0.05, cmap=colormap, vmin=0,  vmax=7)
ax6.set_xticks([])
ax6.set_yticks([])
ax6.annotate('F', xy=(-0.1, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
add_arrows(Y_phate[:, 0], Y_phate[:, 2], ax6, "PHATE 1", "PHATE 3")
ax6.axis('off')

ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
colbar = fig.colorbar(im, cax=ax, orientation='vertical')
colbar.set_ticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
colbar.set_ticklabels([1, 2, 3, 4, 5, 6, 7])
colbar.ax.yaxis.set_tick_params(pad=0.1, labelsize=tickfont_size)
colbar.ax.set_ylabel('#cluster', {"fontsize": label_size}, labelpad=0)


plt.subplots_adjust(left=0.04, right=0.92, top=0.99, bottom=0.07, hspace=0.1, wspace=0.1)
plt.savefig('figure/figure_3_pre.png')
plt.savefig('figure/figure_3_pre.svg')
plt.show()
