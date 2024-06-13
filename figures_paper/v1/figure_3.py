import os
import phate
import numpy as np
from scipy.stats import entropy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.cluster import KMeans
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from pipeline_phate_clustering.null_model_sensibility_analysis.get_result_null_model import get_color_map


def to_percent(n=10):
    def function(y, position):
        # Ignore the passed in position. This has the effect of scaling the default
        # tick locations.
        s = str(np.array(100 * y / n, dtype=int))
        # The percent symbol needs escaping in latex
        if matplotlib.rcParams['text.usetex'] is True:
            return s + r'$\%$'
        else:
            return s + '%'
    return function

label_size = 12.0
label_col_size = 8.0

path = os.path.dirname(os.path.realpath(__file__)) + "/../../paper/result/default/"
nb_randomize_1 = 10000
significatif = 0.05 / (90 * 7)
range_region_label = np.arange(0, 11, 1)
# load value
data_null_model = []
for nb_rand in range(nb_randomize_1):
    data_null_model.append(np.load(path + "/histograms_region_" + str(nb_rand) + ".npy"))
data_patient = np.load(path + "/histograms_region.npy")
nb_cluster = data_patient.shape[0]

pvalue_cluster_all = []
entropy_values_all = []
for index, cluster_region in enumerate(data_patient):
    pvalue_all = np.sum(
        np.sum(np.array(data_null_model) > cluster_region, axis=0) / nb_randomize_1, axis=0) / nb_cluster
    significatif_high_all = pvalue_all > 1.0 - significatif
    significatif_low_all = pvalue_all < significatif
    significatif_all_all = np.logical_or(significatif_low_all, significatif_high_all)
    pvalue_cluster_all.append(
        [[pvalue_all], [significatif_all_all], [significatif_high_all], [significatif_low_all]])
for data in data_null_model:
    entropy_values_all.append(entropy(np.array(data).ravel(), base=None))
pvalue_cluster_all = np.array(pvalue_cluster_all)
cmap_red_blue = get_color_map()
cmap_black_white = ListedColormap(["white", "black", "white"], name='from_list', N=None)

avalanches_patterns = np.load('avalanches_pattern_example_fig_1.npy')
phate_operator = phate.PHATE(n_components=2, n_jobs=8, n_pca=None, decay=1.2, gamma=-1, knn=1, knn_dist='cosine', mds_dist='cosine')
Y_phate_example = phate_operator.fit_transform(avalanches_patterns)
cluster_phate = KMeans(n_clusters=2, random_state=12).fit_predict(Y_phate_example)


# load value
nb_randomize_2 = 100
significatif = 0.05 * 7
data_null_model = []
for nb_rand in range(nb_randomize_2):
    data_null_model.append(np.load(path + "/null_model/" + str(nb_rand) + "_histograms_region.npy"))
data_patient = np.load(path + "/histograms_region.npy")
pvalue_cluster_all_null_model = []
nb_cluster = data_patient.shape[0]
entropy_values_null_model = []
for index, cluster_region in enumerate(data_patient):
    pvalue_all = np.sum(
        np.sum(np.array(data_null_model) > cluster_region, axis=0) / nb_randomize_2, axis=0) / nb_cluster
    significatif_high_all = pvalue_all > 1.0 - significatif
    significatif_low_all = pvalue_all < significatif
    significatif_all_all = np.logical_or(significatif_low_all, significatif_high_all)
    pvalue_cluster_all_null_model.append(
        [[pvalue_all], [significatif_all_all], [significatif_high_all], [significatif_low_all]])
for data in data_null_model:
    entropy_values_null_model.append(entropy(np.array(data).ravel(), base=None))

pvalue_cluster_all_null_model = np.array(pvalue_cluster_all_null_model)



fig = plt.figure(figsize=(6.8, 3.4))
gs = GridSpec(5, 5, figure=fig, height_ratios=[2., 0.4, 0.2, 1, 1], width_ratios=[1, 0.1, 0.3, 1, 1])
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(avalanches_patterns.T, vmin=0.0, vmax=2.0, cmap=cmap_black_white, origin='lower')
ax1.set_xticks(np.arange(0, len(cluster_phate)))
ax1.set_xticklabels([])
ax1.grid(axis='x')
ax1.set_yticks(range_region_label[0::4]+0.5)
ax1.set_yticklabels(range_region_label[0::4])
ax1.set_ylim(ymax=len(range_region_label)-1, ymin=0)
ax1.annotate('A', xy=(-0.4, 0.8), xycoords='axes fraction', weight='bold', fontsize=label_size)
ax2 = fig.add_subplot(gs[1, 0])
im_y_phate = ax2.imshow(Y_phate_example.T, origin='lower')
ax2.set_xticks(np.arange(0, len(cluster_phate)-1)+0.5)
ax2.set_xticklabels([])
ax2.set_yticks([0, 1.1])
ax2.set_yticklabels([0, 1])
ax2.set_ylim(ymin=-0.5, ymax=1.5)
ax2.grid(axis='x')
ax2.annotate('B', xy=(-0.4, 0.9), xycoords='axes fraction', weight='bold', fontsize=label_size)
ax = fig.add_axes([0.27, 0.55, 0.01, 0.07])
colbar = fig.colorbar(im_y_phate, cax=ax, orientation='vertical')
colbar.ax.tick_params(axis='both', labelsize=label_col_size)

ax3 = fig.add_subplot(gs[2, 0])
ax3.imshow(np.expand_dims(cluster_phate, 1).T, cmap='Pastel1')
ax3.set_yticks([])
ax3.set_xticks(np.arange(0, len(cluster_phate)-1)+0.5)
ax3.set_xticklabels([])
ax3.grid(axis='x')
ax3.annotate('C', xy=(-0.4, 0.9), xycoords='axes fraction', weight='bold', fontsize=label_size)

ax4 = fig.add_subplot(gs[0, 1])
example_1 = avalanches_patterns[:, 9].T
np.random.shuffle(example_1)
ax4.imshow(np.expand_dims(example_1, 1), vmin=0.0, vmax=2.0, cmap=cmap_black_white, origin='lower')
ax4.set_yticks([])
ax4.set_xticks([])
ax4.annotate('D', xy=(-2., 0.8), xycoords='axes fraction', weight='bold', fontsize=label_size)

ax5 = fig.add_subplot(gs[0:3, 3])
y, x, _ = ax5.hist(entropy_values_all, bins=10, histtype='step', color='black')
ax5.set_ylim(ymax=0.3*nb_randomize_1)
ax5.vlines(entropy(data_patient.ravel(), base=None), ymin=0.0, ymax=0.3*nb_randomize_1, color='r')
ax5.yaxis.set_major_formatter(FuncFormatter(to_percent(n=nb_randomize_1)))
ax5.tick_params('both', pad=1)
ax5.annotate('E', xy=(-0.3, 0.9), xycoords='axes fraction', weight='bold', fontsize=label_size)
ax6 = fig.add_subplot(gs[0:3, 4])
y, x, _ = ax6.hist(entropy_values_null_model, bins=10, histtype='step', color='black')
ax6.set_ylim(ymax=0.3*nb_randomize_2)
ax6.vlines(entropy(data_patient.ravel(), base=None), ymin=0.0, ymax=0.3*nb_randomize_2, color='r')
ax6.yaxis.set_major_formatter(FuncFormatter(to_percent(n=nb_randomize_2)))
ax6.set_yticks([])
ax6.tick_params('both', pad=1)
ax6.annotate('F', xy=(0.0, 0.9), xycoords='axes fraction', weight='bold', fontsize=label_size)

ax7 = fig.add_subplot(gs[3, :])
ax7.imshow(pvalue_cluster_all[:, 2, :, :].swapaxes(0, 1)[0]-pvalue_cluster_all[:, 3, :, :].swapaxes(0, 1)[0],
           vmin=-1.0, vmax=1.0, cmap=cmap_red_blue, origin='lower')
ax7.tick_params('both', pad=1)
ax7.annotate('G', xy=(-0.05, 0.9), xycoords='axes fraction', weight='bold', fontsize=label_size)
ax8 = fig.add_subplot(gs[4, :])
ax8.imshow(pvalue_cluster_all_null_model[:, 2, :, :].swapaxes(0, 1)[0]-pvalue_cluster_all_null_model[:, 3, :, :].swapaxes(0, 1)[0],
           vmin=-1.0, vmax=1.0, cmap=cmap_red_blue, origin='lower')
ax8.tick_params('both', pad=1)
ax8.annotate('H', xy=(-0.05, 0.9), xycoords='axes fraction', weight='bold', fontsize=label_size)

plt.subplots_adjust(left=0.05, right=0.99, top=0.975, bottom=0.05, wspace=0.0, hspace=0.4)

plt.savefig('figure/figure_3_pre.png')
plt.show()