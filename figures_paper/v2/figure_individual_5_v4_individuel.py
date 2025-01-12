import os
import numpy as np
import phate
from scipy.stats import entropy
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches
import matplotlib
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


np.random.seed(42)
titlefont_size = 12.0
tickfont_size = 8.0
labelfont_size = 10.0
letter_font_size = 12
range_time = (900, 1400)
regions_select = (0, 90)
path_data = os.path.dirname(os.path.realpath(__file__)) + '/../../paper/result/default/'
Avalanches_human = np.load(path_data + 'avalanches.npy', allow_pickle=True)
avalanches_patterns_all = Avalanches_human[0][904:910, :]

cmap_black_white = ListedColormap(["white", "black", "white"], name='from_list', N=None)
cmap_blue_white = ListedColormap(["turquoise", "deepskyblue", "cornflowerblue"], name='from_list', N=None)
cmap_red_white = ListedColormap(["white", "lightsalmon", "orangered"], name='from_list', N=None)
cmap_red_blue = get_color_map()

# entropy
path = os.path.dirname(os.path.realpath(__file__)) + "/../../paper/result/default/"
nb_randomize_1 = 10000
significatif = 0.05
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
# load value
nb_randomize_2 = 100
significatif = 0.2
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

pvalues_prob_transisiton = np.load(path+'/model_diagonal.npy', allow_pickle=True)
diag_per_lows, diag_per_nos, diag_per_highs, no_diag_per_lows, no_diag_per_nos, no_diag_per_highs =\
    np.load(path+'/model_diagonalsignificatif.npy')


fig = plt.figure(figsize=(6.8, 5.6))
gs_null_1 = GridSpec(3, 3, figure=fig)
################################### null model 1 ###########################
# figure null model 1
ax = fig.add_subplot(gs_null_1[0, 0])
ax.set_title('Null model 1\nShuffle cluster label', {"fontsize": titlefont_size}, pad=16)
ax.set_axis_off()
plt.subplots_adjust(left=0.05, right=0.99, bottom=0.07, top=0.89, hspace=0.6)
plt.savefig('figure_5/v2/figure_5_1.svg', dpi=600)

gs_shuffle = GridSpec(4, 6, figure=fig, height_ratios=[0.9, 0.1, 1., 1.], hspace=0.05, wspace=0.25)
# avalanches_patterns_1 = Avalanches_human[0][105:112, :11]
# avalanches_patterns_1[1, 3] = 1
# avalanches_patterns_1[2, 3] = 1
# avalanches_patterns_1[4, 8] = 1
# avalanches_patterns_1[5, 9] = 1
avalanches_patterns_1 = np.load('avalanches_example.npy')
range_region_label = np.arange(0, avalanches_patterns_1.shape[0], 1)
phate_operator = phate.PHATE(n_components=2, n_jobs=8, n_pca=None, decay=1.2, gamma=-1, knn=1, knn_dist='cosine', mds_dist='cosine')
Y_phate_example = phate_operator.fit_transform(avalanches_patterns_1)
cluster_phate = KMeans(n_clusters=3, random_state=12).fit_predict(Y_phate_example)

fig = plt.figure(figsize=(6.8, 5.6))
# avalanches
ax = fig.add_subplot(gs_shuffle[0, 0])
ax.imshow(avalanches_patterns_1.T, vmin=0.0, vmax=2.0, cmap=cmap_black_white, origin='lower')
ax.set_xticks(np.arange(0.5, avalanches_patterns_1.shape[1]))
ax.set_xticklabels([])
ax.grid(axis='x')
ax.set_yticks([])
ax.set_ylim(ymax=len(range_region_label)-0.5, ymin=-0.5)
ax.set_ylabel('ROIs', {"fontsize": labelfont_size})
ax.tick_params(which='both', labelsize=tickfont_size)
plt.subplots_adjust(left=0.05, right=0.99, bottom=0.07, top=0.89, hspace=0.6)
plt.savefig('figure_5/v2/figure_5_2.svg', dpi=600)
# label
fig = plt.figure(figsize=(6.8, 5.6))
ax = fig.add_subplot(gs_shuffle[1, 0])
# cluster_phate_1 = [0, 1, 1, 0, 1, 1, 0]
data = np.expand_dims(cluster_phate, 1).T
ax.imshow(data, cmap=cmap_red_white)
for (j, i), label in np.ndenumerate(data):
    plt.text(i, j+0.2, np.around(label, 2), ha='center', va='center', fontsize=tickfont_size)
ax.set_yticks([])
ax.set_xticks(np.arange(0, len(cluster_phate))+0.5)
ax.set_xticklabels([])
ax.grid(axis='x')
ax.set_ylabel('# cluster')
plt.subplots_adjust(left=0.05, right=0.99, bottom=0.07, top=0.89, hspace=0.6)
plt.savefig('figure_5/v2/figure_5_3.svg', dpi=600)

fig = plt.figure(figsize=(6.8, 5.6))
ax = fig.add_subplot(gs_shuffle[0, 1])
ax.imshow(avalanches_patterns_1.T, vmin=0.0, vmax=2.0, cmap=cmap_black_white, origin='lower')
ax.set_xticks(np.arange(0.5, avalanches_patterns_1.shape[1]))
ax.set_xticklabels([])
ax.grid(axis='x')
ax.set_yticks([])
ax.set_ylim(ymax=len(range_region_label)-0.5, ymin=-0.5)
ax.tick_params(which='both', labelsize=tickfont_size)
plt.subplots_adjust(left=0.05, right=0.99, bottom=0.07, top=0.89, hspace=0.6)
plt.savefig('figure_5/v2/figure_5_4.svg', dpi=600)
# plot label
fig = plt.figure(figsize=(6.8, 5.6))
ax = fig.add_subplot(gs_shuffle[1, 1])
np.random.shuffle(data[0])
np.random.shuffle(data[0])
ax.imshow(data, cmap=cmap_red_white)
for (j, i), label in np.ndenumerate(data):
    plt.text(i, j+0.2, np.around(label, 2), ha='center', va='center', fontsize=tickfont_size)
ax.set_yticks([])
ax.set_xticks(np.arange(0, len(cluster_phate))+0.5)
ax.set_xticklabels([])
ax.set_xlabel('# avalanche pattern                    ', {"fontsize": labelfont_size}, labelpad=0)
ax.grid(axis='x')
plt.subplots_adjust(left=0.05, right=0.99, bottom=0.07, top=0.89, hspace=0.6)
plt.savefig('figure_5/v2/figure_5_5.svg', dpi=600)

# matrix significant
gs_significant = GridSpec(4, 4, figure=fig, height_ratios=[1.0, 0.3, 0.7, 1.],
                          width_ratios=[0.25, 2.25, 0.2, 0.6])
fig = plt.figure(figsize=(6.8, 5.6))
ax = fig.add_subplot(gs_significant[2, 0])
ax.imshow((pvalue_cluster_all[:, 2, :, :20].swapaxes(0, 1)[0]-pvalue_cluster_all[:, 3, :, :20].swapaxes(0, 1)[0]).T,
           vmin=-1.0, vmax=1.0, cmap=cmap_red_blue, origin='lower')
ax.tick_params('both', pad=1)
ax.set_xticks([0, 3, 6])
ax.set_xticklabels([1, 4, 7])
ax.set_xlabel('# cluster  ', {"fontsize": labelfont_size})
ax.set_ylabel('ROIs', {"fontsize": labelfont_size}, labelpad=0)
ax.set_yticks([])
ax.set_title('                    Significant cluster\n                    affiliation', {"fontsize": titlefont_size}, pad=13)
ax.tick_params(which='both', labelsize=tickfont_size)
for i in range(7):
    ax.annotate('...', xy=(i-1., 20.1), fontsize=titlefont_size, annotation_clip=False, rotation=90, va='bottom', ha='center')
plt.subplots_adjust(left=0.05, right=0.99, bottom=0.07, top=0.89, hspace=0.6)
plt.savefig('figure_5/v2/figure_5_6.svg', dpi=600)
########################### legend significant ##################################
gs_null_legend = GridSpec(8, 5, figure=fig, height_ratios=[0.9, 0.1, 0.6, 0.1, 0.1, 0.1, 0.4, 1.],
                          width_ratios=[0.25, 1., 1.25, 0.2, 0.6], bottom=0.005)
fig = plt.figure(figsize=(6.8, 5.6))
ax = fig.add_subplot(gs_null_legend[3, 1])
ax.imshow([[-1.0]], vmin=-1.0, vmax=1.0, cmap=cmap_red_blue, origin='lower')
ax.set_yticks([0])
ax.set_yticklabels(['significant\nactive'], fontdict={'fontsize': labelfont_size})
ax.set_xticks([])
ax.set_xlim(xmin=0, xmax=0.5)
ax = fig.add_subplot(gs_null_legend[4, 1])
ax.imshow([[0.0]], vmin=-1.0, vmax=1.0, cmap=cmap_red_blue, origin='lower')
ax.set_yticks([0])
ax.set_yticklabels(['no\nsignificant'], fontdict={'fontsize': labelfont_size})
ax.set_xticks([])
ax.set_xlim(xmin=0, xmax=0.5)
ax = fig.add_subplot(gs_null_legend[5, 1])
ax.imshow([[1.0]], vmin=-1.0, vmax=1.0, cmap=cmap_red_blue, origin='lower')
ax.set_yticks([0])
ax.set_yticklabels(['significant\ninactive'], fontdict={'fontsize': labelfont_size})
ax.set_xticks([])
ax.set_xlim(xmin=0, xmax=0.5)
plt.subplots_adjust(left=0.05, right=0.99, bottom=0.07, top=0.89, hspace=0.6)
plt.savefig('figure_5/v2/figure_5_7.svg', dpi=600)


# entropy
gs_entropy = GridSpec(3, 5, figure=fig, width_ratios=[0.2, 0.15, 0.85, 1.9, 1.], wspace=0.0)
d = .015  # how big to make the diagonal lines in axes coordinates
fig = plt.figure(figsize=(6.8, 5.6))
ax = fig.add_subplot(gs_entropy[2, 2])
y, x, _ = ax.hist(entropy_values_all, bins=10, histtype='step', color='black')
ax.set_ylim(ymax=0.3*nb_randomize_1+500)
ax.tick_params('both', pad=1)
ax.set_xlabel('entropy', {"fontsize": labelfont_size})
ax.spines['left'].set_visible(False)
ax.set_yticks([])
# arguments to pass plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (1-d, 1+d), **kwargs)
ax.plot((-d, +d), (-d, +d), **kwargs)
ax.tick_params('both', labelsize=tickfont_size)

ax = fig.add_subplot(gs_entropy[2, 1])
arr_1 = mpatches.FancyArrowPatch((entropy(data_patient.ravel(), base=None), 700.),
                                 (entropy(data_patient.ravel(), base=None), -100),
                                 color='r',
                                 arrowstyle='->,head_width=.15', mutation_scale=20)
arr_1.set_clip_on(False)
ax.add_patch(arr_1)
ax.annotate('data', (1.5, 1.0), xycoords=arr_1, ha='center', va='bottom', annotation_clip=False, color='red')
y, x, _ = ax.hist(entropy_values_all, bins=10, histtype='step', color='black')
ax.set_ylim(ymax=0.3*nb_randomize_1+500, ymin=0.0)
ax.yaxis.set_major_formatter(FuncFormatter(to_percent(n=nb_randomize_1)))
ax.spines['right'].set_visible(False)
ax.set_ylabel('% of distribution', {"fontsize": labelfont_size}, labelpad=0)
ax.set_xlim(xmax=(entropy(data_patient.ravel(), base=None)+0.005), xmin=(entropy(data_patient.ravel(), base=None)-0.005))
ax.set_xticks([np.around(entropy(data_patient.ravel(), base=None), decimals=2)])
ax.tick_params('x', pad=10)
kwargs.update(transform=ax.transAxes)  # switch to the bottom axes
ax.plot((0.95-d, 1.05+d), (-d, +d), **kwargs)
ax.plot((0.95-d, 1.05+d), (1-d, 1+d), **kwargs)
plt.subplots_adjust(wspace=0.02)
ax.tick_params('both', labelsize=tickfont_size)
plt.subplots_adjust(left=0.05, right=0.99, bottom=0.07, top=0.89, hspace=0.6)
plt.savefig('figure_5/v2/figure_5_8.svg', dpi=600)


################################### null model 3 ###########################
# example 1
fig = plt.figure(figsize=(6.8, 5.6))
ax = fig.add_subplot(gs_null_1[0, 2])
ax.set_title('Null model 3\nShuffle active region\nby avalanche pattern', {"fontsize": titlefont_size}, pad=2, y=1)
ax.set_axis_off()
plt.subplots_adjust(left=0.05, right=0.99, bottom=0.07, top=0.89, hspace=0.6)
plt.savefig('figure_5/v2/figure_5_9.svg', dpi=600)
fig = plt.figure(figsize=(6.8, 5.6))
ax = fig.add_subplot(gs_shuffle[0, 4])
ax.imshow(avalanches_patterns_1.T, vmin=0.0, vmax=2.0, cmap=cmap_black_white, origin='lower')
ax.set_xticks(np.arange(0.5, avalanches_patterns_1.shape[1]))
ax.set_xticklabels([])
ax.grid(axis='x')
ax.set_yticks([])
ax.set_ylim(ymax=len(range_region_label)-0.5, ymin=-0.5)
ax.tick_params(which='both', labelsize=tickfont_size)
plt.subplots_adjust(left=0.05, right=0.99, bottom=0.07, top=0.89, hspace=0.6)
plt.savefig('figure_5/v2/figure_5_10.svg', dpi=600)
# plot shuffles
fig = plt.figure(figsize=(6.8, 5.6))
ax = fig.add_subplot(gs_shuffle[0, 5])
avalanches_patterns_shuffle = []
for av in avalanches_patterns_1:
    av_copy = np.copy(av)
    np.random.shuffle(av_copy)
    avalanches_patterns_shuffle.append(av_copy)
ax.imshow(np.concatenate([avalanches_patterns_shuffle]).T, vmin=0.0, vmax=2.0, cmap=cmap_black_white, origin='lower')
ax.set_xticks(np.arange(0.5, avalanches_patterns_1.shape[1]))
ax.set_xticklabels([])
ax.grid(axis='x')
ax.set_yticks([])
ax.set_ylim(ymax=len(range_region_label)-0.5, ymin=-0.5)
ax.tick_params(which='both', labelsize=tickfont_size)
plt.subplots_adjust(left=0.05, right=0.99, bottom=0.07, top=0.89, hspace=0.6)
plt.savefig('figure_5/v2/figure_5_11.svg', dpi=600)

# plot label
fig = plt.figure(figsize=(6.8, 5.6))
ax = fig.add_subplot(gs_shuffle[1, 4])
# cluster_phate_1 = [0, 1, 1, 0, 1, 1, 0]
data = np.expand_dims(cluster_phate, 1).T
ax.imshow(data, cmap=cmap_red_white)
for (j, i), label in np.ndenumerate(data):
    plt.text(i, j+0.2, np.around(label, 2), ha='center', va='center', fontsize=tickfont_size)
ax.set_yticks([])
ax.set_xticks(np.arange(0, len(cluster_phate))+0.5)
ax.set_xticklabels([])
ax.grid(axis='x')
plt.subplots_adjust(left=0.05, right=0.99, bottom=0.07, top=0.89, hspace=0.6)
plt.savefig('figure_5/v2/figure_5_12.svg', dpi=600)
fig = plt.figure(figsize=(6.8, 5.6))
ax = fig.add_subplot(gs_shuffle[1, 5])
np.random.shuffle(data[0])
ax.imshow(data, cmap=cmap_blue_white)
for (j, i), label in np.ndenumerate(data):
    plt.text(i, j+0.2, np.around(label, 2), ha='center', va='center', fontsize=tickfont_size)
ax.set_yticks([])
ax.set_xticks(np.arange(0, len(cluster_phate))+0.5)
ax.set_xticklabels([])
ax.set_xlabel('# avalanche pattern                    ', {"fontsize": labelfont_size}, labelpad=0)
ax.grid(axis='x')
plt.subplots_adjust(left=0.05, right=0.99, bottom=0.07, top=0.89, hspace=0.6)
plt.savefig('figure_5/v2/figure_5_13.svg', dpi=600)

# matrix significant
fig = plt.figure(figsize=(6.8, 5.6))
ax = fig.add_subplot(gs_significant[2, 2])
ax.imshow((pvalue_cluster_all_null_model[:, 2, :, :20].swapaxes(0, 1)[0]-pvalue_cluster_all_null_model[:, 3, :, :20].swapaxes(0, 1)[0]).T,
          vmin=-1.0, vmax=1.0, cmap=cmap_red_blue, origin='lower')
ax.tick_params('both', pad=1)
ax.set_xticks([0, 3, 6])
ax.set_xticklabels([1, 4, 7])
ax.set_xlabel('# cluster  ', {"fontsize": labelfont_size})
ax.set_ylabel('ROIs', {"fontsize": labelfont_size}, labelpad=0)
ax.set_yticks([])
ax.set_title('                    Significant cluster\n                    affiliation', {"fontsize": titlefont_size}, pad=13)
ax.tick_params(which='both', labelsize=tickfont_size)
for i in range(7):
    ax.annotate('...', xy=(i-1., 20.1), fontsize=titlefont_size, annotation_clip=False, rotation=90, va='bottom', ha='center')
plt.subplots_adjust(left=0.05, right=0.99, bottom=0.07, top=0.89, hspace=0.6)
plt.savefig('figure_5/v2/figure_5_14.svg', dpi=600)
########################### legend significant ##################################
gs_null_legend = GridSpec(8, 6, figure=fig, height_ratios=[0.9, 0.1, 0.6, 0.1, 0.1, 0.1, 0.4, 1.],
                          width_ratios=[0.25, 1., 1.25, 0.2, 0.4, 0.2], bottom=0.005)
fig = plt.figure(figsize=(6.8, 5.6))
ax = fig.add_subplot(gs_null_legend[3, 5])
ax.imshow([[-1.0]], vmin=-1.0, vmax=1.0, cmap=cmap_red_blue, origin='lower')
ax.set_yticks([0])
ax.set_yticklabels(['significant\nactive'], fontdict={'fontsize': labelfont_size})
ax.set_xticks([])
ax.set_xlim(xmin=0, xmax=0.5)
ax = fig.add_subplot(gs_null_legend[4, 5])
ax.imshow([[0.0]], vmin=-1.0, vmax=1.0, cmap=cmap_red_blue, origin='lower')
ax.set_yticks([0])
ax.set_yticklabels(['no\nsignificant'], fontdict={'fontsize': labelfont_size})
ax.set_xticks([])
ax.set_xlim(xmin=0, xmax=0.5)
ax = fig.add_subplot(gs_null_legend[5, 5])
ax.imshow([[1.0]], vmin=-1.0, vmax=1.0, cmap=cmap_red_blue, origin='lower')
ax.set_yticks([0])
ax.set_yticklabels(['significant\ninactive'], fontdict={'fontsize': labelfont_size})
ax.set_xticks([])
ax.set_xlim(xmin=0, xmax=0.5)
plt.subplots_adjust(left=0.05, right=0.99, bottom=0.07, top=0.89, hspace=0.6)
plt.savefig('figure_5/v2/figure_5_15.svg', dpi=600)

# entropy
gs_entropy_1 = GridSpec(3, 3, figure=fig, width_ratios=[1., 1., 0.6])
fig = plt.figure(figsize=(6.8, 5.6))
ax = fig.add_subplot(gs_entropy_1[2, 2])
y, x, _ = ax.hist(entropy_values_null_model, bins=10, histtype='step', color='black')
ax.set_ylim(ymax=0.3*nb_randomize_2+5)
arr_1 = mpatches.FancyArrowPatch((entropy(data_patient.ravel(), base=None), 5.7),
                                 (entropy(data_patient.ravel(), base=None), 0.0),
                                 color='r',
                                 arrowstyle='->,head_width=.15', mutation_scale=20)
arr_1.set_clip_on(False)
ax.add_patch(arr_1)
ax.annotate('data', (0.5, 1.), xycoords=arr_1, ha='center', va='bottom', annotation_clip=False, color='red')
ax.yaxis.set_major_formatter(FuncFormatter(to_percent(n=nb_randomize_2)))
ax.set_xlim(xmax=(entropy(data_patient.ravel(), base=None)+0.1))
ax.tick_params('both', pad=1)
ax.set_ylabel('% of distribution', {"fontsize": labelfont_size}, labelpad=0)
ax.set_xlabel('entropy', {"fontsize": labelfont_size})
ax.tick_params(which='both', labelsize=tickfont_size)
plt.subplots_adjust(left=0.05, right=0.99, bottom=0.07, top=0.89, hspace=0.6)
plt.savefig('figure_5/v2/figure_5_16.svg', dpi=600)


############################### null model 2 ########################
fig = plt.figure(figsize=(6.8, 5.6))
ax = fig.add_subplot(gs_null_1[0, 1])
ax.set_title('Null model 2\nShuffle avalanche\npattern order', {"fontsize": titlefont_size}, pad=2, y=1)
ax.set_axis_off()
plt.subplots_adjust(left=0.05, right=0.99, bottom=0.07, top=0.89, hspace=0.6)
plt.savefig('figure_5/v2/figure_5_17.svg', dpi=600)

fig = plt.figure(figsize=(6.8, 5.6))
ax = fig.add_subplot(gs_shuffle[0, 2])
ax.imshow(avalanches_patterns_1.T, vmin=0.0, vmax=2.0, cmap=cmap_black_white, origin='lower')
ax.set_xticks(np.arange(0.5, avalanches_patterns_1.shape[1]))
ax.set_xticklabels([])
ax.grid(axis='x')
ax.set_yticks([])
ax.set_ylim(ymax=len(range_region_label)-0.5, ymin=-0.5)
ax.tick_params(which='both', labelsize=tickfont_size)
plt.subplots_adjust(left=0.05, right=0.99, bottom=0.07, top=0.89, hspace=0.6)
plt.savefig('figure_5/v2/figure_5_18.svg', dpi=600)
fig = plt.figure(figsize=(6.8, 5.6))
ax = fig.add_subplot(gs_shuffle[0, 3])
avalanches_patterns_copy = np.copy(avalanches_patterns_1)
indexes = np.arange(0, avalanches_patterns_1.shape[0], 1)
np.random.shuffle(indexes)
avalanches_patterns_copy = avalanches_patterns_copy[indexes]
ax.imshow(avalanches_patterns_copy.T, vmin=0.0, vmax=2.0, cmap=cmap_black_white, origin='lower')
ax.set_xticks(np.arange(0.5, avalanches_patterns_1.shape[1]))
ax.set_xticklabels([])
ax.grid(axis='x')
ax.set_yticks([])
ax.set_ylim(ymax=len(range_region_label)-0.5, ymin=-0.5)
ax.tick_params(which='both', labelsize=tickfont_size)
plt.subplots_adjust(left=0.05, right=0.99, bottom=0.07, top=0.89, hspace=0.6)
plt.savefig('figure_5/v2/figure_5_19.svg', dpi=600)

# plot label
fig = plt.figure(figsize=(6.8, 5.6))
ax = fig.add_subplot(gs_shuffle[1, 2])
# cluster_phate_1 = [0, 1, 1, 0, 1, 1, 0, ]
data = np.expand_dims(cluster_phate, 1).T
ax.imshow(data, cmap=cmap_red_white)
for (j, i), label in np.ndenumerate(data):
    plt.text(i, j+0.2, np.around(label, 2), ha='center', va='center', fontsize=tickfont_size)
ax.set_yticks([])
ax.set_xticks(np.arange(0, len(cluster_phate))+0.5)
ax.set_xticklabels([])
# ax.set_xlabel('# avalanche pattern', {"fontsize": label_size})
ax.grid(axis='x')
plt.subplots_adjust(left=0.05, right=0.99, bottom=0.07, top=0.89, hspace=0.6)
plt.savefig('figure_5/v2/figure_5_20.svg', dpi=600)
fig = plt.figure(figsize=(6.8, 5.6))
ax = fig.add_subplot(gs_shuffle[1, 3])
ax.imshow([data[0][indexes]], cmap=cmap_red_white, vmin=0, vmax=2)
for (j, i), label in np.ndenumerate([data[0][indexes]]):
    plt.text(i, j+0.2, np.around(label, 2), ha='center', va='center', fontsize=tickfont_size)
ax.set_yticks([])
ax.set_xticks(np.arange(0, len(cluster_phate))+0.5)
ax.set_xticklabels([])
ax.set_xlabel('# avalanche pattern                    ', {"fontsize": labelfont_size}, labelpad=0)
ax.grid(axis='x')
plt.subplots_adjust(left=0.05, right=0.99, bottom=0.07, top=0.89, hspace=0.6)
plt.savefig('figure_5/v2/figure_5_21.svg', dpi=600)

# transitions
gs_shuffle_2 = GridSpec(4, 6, figure=fig, height_ratios=[0.9, 0.1, 1., 1.],
                        width_ratios=[1.3, 1.3, 0.45, 1.2, 1.2, 1.2])
fig = plt.figure(figsize=(6.8, 5.6))
ax = fig.add_subplot(gs_shuffle_2[1:3, 1:3])
transitions = np.load(path + "/transition_all.npy")
im_transition = ax.imshow(transitions, vmin=0.0, vmax=0.35)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
colorbar_transition = fig.colorbar(im_transition, cax=cax)
colorbar_transition.ax.yaxis.set_tick_params(pad=0.1, labelsize=tickfont_size)
colorbar_transition.ax.set_ylabel('% of transition', {"fontsize": labelfont_size}, labelpad=2)
ax.set_xticks([0.1, 3.1, 6.1])
ax.set_xticklabels([1, 4, 7])
ax.set_yticks([0.1, 3.1, 6.1])
ax.set_yticklabels([1, 4, 7])
ax.set_title('   Observed\nCTM', {"fontsize": titlefont_size}, pad=16)
ax.set_ylabel('# cluster', {"fontsize": labelfont_size}, labelpad=0)
ax.set_xlabel('# cluster', {"fontsize": labelfont_size}, labelpad=0)
ax.tick_params(which='both', labelsize=tickfont_size)
plt.subplots_adjust(left=0.05, right=0.99, bottom=0.07, top=0.89, hspace=0.6)
plt.savefig('figure_5/v2/figure_5_22.svg', dpi=600)

fig = plt.figure(figsize=(6.8, 5.6))
nb = 3
for i in range(nb):
    gs_null_3 = GridSpec(3, 3, figure=fig, height_ratios=[i*0.5, 1., 1-i*0.5],
                         width_ratios=[0.1-i*0.1, 1., i*0.1])
    ax = fig.add_subplot(gs_null_3[1, 1])
    transitions = np.load(path + "/transition_all"+str(i)+".npy")
    im_transition = ax.imshow(transitions, vmin=0.0, vmax=0.35)
    if i == nb-1:
        ax.set_xticks([0.1, 3.1, 6.1])
        ax.set_xticklabels([1, 4, 7])
        ax.set_yticks([0.1, 3.1, 6.1])
        ax.set_yticklabels([1, 4, 7])
        ax.set_ylabel('# cluster', {"fontsize": labelfont_size}, labelpad=0)
        ax.set_xlabel('# cluster', {"fontsize": labelfont_size}, labelpad=0)
        ax.tick_params(which='both', labelsize=tickfont_size)
    elif i == 0:
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # colorbar_transition = fig.colorbar(im_transition, cax=cax)
        # colorbar_transition.ax.yaxis.set_tick_params(pad=0.1, labelsize=tickfont_size)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Null CTM    ', {"fontsize": titlefont_size}, pad=12)
    else:
        # ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
plt.subplots_adjust(left=0.05, right=0.99, bottom=0.07, top=0.89, hspace=0.6)
plt.savefig('figure_5/v2/figure_5_23.svg', dpi=600)

# # significant result
#
# load value
significatif = 0.05
nb_randomize = 100
data_null_model_all = {'transition': []}
for nb_rand in range(nb_randomize):
    data_null_model_all['transition'].append(np.load(path + "/transition_all" + str(nb_rand) + ".npy"))
data_patient_all = {'transition': np.load(path + "/transition_all.npy")}
nb_cluster_all = data_patient_all['transition'].shape[1]
pvalue_all = np.sum(np.array(data_null_model_all['transition']) > data_patient_all['transition'], axis=0) / nb_randomize
significatif_high_all = pvalue_all > 1.0 - significatif
significatif_low_all = pvalue_all < significatif
significatif_all_all = np.logical_or(significatif_low_all, significatif_high_all)
fig = plt.figure(figsize=(6.8, 5.6))
ax = fig.add_subplot(gs_shuffle_2[2:4, 1:])
im_2 = ax.imshow(pvalue_all, vmin=significatif, vmax=1 - significatif, cmap=cmap_red_blue)
ax.set_xticks([0.1, 3.1, 6.1])
ax.set_xticklabels([1, 4, 7])
ax.set_yticks([0.1, 3.1, 6.1])
ax.set_yticklabels([1, 4, 7])
ax.set_ylabel('# cluster', {"fontsize": labelfont_size}, labelpad=0)
ax.set_xlabel('# cluster    ', {"fontsize": labelfont_size})
ax.tick_params(which='both', labelsize=tickfont_size)
ax.set_title('Significant    \nCTM', {"fontsize": titlefont_size}, pad=0)
gs_shuffle_3 = GridSpec(4, 6, figure=fig, height_ratios=[0.9, 0.1, 1.25, 0.75],
                        width_ratios=[1., 1., 1.25, 0.75, 1., 1.])
plt.subplots_adjust(left=0.05, right=0.99, bottom=0.07, top=0.89, hspace=0.6)
plt.savefig('figure_5/v2/figure_5_24.svg', dpi=600)
fig = plt.figure(figsize=(6.8, 5.6))
ax = fig.add_subplot(gs_shuffle_3[3, 3])
max_nb_cluster = 15
per_diag = [pvalue['data_per_diagonal_all'] for pvalue in pvalues_prob_transisiton ]
per_diag_rand = np.array([pvalue['shuffle_per_diagonal_all'] for pvalue in pvalues_prob_transisiton ])
ax.plot(range(2, max_nb_cluster), per_diag, 'b', label='data')
ax.plot(range(2, max_nb_cluster), per_diag_rand.mean(axis=1), 'g', label='null model')
ax.fill_between(range(2, max_nb_cluster),
                per_diag_rand.mean(axis=1) + per_diag_rand.std(axis=1),
                per_diag_rand.mean(axis=1) - per_diag_rand.std(axis=1), 'g', alpha=0.5)
ax.set_ylabel('significant transition', {"fontsize": labelfont_size}, labelpad=-2)
ax.set_ylim(ymin=0.20)
ax.set_xlabel('nb clusters', {"fontsize": labelfont_size}, labelpad=0)
ax.legend(fontsize=tickfont_size, handlelength=0.5, borderpad=0.2, labelspacing=0.1, loc='lower center')
ax.tick_params(which='both', labelsize=tickfont_size, pad=0)
ax.set_title('Significant\ndiagonal', {"fontsize": titlefont_size}, pad=0)
plt.subplots_adjust(left=0.05, right=0.99, bottom=0.07, top=0.89, hspace=0.6)
plt.savefig('figure_5/v2/figure_5_25.svg', dpi=600)


fig = plt.figure(figsize=(6.8, 5.6))
plt.gca().set_axis_off()
plt.annotate('A', (0., 0.975), xycoords='figure fraction', fontsize=titlefont_size, weight='bold')
plt.annotate('B', (0., 0.545), xycoords='figure fraction', fontsize=titlefont_size, weight='bold')
plt.annotate('C', (0., 0.25), xycoords='figure fraction', fontsize=titlefont_size, weight='bold')

plt.annotate('D', (0.35, 0.975), xycoords='figure fraction', fontsize=titlefont_size, weight='bold')
plt.annotate('E', (0.35, 0.545), xycoords='figure fraction', fontsize=titlefont_size, weight='bold')
plt.annotate('F', (0.35, 0.25), xycoords='figure fraction', fontsize=titlefont_size, weight='bold')
plt.annotate('G', (0.53, 0.545), xycoords='figure fraction', fontsize=titlefont_size, weight='bold')
plt.annotate('H', (0.53, 0.25), xycoords='figure fraction', fontsize=titlefont_size, weight='bold')


plt.annotate('J', (0.69, 0.975), xycoords='figure fraction', fontsize=titlefont_size, weight='bold')
plt.annotate('K', (0.69, 0.545), xycoords='figure fraction', fontsize=titlefont_size, weight='bold')
plt.annotate('L', (0.69, 0.25), xycoords='figure fraction', fontsize=titlefont_size, weight='bold')
plt.subplots_adjust(left=0.05, right=0.99, bottom=0.07, top=0.89, hspace=0.6)
plt.savefig('figure_5/v2/figure_5_26.svg', dpi=600)


# plt.savefig('figure_5/v2/figure_5_pre_v4_1.png')
# plt.savefig('figure_5/v2/figure_5_pre_v4_1.svg')
# plt.show()

