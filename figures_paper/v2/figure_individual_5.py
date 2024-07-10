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
label_size = 12.0
tickfont_size = 10.0
range_time = (900, 1400)
regions_select = (0, 90)
path_data = os.path.dirname(os.path.realpath(__file__)) + '/../../paper/result/default/'
Avalanches_human = np.load(path_data + 'avalanches.npy', allow_pickle=True)
avalanches_patterns_all = Avalanches_human[0][904:908, :]

avalanches_patterns = np.load('../v1/avalanches_pattern_example_fig_1.npy')
phate_operator = phate.PHATE(n_components=2, n_jobs=8, n_pca=None, decay=1.2, gamma=-1, knn=1, knn_dist='cosine', mds_dist='cosine')
Y_phate_example = phate_operator.fit_transform(avalanches_patterns)
cluster_phate = KMeans(n_clusters=2, random_state=12).fit_predict(Y_phate_example)
cmap_black_white = ListedColormap(["white", "black", "white"], name='from_list', N=None)

# entropy
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

pvalues_prob_transisiton = np.load(path+'/model_diagonal.npy', allow_pickle=True)
diag_per_lows, diag_per_nos, diag_per_highs, no_diag_per_lows, no_diag_per_nos, no_diag_per_highs =\
    np.load(path+'/model_diagonalsignificatif.npy')


cmap_red_blue = get_color_map()


# figure null model 1
plt.figure()
ax = plt.gca()
data = np.expand_dims(cluster_phate, 1).T
ax.imshow(data, cmap='Pastel1')
for (j, i), label in np.ndenumerate(data):
    plt.text(i, j, np.around(label, 2), ha='center', va='center', fontsize=tickfont_size)
ax.set_yticks([])
ax.set_xticks(np.arange(0, len(cluster_phate)-1)+0.5)
ax.set_xticklabels([])
ax.set_xlabel('# avalanches pattern', {"fontsize": label_size})
ax.grid(axis='x')
plt.savefig('figure_5/null_model_1_1.svg')

plt.figure()
ax = plt.gca()
np.random.shuffle(data[0])
ax.imshow(data, cmap='Pastel1')
for (j, i), label in np.ndenumerate(data):
    plt.text(i, j, np.around(label, 2), ha='center', va='center', fontsize=tickfont_size)
ax.set_yticks([])
ax.set_xticks(np.arange(0, len(cluster_phate)-1)+0.5)
ax.set_xticklabels([])
ax.set_xlabel('# avalanches pattern', {"fontsize": label_size})
ax.grid(axis='x')
plt.savefig('figure_5/null_model_1_2.svg')
plt.close('all')

# figure null model 2
plt.figure(figsize=(5, 5))
ax = plt.gca()
ax.imshow(avalanches_patterns.T, vmin=0.0, vmax=2.0, cmap=cmap_black_white, origin='lower')
ax.set_xticks(np.arange(0.5, len(cluster_phate)))
ax.set_xticklabels([])
ax.grid(axis='x')
ax.set_yticks(range_region_label[0::4]+0.5)
ax.set_yticklabels(range_region_label[0::4])
ax.set_ylim(ymax=len(range_region_label)-1, ymin=0)
ax.set_ylabel('# region', {"fontsize": label_size})
ax.set_xlabel('# avalanches pattern', {"fontsize": label_size})
ax.tick_params(which='both', labelsize=tickfont_size)
plt.savefig('figure_5/null_model_2_1.svg')

plt.figure(figsize=(5, 5))
ax = plt.gca()
avalanches_patterns_copy = np.copy(avalanches_patterns)
np.random.shuffle(avalanches_patterns_copy)
ax.imshow(avalanches_patterns_copy.T, vmin=0.0, vmax=2.0, cmap=cmap_black_white, origin='lower')
ax.set_xticks(np.arange(0.5, len(cluster_phate)))
ax.set_xticklabels([])
ax.grid(axis='x')
ax.set_yticks(range_region_label[0::4]+0.5)
ax.set_yticklabels(range_region_label[0::4])
ax.set_ylim(ymax=len(range_region_label)-1, ymin=0)
ax.set_ylabel('# region', {"fontsize": label_size})
ax.set_xlabel('# avalanches pattern', {"fontsize": label_size})
ax.tick_params(which='both', labelsize=tickfont_size)
plt.savefig('figure_5/null_model_2_2.svg')

# v2
plt.figure(figsize=(2, 5))
ax = plt.gca()
ax.imshow(avalanches_patterns_all.T, vmin=0.0, vmax=2.0, cmap=cmap_black_white, origin='lower')
ax.set_yticks(np.arange(0, avalanches_patterns_all.shape[1], 1)[0::20])
ax.set_xticks([])
ax.set_ylabel('# region', {"fontsize": label_size})
ax.set_xlabel('# avalanches\npattern', {"fontsize": label_size})
ax.tick_params(which='both', labelsize=tickfont_size)
plt.savefig('figure_5/null_model_2_1_v2.svg')

plt.figure(figsize=(2, 5))
ax = plt.gca()
avalanches_patterns_all_copy = np.copy(avalanches_patterns_all)
np.random.shuffle(avalanches_patterns_all_copy)
ax.imshow(avalanches_patterns_all_copy.T, vmin=0.0, vmax=2.0, cmap=cmap_black_white, origin='lower')
ax.set_yticks(np.arange(0, avalanches_patterns_all.shape[1], 1)[0::20])
ax.set_xticks([])
ax.set_ylabel('# region', {"fontsize": label_size})
ax.set_xlabel('# avalanches\npattern', {"fontsize": label_size})
ax.tick_params(which='both', labelsize=tickfont_size)
plt.savefig('figure_5/null_model_2_2_v2.svg')

# transitions
fig = plt.figure(figsize=(3, 2.5))
ax = plt.gca()
transitions = np.load(path + "/transition_all.npy")
im_transition = ax.imshow(transitions, vmin=0.0, vmax=0.35)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
colorbar_transition = fig.colorbar(im_transition, cax=cax)
colorbar_transition.ax.yaxis.set_tick_params(pad=0.1, labelsize=tickfont_size)
colorbar_transition.ax.set_ylabel('% of transition', {"fontsize": label_size}, labelpad=2)
ax.set_xticks([0.1, 3.1, 6.1])
ax.set_xticklabels([1, 4, 7])
ax.set_yticks([0.1, 3.1, 6.1])
ax.set_yticklabels([1, 4, 7])
ax.set_title('CTM', {"fontsize": label_size})
ax.set_ylabel('#cluster', {"fontsize": label_size})
ax.set_xlabel('#cluster', {"fontsize": label_size})
ax.tick_params(which='both', labelsize=tickfont_size)
plt.savefig('figure_5/null_model_2_3.svg')

fig = plt.figure(figsize=(6, 5))
ax = plt.gca()
transitions = np.load(path + "/transition_all0.npy")
im_transition = ax.imshow(transitions, vmin=0.0, vmax=0.35)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
colorbar_transition = fig.colorbar(im_transition, cax=cax)
colorbar_transition.ax.yaxis.set_tick_params(pad=0.1, labelsize=tickfont_size)
colorbar_transition.ax.set_ylabel('% of transition', {"fontsize": label_size}, labelpad=2)
ax.set_xticks([0.1, 3.1, 6.1])
ax.set_xticklabels([0, 3, 6])
ax.set_yticks([0.1, 3.1, 6.1])
ax.set_yticklabels([0, 3, 6])
ax.annotate('Q', xy=(-0.3, 1.1), xycoords='axes fraction', weight='bold', fontsize=label_size)
ax.set_title('CTM shuffle', {"fontsize": label_size})
ax.set_ylabel('#cluster', {"fontsize": label_size})
ax.set_xlabel('#cluster', {"fontsize": label_size})
ax.tick_params(which='both', labelsize=tickfont_size)
plt.savefig('figure_5/null_model_2_4.svg')

# figure null model 3
plt.figure(figsize=(5, 5))
ax = plt.gca()
ax.imshow(avalanches_patterns.T, vmin=0.0, vmax=2.0, cmap=cmap_black_white, origin='lower')
ax.set_xticks(np.arange(0.5, len(cluster_phate)))
ax.set_xticklabels([])
ax.grid(axis='x')
ax.set_yticks(range_region_label[0::4]+0.5)
ax.set_yticklabels(range_region_label[0::4])
ax.set_ylim(ymax=len(range_region_label)-1, ymin=0)
ax.set_ylabel('# region', {"fontsize": label_size})
ax.set_xlabel('# avalanches pattern', {"fontsize": label_size})
ax.tick_params(which='both', labelsize=tickfont_size)
plt.savefig('figure_5/null_model_3_1.svg')

plt.figure(figsize=(5, 5))
ax = plt.gca()
avalanches_patterns_shuffle = []
for av in avalanches_patterns:
    av_copy = np.copy(av)
    np.random.shuffle(av_copy)
    avalanches_patterns_shuffle.append(av_copy)
ax.imshow(np.concatenate([avalanches_patterns_shuffle]).T, vmin=0.0, vmax=2.0, cmap=cmap_black_white, origin='lower')
ax.set_xticks(np.arange(0.5, len(cluster_phate)))
ax.set_xticklabels([])
ax.grid(axis='x')
ax.set_yticks(range_region_label[0::4]+0.5)
ax.set_yticklabels(range_region_label[0::4])
ax.set_ylim(ymax=len(range_region_label)-1, ymin=0)
ax.set_ylabel('# region', {"fontsize": label_size})
ax.set_xlabel('# avalanches pattern', {"fontsize": label_size})
ax.tick_params(which='both', labelsize=tickfont_size)
plt.savefig('figure_5/null_model_3_2.svg')
plt.close('all')

# v2
plt.figure(figsize=(2, 5))
ax = plt.gca()
ax.imshow(avalanches_patterns_all.T, vmin=0.0, vmax=2.0, cmap=cmap_black_white, origin='lower')
ax.set_yticks(np.arange(0, avalanches_patterns_all.shape[1], 1)[0::20])
ax.set_xticks([])
ax.set_ylabel('# region', {"fontsize": label_size})
ax.set_xlabel('# avalanches\npattern', {"fontsize": label_size})
ax.tick_params(which='both', labelsize=tickfont_size)
plt.savefig('figure_5/null_model_3_1_v2.svg')

plt.figure(figsize=(2, 5))
ax = plt.gca()
avalanches_patterns_all_shuffle = []
for av in avalanches_patterns_all:
    av_copy = np.copy(av)
    np.random.shuffle(av_copy)
    avalanches_patterns_all_shuffle.append(av_copy)
ax.imshow(np.concatenate([avalanches_patterns_all_shuffle]).T, vmin=0.0, vmax=2.0, cmap=cmap_black_white, origin='lower')
ax.set_yticks(np.arange(0, avalanches_patterns_all.shape[1], 1)[0::20])
ax.set_xticks([])
ax.set_ylabel('# region', {"fontsize": label_size})
ax.set_xlabel('# avalanches\npattern', {"fontsize": label_size})
ax.tick_params(which='both', labelsize=tickfont_size)
plt.savefig('figure_5/null_model_3_2_v2.svg')
plt.close('all')

# entropy
plt.figure(figsize=(5, 5))
ax = plt.gca()
y, x, _ = ax.hist(entropy_values_null_model, bins=10, histtype='step', color='black')
ax.set_ylim(ymax=0.3*nb_randomize_2+5)
arr_1 = mpatches.FancyArrowPatch((entropy(data_patient.ravel(), base=None), -2),
                                 (entropy(data_patient.ravel(), base=None), 0.0),
                                 color='r',
                                 arrowstyle='->,head_width=.15', mutation_scale=20)
arr_1.set_clip_on(False)
ax.add_patch(arr_1)
ax.annotate('data', (0.5, -1.), xycoords=arr_1, ha='center', va='bottom', annotation_clip=False, color='red')
ax.yaxis.set_major_formatter(FuncFormatter(to_percent(n=nb_randomize_2)))
ax.set_xlim(xmax=(entropy(data_patient.ravel(), base=None)+0.1))
ax.tick_params('both', pad=1)
ax.set_ylabel('percentage of distribution', {"fontsize": label_size})
ax.set_xlabel('entropy', {"fontsize": label_size})
ax.set_title('distribution of entropy (null model 1)')
plt.savefig('figure_5/null_model_1_entropy.svg')

fig = plt.figure(figsize=(8, 5))
gs = GridSpec(1, 2, figure=fig, width_ratios=[0.2, 1.0])
d = .015  # how big to make the diagonal lines in axes coordinates
ax = fig.add_subplot(gs[1])
y, x, _ = ax.hist(entropy_values_all, bins=10, histtype='step', color='black')
ax.set_ylim(ymax=0.3*nb_randomize_1+500)
ax.tick_params('both', pad=1)
ax.set_xlabel('entropy', {"fontsize": label_size})
ax.set_title('distribution of entropy (null model 3)')
ax.spines['left'].set_visible(False)
ax.set_yticks([])
# arguments to pass plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (1-d, 1+d), **kwargs)
ax.plot((-d, +d), (-d, +d), **kwargs)

ax = fig.add_subplot(gs[0])
arr_1 = mpatches.FancyArrowPatch((entropy(data_patient.ravel(), base=None), -300),
                                 (entropy(data_patient.ravel(), base=None), 0.0),
                                 color='r',
                                 arrowstyle='->,head_width=.15', mutation_scale=20)
arr_1.set_clip_on(False)
ax.add_patch(arr_1)
ax.annotate('data', (0.5, -1.), xycoords=arr_1, ha='center', va='bottom', annotation_clip=False, color='red')
y, x, _ = ax.hist(entropy_values_all, bins=10, histtype='step', color='black')
ax.set_ylim(ymax=0.3*nb_randomize_1+500, ymin=0.0)
ax.yaxis.set_major_formatter(FuncFormatter(to_percent(n=nb_randomize_1)))
ax.spines['right'].set_visible(False)
ax.set_ylabel('percentage of distribution', {"fontsize": label_size})
ax.set_xlim(xmax=(entropy(data_patient.ravel(), base=None)+0.005), xmin=(entropy(data_patient.ravel(), base=None)-0.005))
kwargs.update(transform=ax.transAxes)  # switch to the bottom axes
ax.plot((0.95-d, 1.05+d), (-d, +d), **kwargs)
ax.plot((0.95-d, 1.05+d), (1-d, 1+d), **kwargs)
plt.subplots_adjust(wspace=0.02)
plt.savefig('figure_5/null_model_1_entropy.svg')

# matrix significant
fig = plt.figure(figsize=(2, 5))
ax = plt.gca()
ax.imshow((pvalue_cluster_all[:, 2, :, :].swapaxes(0, 1)[0]-pvalue_cluster_all[:, 3, :, :].swapaxes(0, 1)[0]).T,
           vmin=-1.0, vmax=1.0, cmap=cmap_red_blue, origin='lower')
ax.tick_params('both', pad=1)
ax.set_xticks([0, 6])
ax.set_xticklabels([1, 7])
ax.set_ylabel('# region', {"fontsize": label_size})
ax.set_xlabel('# cluster', {"fontsize": label_size})
ax.set_yticks(np.arange(0, avalanches_patterns_all.shape[1], 1)[0::20])
plt.savefig('figure_5/null_model_1_significant.svg')

fig = plt.figure(figsize=(2, 5))
ax = plt.gca()
ax.imshow((pvalue_cluster_all_null_model[:, 2, :, :].swapaxes(0, 1)[0]-pvalue_cluster_all_null_model[:, 3, :, :].swapaxes(0, 1)[0]).T,
          vmin=-1.0, vmax=1.0, cmap=cmap_red_blue, origin='lower')
ax.tick_params('both', pad=1)
ax.set_xticks([0, 6])
ax.set_xticklabels([1, 7])
ax.set_ylabel('# region', {"fontsize": label_size})
ax.set_xlabel('# cluster', {"fontsize": label_size})
ax.set_yticks(np.arange(0, avalanches_patterns_all.shape[1], 1)[0::20])
plt.savefig('figure_5/null_model_3_significant.svg')

fig = plt.figure(figsize=(2, 1))
ax = plt.gca()
ax.imshow([[-1.0], [0.0], [1.0]], vmin=-1.0, vmax=1.0, cmap=cmap_red_blue, origin='lower')
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(['significant active', 'no significant', 'significant inactive'])
ax.set_xticks([])
ax.set_xlim(xmin=0, xmax=0.5)
plt.subplots_adjust(right=1., left=0.60)
plt.savefig('figure_5/null_model_legend_significant.svg')
plt.close('all')

# load value
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
fig = plt.figure(figsize=(5, 5))
ax = plt.gca()
im_2 = ax.imshow(pvalue_all, vmin=significatif, vmax=1 - significatif, cmap=cmap_red_blue)
ax.set_xticks([0.1, 3.1, 6.1])
ax.set_xticklabels([1, 4, 7])
ax.set_yticks([0.1, 3.1, 6.1])
ax.set_yticklabels([1, 4, 7])
ax.set_ylabel('# cluster', {"fontsize": label_size})
ax.set_xlabel('# cluster', {"fontsize": label_size})
ax.set_title('CTM significant', {"fontsize": label_size})
plt.savefig('figure_5/null_model_transition_significant.svg')

fig = plt.figure(figsize=(5, 5))
ax = plt.gca()
max_nb_cluster = 15
per_diag = [pvalue['data_per_diagonal_all'] for pvalue in pvalues_prob_transisiton ]
per_diag_rand = np.array([pvalue['shuffle_per_diagonal_all'] for pvalue in pvalues_prob_transisiton ])
ax.plot(range(2, max_nb_cluster), per_diag, 'b', label='data')
ax.plot(range(2, max_nb_cluster), per_diag_rand.mean(axis=1), 'g', label='null model')
ax.fill_between(range(2, max_nb_cluster),
                 per_diag_rand.mean(axis=1) + per_diag_rand.std(axis=1),
                 per_diag_rand.mean(axis=1) - per_diag_rand.std(axis=1), 'g', alpha=0.5)
ax.set_ylabel('significant of transition', {"fontsize": label_size})
ax.set_xlabel('number of clusters', {"fontsize": label_size})
plt.savefig('figure_5/null_model_transition_diagnal_significant.svg')

plt.show()
