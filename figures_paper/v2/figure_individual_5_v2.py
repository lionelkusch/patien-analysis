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


fig = plt.figure(figsize=(6.8, 10.5))
gs_null_1 = GridSpec(6, 3, figure=fig)

# titles
ax = fig.add_subplot(gs_null_1[:2, 0])
ax.set_title('Shuffle', {"fontsize": label_size})
ax.set_ylabel('Null model 1', {"fontsize": label_size}, labelpad=30)
ax.set_yticks([])
ax.get_yaxis().set_visible(True)
ax.get_xaxis().set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax = fig.add_subplot(gs_null_1[2:4, 0])
ax.set_ylabel('Null model 2', {"fontsize": label_size}, labelpad=30)
ax.set_yticks([])
ax.get_yaxis().set_visible(True)
ax.get_xaxis().set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax = fig.add_subplot(gs_null_1[4:6, 0])
ax.set_ylabel('Null model 3', {"fontsize": label_size}, labelpad=30)
ax.set_yticks([])
ax.get_yaxis().set_visible(True)
ax.get_xaxis().set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax = fig.add_subplot(gs_null_1[:2, 2])
ax.set_title('Significant result', {"fontsize": label_size}, pad=70)
ax.set_yticks([])
ax.get_yaxis().set_visible(True)
ax.get_xaxis().set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# figure null model 1
ax = fig.add_subplot(gs_null_1[0, 0])
data = np.expand_dims(cluster_phate, 1).T
ax.imshow(data, cmap='Pastel1')
for (j, i), label in np.ndenumerate(data):
    plt.text(i, j, np.around(label, 2), ha='center', va='center', fontsize=tickfont_size)
ax.set_yticks([])
ax.set_xticks(np.arange(0, len(cluster_phate)-1)+0.5)
ax.set_xticklabels([])
ax.set_xlabel('# avalanche pattern', {"fontsize": label_size})
ax.grid(axis='x')

# plot shuffles
ax = fig.add_subplot(gs_null_1[1, 0])
np.random.shuffle(data[0])
ax.imshow(data, cmap='Pastel1')
for (j, i), label in np.ndenumerate(data):
    plt.text(i, j, np.around(label, 2), ha='center', va='center', fontsize=tickfont_size)
ax.set_yticks([])
ax.set_xticks(np.arange(0, len(cluster_phate)-1)+0.5)
ax.set_xticklabels([])
ax.set_xlabel('# avalanche pattern', {"fontsize": label_size})
ax.grid(axis='x')

# entropy
ax = fig.add_subplot(gs_null_1[:2, 1])
y, x, _ = ax.hist(entropy_values_null_model, bins=10, histtype='step', color='black')
ax.set_ylim(ymax=0.3*nb_randomize_2+5)
arr_1 = mpatches.FancyArrowPatch((entropy(data_patient.ravel(), base=None), -4.5),
                                 (entropy(data_patient.ravel(), base=None), 0.0),
                                 color='r',
                                 arrowstyle='->,head_width=.15', mutation_scale=20)
arr_1.set_clip_on(False)
ax.add_patch(arr_1)
ax.annotate('data', (0.5, -0.6), xycoords=arr_1, ha='center', va='bottom', annotation_clip=False, color='red')
ax.yaxis.set_major_formatter(FuncFormatter(to_percent(n=nb_randomize_2)))
ax.set_xlim(xmax=(entropy(data_patient.ravel(), base=None)+0.1))
ax.tick_params('both', pad=1)
ax.set_ylabel('% of distribution', {"fontsize": label_size}, labelpad=0)
ax.set_xlabel('entropy', {"fontsize": label_size})
ax.set_title('distribution of entropy', {"fontsize": label_size})

# matrix significant
gs_null_1_sign = GridSpec(3, 5, figure=fig, width_ratios=[1., 1., 1., 1., 0.9])
ax = fig.add_subplot(gs_null_1_sign[0, 3])
ax.imshow((pvalue_cluster_all[:, 2, :, :].swapaxes(0, 1)[0]-pvalue_cluster_all[:, 3, :, :].swapaxes(0, 1)[0]).T,
           vmin=-1.0, vmax=1.0, cmap=cmap_red_blue, origin='lower')
ax.tick_params('both', pad=1)
ax.set_xticks([0, 6])
ax.set_xticklabels([1, 7])
ax.set_xlabel('# cluster  ', {"fontsize": label_size})
ax.set_ylabel('# region', {"fontsize": label_size})
ax.set_yticks(np.arange(0, avalanches_patterns_all.shape[1], 1)[0::20])


# ######################   v1     ######################################
# # figure null model 3
# ax = fig.add_subplot(gs_null_1[4, 0])
# ax.imshow(avalanches_patterns.T, vmin=0.0, vmax=2.0, cmap=cmap_black_white, origin='lower')
# ax.set_xticks(np.arange(0.5, len(cluster_phate)))
# ax.set_xticklabels([])
# ax.grid(axis='x')
# ax.set_yticks(range_region_label[0::4]+0.5)
# ax.set_yticklabels(range_region_label[0::4])
# ax.set_ylim(ymax=len(range_region_label)-1, ymin=0)
# ax.set_ylabel('# region', {"fontsize": label_size})
# # ax.set_xlabel('# avalanches pattern', {"fontsize": label_size})
# ax.tick_params(which='both', labelsize=tickfont_size)
# ax = fig.add_subplot(gs_null_1[5, 0])
# avalanches_patterns_shuffle = []
# for av in avalanches_patterns:
#     av_copy = np.copy(av)
#     np.random.shuffle(av_copy)
#     avalanches_patterns_shuffle.append(av_copy)
# ax.imshow(np.concatenate([avalanches_patterns_shuffle]).T, vmin=0.0, vmax=2.0, cmap=cmap_black_white, origin='lower')
# ax.set_xticks(np.arange(0.5, len(cluster_phate)))
# ax.set_xticklabels([])
# ax.grid(axis='x')
# ax.set_yticks(range_region_label[0::4]+0.5)
# ax.set_yticklabels(range_region_label[0::4])
# ax.set_ylim(ymax=len(range_region_label)-1, ymin=0)
# ax.set_ylabel('# region', {"fontsize": label_size})
# ax.set_xlabel('# avalanches pattern', {"fontsize": label_size})
# ax.tick_params(which='both', labelsize=tickfont_size)
######################   v2     ######################################
gs_null_3 = GridSpec(3, 6, figure=fig, width_ratios=[0.8, 0.8, 1., 1., 1., 1.])
ax = fig.add_subplot(gs_null_3[2, 0])
ax.imshow(avalanches_patterns_all.T, vmin=0.0, vmax=2.0, cmap=cmap_black_white, origin='lower')
ax.set_yticks(np.arange(0, avalanches_patterns_all.shape[1], 1)[0::20])
ax.set_ylabel('# region', {"fontsize": label_size})
ax.set_xticks([])
ax.set_xlabel('           # avalanches pattern', {"fontsize": label_size})
ax.tick_params(which='both', labelsize=tickfont_size)
ax = fig.add_subplot(gs_null_3[2, 1])
avalanches_patterns_all_shuffle = []
for av in avalanches_patterns_all:
    av_copy = np.copy(av)
    np.random.shuffle(av_copy)
    avalanches_patterns_all_shuffle.append(av_copy)
ax.imshow(np.concatenate([avalanches_patterns_all_shuffle]).T, vmin=0.0, vmax=2.0, cmap=cmap_black_white, origin='lower')
ax.set_yticks(np.arange(0, avalanches_patterns_all.shape[1], 1)[0::20])
ax.set_ylabel('# region', {"fontsize": label_size})
ax.set_xticks([])
# ax.set_ylabel('# avalanches\npattern', {"fontsize": label_size})
ax.tick_params(which='both', labelsize=tickfont_size)

ax = fig.add_subplot(gs_null_1_sign[2, 3])
ax.imshow((pvalue_cluster_all_null_model[:, 2, :, :].swapaxes(0, 1)[0]-pvalue_cluster_all_null_model[:, 3, :, :].swapaxes(0, 1)[0]).T,
          vmin=-1.0, vmax=1.0, cmap=cmap_red_blue, origin='lower')
ax.tick_params('both', pad=1)
ax.set_xticks([0, 6])
ax.set_xticklabels([1, 7])
ax.set_xlabel('# cluster', {"fontsize": label_size})
ax.set_yticks(np.arange(0, avalanches_patterns_all.shape[1], 1)[0::20])
ax.set_ylabel('# region', {"fontsize": label_size})

gs = GridSpec(3, 6, figure=fig, width_ratios=[0.5, 0.5, 0.15, 0.85, 0.5, 0.5], wspace=0.0)
d = .015  # how big to make the diagonal lines in axes coordinates
ax = fig.add_subplot(gs[2, 3])
y, x, _ = ax.hist(entropy_values_all, bins=10, histtype='step', color='black')
ax.set_ylim(ymax=0.3*nb_randomize_1+500)
ax.tick_params('both', pad=1)
ax.set_xlabel('entropy', {"fontsize": label_size})
ax.spines['left'].set_visible(False)
ax.set_yticks([])
# arguments to pass plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (1-d, 1+d), **kwargs)
ax.plot((-d, +d), (-d, +d), **kwargs)

ax = fig.add_subplot(gs[2, 2])
arr_1 = mpatches.FancyArrowPatch((entropy(data_patient.ravel(), base=None), -700),
                                 (entropy(data_patient.ravel(), base=None), 0.0),
                                 color='r',
                                 arrowstyle='->,head_width=.15', mutation_scale=20)
arr_1.set_clip_on(False)
ax.add_patch(arr_1)
ax.annotate('data', (0.5, -0.5), xycoords=arr_1, ha='center', va='bottom', annotation_clip=False, color='red')
y, x, _ = ax.hist(entropy_values_all, bins=10, histtype='step', color='black')
ax.set_ylim(ymax=0.3*nb_randomize_1+500, ymin=0.0)
ax.yaxis.set_major_formatter(FuncFormatter(to_percent(n=nb_randomize_1)))
ax.spines['right'].set_visible(False)
ax.set_ylabel('% of distribution', {"fontsize": label_size}, labelpad=0)
ax.set_xlim(xmax=(entropy(data_patient.ravel(), base=None)+0.005), xmin=(entropy(data_patient.ravel(), base=None)-0.005))
ax.set_xticks([np.around(entropy(data_patient.ravel(), base=None), decimals=2)])
kwargs.update(transform=ax.transAxes)  # switch to the bottom axes
ax.plot((0.95-d, 1.05+d), (-d, +d), **kwargs)
ax.plot((0.95-d, 1.05+d), (1-d, 1+d), **kwargs)
plt.subplots_adjust(wspace=0.02)


gs_null_2 = GridSpec(6, 7, figure=fig, wspace=0.7, width_ratios=[1., 1., 0.5, 1., 1., 1., 1.])

############################### null model 3 ########################
ax = fig.add_subplot(gs_null_2[2:4, 0])
ax.imshow(avalanches_patterns.T, vmin=0.0, vmax=2.0, cmap=cmap_black_white, origin='lower')
ax.set_xticks(np.arange(0.5, len(cluster_phate)))
ax.set_xticklabels([])
ax.grid(axis='x')
ax.set_yticks(range_region_label[0::4]+0.5)
ax.set_yticklabels(range_region_label[0::4])
ax.set_ylim(ymax=len(range_region_label)-1, ymin=0)
ax.set_ylabel('# region', {"fontsize": label_size})
ax.set_xlabel('# avalanche \npattern', {"fontsize": label_size})
ax.tick_params(which='both', labelsize=tickfont_size)
ax = fig.add_subplot(gs_null_2[2:4, 3])
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
ax.set_xlabel('# avalanche\npattern', {"fontsize": label_size})
ax.tick_params(which='both', labelsize=tickfont_size)
# ######################    v2     #########################""
# ax = fig.add_subplot(gs_null_1[2, 0])
# ax.imshow(avalanches_patterns_all, vmin=0.0, vmax=2.0, cmap=cmap_black_white, origin='lower')
# ax.set_xticks(np.arange(0, avalanches_patterns_all.shape[1], 1)[0::20])
# ax.set_xlabel('# region', {"fontsize": label_size})
# ax.set_yticks([])
# ax.set_ylabel('# avalanches pattern                 ', {"fontsize": label_size})
# ax.tick_params(which='both', labelsize=tickfont_size)
# ax = fig.add_subplot(gs_null_1[3, 0])
# avalanches_patterns_all_copy = np.copy(avalanches_patterns_all)
# np.random.shuffle(avalanches_patterns_all_copy)
# ax.imshow(avalanches_patterns_all_copy, vmin=0.0, vmax=2.0, cmap=cmap_black_white, origin='lower')
# ax.set_xticks(np.arange(0, avalanches_patterns_all.shape[1], 1)[0::20])
# ax.set_xlabel('# region', {"fontsize": label_size})
# ax.set_yticks([])
# # ax.set_ylabel('# avalanches\npattern', {"fontsize": label_size})
# ax.tick_params(which='both', labelsize=tickfont_size)

# transitions
ax = fig.add_subplot(gs_null_2[2:4, 1])
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
ax.set_ylabel('#cluster', {"fontsize": label_size}, labelpad=0)
ax.set_xlabel('#cluster', {"fontsize": label_size}, labelpad=0)
ax.tick_params(which='both', labelsize=tickfont_size)

for i in range(10):
    gs_null_3 = GridSpec(3, 7, figure=fig, height_ratios=[0.8 + i*0.05, 1., 1.],
                         width_ratios=[1., 1., 1., 0.8 + (10-i)*0.1, 1., 1., 1.])
    ax = fig.add_subplot(gs_null_3[1, 4])
    transitions = np.load(path + "/transition_all"+str(i)+".npy")
    im_transition = ax.imshow(transitions, vmin=0.0, vmax=0.35)
    if i == 9:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        colorbar_transition = fig.colorbar(im_transition, cax=cax)
        colorbar_transition.ax.yaxis.set_tick_params(pad=0.1, labelsize=tickfont_size)
        ax.set_xticks([0.1, 3.1, 6.1])
        ax.set_xticklabels([1, 4, 7])
        ax.set_yticks([0.1, 3.1, 6.1])
        ax.set_yticklabels([1, 4, 7])
        ax.set_title('CTM shuffle', {"fontsize": label_size}, pad=50)
        ax.set_ylabel('   #cluster', {"fontsize": label_size}, labelpad=0)
        ax.set_xlabel('#cluster', {"fontsize": label_size}, labelpad=0)
        ax.tick_params(which='both', labelsize=tickfont_size)
    else:
        ax.set_axis_off()

gs_null_2_signif = GridSpec(7, 4, figure=fig, wspace=0.7, width_ratios=[1., 1., 0.5, 1.],
                            height_ratios=[1., 1., 1., 0.3, 1., 1., 1.], right=0.99)
# significant result

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
ax = fig.add_subplot(gs_null_2_signif[2, 3])
im_2 = ax.imshow(pvalue_all, vmin=significatif, vmax=1 - significatif, cmap=cmap_red_blue)
ax.set_xticks([0.1, 3.1, 6.1])
ax.set_xticklabels([1, 4, 7])
ax.set_yticks([0.1, 3.1, 6.1])
ax.set_yticklabels([1, 4, 7])
ax.set_ylabel('# cluster', {"fontsize": label_size})
ax.set_xlabel('# cluster', {"fontsize": label_size})
ax.set_title('CTM significant  ', {"fontsize": label_size})
ax = fig.add_subplot(gs_null_2_signif[4, 3])
max_nb_cluster = 15
per_diag = [pvalue['data_per_diagonal_all'] for pvalue in pvalues_prob_transisiton ]
per_diag_rand = np.array([pvalue['shuffle_per_diagonal_all'] for pvalue in pvalues_prob_transisiton ])
ax.plot(range(2, max_nb_cluster), per_diag, 'b', label='data')
ax.plot(range(2, max_nb_cluster), per_diag_rand.mean(axis=1), 'g', label='null model')
ax.fill_between(range(2, max_nb_cluster),
                per_diag_rand.mean(axis=1) + per_diag_rand.std(axis=1),
                per_diag_rand.mean(axis=1) - per_diag_rand.std(axis=1), 'g', alpha=0.5)
ax.set_ylabel('significant transition', {"fontsize": label_size}, labelpad=-3)
ax.set_xlabel('nb clusters', {"fontsize": label_size}, labelpad=0)
ax.legend(bbox_to_anchor=(1., -0.4))


########################### legend significant
gs_null_legend = GridSpec(3, 4, figure=fig, height_ratios=[0.5, 1., 1.],
                          width_ratios=[1., 1., 1., 0.1], right=0.99, top=0.97)
ax = fig.add_subplot(gs_null_legend[0, 3])
ax.imshow([[-1.0], [0.0], [1.0]], vmin=-1.0, vmax=1.0, cmap=cmap_red_blue, origin='lower')
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(['significant active', 'no significant', 'significant inactive'])
ax.set_xticks([])
ax.set_xlim(xmin=0, xmax=0.5)

plt.subplots_adjust(left=0.095, right=0.97, bottom=0.07, top=0.95, wspace=0.7)

plt.savefig('figure/figure_5_pre_v2.png')
plt.savefig('figure/figure_5_pre_v2.svg')
plt.show()

