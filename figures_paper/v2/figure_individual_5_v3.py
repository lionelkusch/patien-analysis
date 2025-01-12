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
avalanches_patterns_all = Avalanches_human[0][904:910, :]

avalanches_patterns = Avalanches_human[0][105:125, :]
phate_operator = phate.PHATE(n_components=2, n_jobs=8, n_pca=None, decay=1.2, gamma=-1, knn=1, knn_dist='cosine', mds_dist='cosine')
Y_phate_example = phate_operator.fit_transform(avalanches_patterns)
cluster_phate = KMeans(n_clusters=2, random_state=12).fit_predict(Y_phate_example)
cmap_black_white = ListedColormap(["white", "black", "white"], name='from_list', N=None)

# entropy
path = os.path.dirname(os.path.realpath(__file__)) + "/../../paper/result/default/"
nb_randomize_1 = 10000
significatif = 0.05
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
cmap_red_blue = get_color_map()


fig = plt.figure(figsize=(6.8, 7.6))
gs_null_1 = GridSpec(6, 3, figure=fig)

# titles
ax = fig.add_subplot(gs_null_1[:2, 0])
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

################################### null model 1 ###########################
# figure null model 1
gs_shuffle = GridSpec(10, 2, figure=fig, width_ratios=[1.0, 0.1],
                      height_ratios=[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
ax = fig.add_subplot(gs_shuffle[0, 0])
cluster_phate_1 = np.random.randint(0, 2, 25)
data = np.expand_dims(cluster_phate_1, 1).T
ax.imshow(data, cmap='Pastel1')
for (j, i), label in np.ndenumerate(data):
    plt.text(i, j, np.around(label, 2), ha='center', va='center', fontsize=tickfont_size)
ax.set_yticks([])
ax.set_xticks(np.arange(0, len(cluster_phate_1)-1)+0.5)
ax.set_xticklabels([])
# ax.set_xlabel('# avalanche pattern', {"fontsize": label_size})
ax.grid(axis='x')
ax.annotate('A', xy=(-0.05, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)

# plot shuffles
ax = fig.add_subplot(gs_shuffle[1, 0])
np.random.shuffle(data[0])
ax.imshow(data, cmap='Pastel1')
for (j, i), label in np.ndenumerate(data):
    plt.text(i, j, np.around(label, 2), ha='center', va='center', fontsize=tickfont_size)
ax.set_yticks([])
ax.set_xticks(np.arange(0, len(cluster_phate_1)-1)+0.5)
ax.set_xticklabels([])
ax.set_xlabel('# avalanche pattern', {"fontsize": label_size}, labelpad=0)
ax.grid(axis='x')
ax.annotate('B', xy=(-0.05, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)

# entropy
gs_entropy = GridSpec(5, 6, figure=fig, height_ratios=[1., 0.5, 0.5, 0.25, .75], width_ratios=[1., 1., 1., 1., 0.15, 0.85], wspace=0.0)
d = .015  # how big to make the diagonal lines in axes coordinates
ax = fig.add_subplot(gs_entropy[0, 5])
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
ax = fig.add_subplot(gs_entropy[0, 4])
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
ax.set_ylabel('% of distribution', {"fontsize": label_size}, labelpad=0)
ax.set_xlim(xmax=(entropy(data_patient.ravel(), base=None)+0.005), xmin=(entropy(data_patient.ravel(), base=None)-0.005))
ax.set_xticks([np.around(entropy(data_patient.ravel(), base=None), decimals=2)])
ax.tick_params('x', pad=10)
kwargs.update(transform=ax.transAxes)  # switch to the bottom axes
ax.plot((0.95-d, 1.05+d), (-d, +d), **kwargs)
ax.plot((0.95-d, 1.05+d), (1-d, 1+d), **kwargs)
plt.subplots_adjust(wspace=0.02)
ax.annotate('K', xy=(-2., 0.95), xycoords='axes fraction', weight='bold', fontsize=label_size)

# matrix significant
ax = fig.add_subplot(gs_shuffle[2, 0])
ax.imshow((pvalue_cluster_all[:, 2, :, :].swapaxes(0, 1)[0]-pvalue_cluster_all[:, 3, :, :].swapaxes(0, 1)[0]),
           vmin=-1.0, vmax=1.0, cmap=cmap_red_blue, origin='lower')
ax.tick_params('both', pad=1)
ax.set_yticks([0, 6])
ax.set_yticklabels([1, 7])
ax.set_ylabel('# cluster  ', {"fontsize": label_size})
ax.set_xlabel('# region', {"fontsize": label_size}, labelpad=0)
ax.set_xticks(np.arange(0, avalanches_patterns_all.shape[1], 1)[0::20])
ax.annotate('C', xy=(-0.05, 1.2), xycoords='axes fraction', weight='bold', fontsize=label_size)


################################### null model 3 ###########################
# example 1
ax = fig.add_subplot(gs_shuffle[7, 0])
ax.imshow(avalanches_patterns_all, vmin=0.0, vmax=2.0, cmap=cmap_black_white, origin='lower')
ax.set_xticks(np.arange(0, avalanches_patterns_all.shape[1], 1)[0::20])
# ax.set_xlabel('# region', {"fontsize": label_size})
ax.set_yticks([])
ax.set_ylabel('# avalanches               \npattern               ', {"fontsize": label_size})
ax.tick_params(which='both', labelsize=tickfont_size)
ax.annotate('H', xy=(-0.05, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
# plot shuffles
ax = fig.add_subplot(gs_shuffle[8, 0])
avalanches_patterns_all_shuffle = []
for av in avalanches_patterns_all:
    av_copy = np.copy(av)
    np.random.shuffle(av_copy)
    avalanches_patterns_all_shuffle.append(av_copy)
ax.imshow(np.concatenate([avalanches_patterns_all_shuffle]), vmin=0.0, vmax=2.0, cmap=cmap_black_white, origin='lower')
ax.set_xticks(np.arange(0, avalanches_patterns_all.shape[1], 1)[0::20])
ax.set_xlabel('# region', {"fontsize": label_size}, labelpad=-2)
ax.set_yticks([])
ax.tick_params(which='both', labelsize=tickfont_size)
ax.annotate('I', xy=(-0.05, 0.0), xycoords='axes fraction', weight='bold', fontsize=label_size)
# significant
ax = fig.add_subplot(gs_shuffle[9, 0])
ax.imshow((pvalue_cluster_all_null_model[:, 2, :, :].swapaxes(0, 1)[0]-pvalue_cluster_all_null_model[:, 3, :, :].swapaxes(0, 1)[0]),
          vmin=-1.0, vmax=1.0, cmap=cmap_red_blue, origin='lower')
ax.tick_params('both', pad=1)
ax.set_yticks([0, 6])
ax.set_yticklabels([1, 7])
ax.set_ylabel('# cluster', {"fontsize": label_size})
ax.set_xticks(np.arange(0, avalanches_patterns_all.shape[1], 1)[0::20])
ax.set_xlabel('# region', {"fontsize": label_size}, labelpad=0)
ax.annotate('J', xy=(-0.05, 1.2), xycoords='axes fraction', weight='bold', fontsize=label_size)
# entropy
ax = fig.add_subplot(gs_entropy[4, 4:])
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
ax.set_ylabel('% of distribution', {"fontsize": label_size}, labelpad=0)
ax.set_xlabel('entropy', {"fontsize": label_size})
ax.annotate('N', xy=(-0.3, 0.95), xycoords='axes fraction', weight='bold', fontsize=label_size)


############################### null model 2 ########################
gs_significant = GridSpec(4, 3, figure=fig, height_ratios=[1., 0.5, 0.5, 1.0])
gs_null_model_2 = GridSpec(6, 3, figure=fig)
ax = fig.add_subplot(gs_null_model_2[2, 0:1])
ax.imshow(avalanches_patterns.T, vmin=0.0, vmax=2.0, cmap=cmap_black_white, origin='lower')
ax.set_xticks(np.arange(0.5, len(cluster_phate)))
ax.set_xticklabels([])
ax.grid(axis='x')
ax.set_yticks(range_region_label[0::4]+0.5)
ax.set_yticklabels(range_region_label[0::4])
ax.set_ylim(ymax=len(range_region_label)-1, ymin=0)
ax.set_ylabel('# region', {"fontsize": label_size})
# ax.set_xlabel('# avalanche pattern', {"fontsize": label_size})
ax.tick_params(which='both', labelsize=tickfont_size)
ax.annotate('D', xy=(-0.1, 1.1), xycoords='axes fraction', weight='bold', fontsize=label_size)
ax = fig.add_subplot(gs_null_model_2[3, 0:1])
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
ax.set_xlabel('# avalanche\npattern', {"fontsize": label_size}, labelpad=0)
ax.tick_params(which='both', labelsize=tickfont_size)
ax.annotate('E', xy=(-0.1, 1.1), xycoords='axes fraction', weight='bold', fontsize=label_size)

# transitions
ax = fig.add_subplot(gs_null_model_2[2, 1])
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
ax.set_title('CTM', {"fontsize": label_size}, pad=0)
ax.set_ylabel('# cluster', {"fontsize": label_size}, labelpad=0)
ax.set_xlabel('# cluster', {"fontsize": label_size}, labelpad=0)
ax.tick_params(which='both', labelsize=tickfont_size)
ax.annotate('F', xy=(-0.4, 1.), xycoords='axes fraction', weight='bold', fontsize=label_size)

nb = 5
for i in range(nb):
    gs_null_3 = GridSpec(7, 5, figure=fig, height_ratios=[i*0.1, 5.7, 1., 1., 1., 5., 1-i*0.1],
                         width_ratios=[1.-i*0.1, 0.1, 1., 0.7, i*0.1])
    ax = fig.add_subplot(gs_null_3[4, 2])
    transitions = np.load(path + "/transition_all"+str(i)+".npy")
    im_transition = ax.imshow(transitions, vmin=0.0, vmax=0.35)
    if i == nb-1:

        ax.set_xticks([0.1, 3.1, 6.1])
        ax.set_xticklabels([1, 4, 7])
        ax.set_yticks([0.1, 3.1, 6.1])
        ax.set_yticklabels([1, 4, 7])
        ax.set_title('        CTM shuffle', {"fontsize": label_size}, pad=13)
        ax.set_ylabel('   # cluster', {"fontsize": label_size}, labelpad=0)
        ax.set_xlabel('# cluster', {"fontsize": label_size}, labelpad=0)
        ax.tick_params(which='both', labelsize=tickfont_size)
    elif i == 0:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        colorbar_transition = fig.colorbar(im_transition, cax=cax)
        colorbar_transition.ax.yaxis.set_tick_params(pad=0.1, labelsize=tickfont_size)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.annotate('G', xy=(-2.5, 1.5), xycoords='axes fraction', weight='bold', fontsize=label_size)
    else:
        # ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])

# significant result

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
ax = fig.add_subplot(gs_significant[1, 2])
im_2 = ax.imshow(pvalue_all, vmin=significatif, vmax=1 - significatif, cmap=cmap_red_blue)
ax.set_xticks([0.1, 3.1, 6.1])
ax.set_xticklabels([1, 4, 7])
ax.set_yticks([0.1, 3.1, 6.1])
ax.set_yticklabels([1, 4, 7])
ax.set_ylabel('# cluster', {"fontsize": label_size})
ax.set_xlabel('# cluster', {"fontsize": label_size})
# ax.set_title('CTM significant  ', {"fontsize": label_size})
ax.annotate('L', xy=(-0.6, 1.1), xycoords='axes fraction', weight='bold', fontsize=label_size)
ax = fig.add_subplot(gs_significant[2, 2])
max_nb_cluster = 15
per_diag = [pvalue['data_per_diagonal_all'] for pvalue in pvalues_prob_transisiton ]
per_diag_rand = np.array([pvalue['shuffle_per_diagonal_all'] for pvalue in pvalues_prob_transisiton ])
ax.plot(range(2, max_nb_cluster), per_diag, 'b', label='data')
ax.plot(range(2, max_nb_cluster), per_diag_rand.mean(axis=1), 'g', label='null model')
ax.fill_between(range(2, max_nb_cluster),
                per_diag_rand.mean(axis=1) + per_diag_rand.std(axis=1),
                per_diag_rand.mean(axis=1) - per_diag_rand.std(axis=1), 'g', alpha=0.5)
ax.set_ylabel('significant transition', {"fontsize": label_size}, labelpad=0)
ax.set_xlabel('nb clusters', {"fontsize": label_size}, labelpad=0)
ax.legend(bbox_to_anchor=(1., -0.45))
ax.annotate('M', xy=(-0.2, 1.1), xycoords='axes fraction', weight='bold', fontsize=label_size)


########################### legend significant ##################################
gs_null_legend = GridSpec(2, 5, figure=fig, height_ratios=[1., 0.03],
                          width_ratios=[0.2, 1., 1., 1.4, 1.], bottom=0.005)
ax = fig.add_subplot(gs_null_legend[1, 1])
ax.imshow([[-1.0]], vmin=-1.0, vmax=1.0, cmap=cmap_red_blue, origin='lower')
ax.set_yticks([0])
ax.set_yticklabels(['significant active'])
ax.set_xticks([])
ax.set_xlim(xmin=0, xmax=0.5)
ax = fig.add_subplot(gs_null_legend[1, 2])
ax.imshow([[0.0]], vmin=-1.0, vmax=1.0, cmap=cmap_red_blue, origin='lower')
ax.set_yticks([0])
ax.set_yticklabels(['no significant'])
ax.set_xticks([])
ax.set_xlim(xmin=0, xmax=0.5)
ax = fig.add_subplot(gs_null_legend[1, 3])
ax.imshow([[1.0]], vmin=-1.0, vmax=1.0, cmap=cmap_red_blue, origin='lower')
ax.set_yticks([0])
ax.set_yticklabels(['significant inactive'])
ax.set_xticks([])
ax.set_xlim(xmin=0, xmax=0.5)

plt.subplots_adjust(left=0.095, right=0.97, bottom=0.07, top=0.98, wspace=0.7, hspace=0.42)

plt.savefig('figure/figure_5_pre_v3.png')
plt.savefig('figure/figure_5_pre_v3.svg')
plt.show()

