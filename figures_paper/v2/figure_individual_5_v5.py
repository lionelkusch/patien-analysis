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

plt.rcParams['svg.fonttype'] = 'none'


def to_percent(n=10):
    def function(y, position):
        # Ignore the passed in position. This has the effect of scaling the default
        # tick locations.
        s = str(np.array(100 * y / n, dtype=int))
        # The percent symbol needs escaping in latex
        if matplotlib.rcParams['text.usetex'] is True:
            return s
        else:
            return s
    return function


np.random.seed(42)
titlefont_size = 10.0
tickfont_size = 8.0
labelfont_size = 9.0
letter_font_size = 12
range_time = (900, 1400)
regions_select = (0, 90)
path_data = os.path.dirname(os.path.realpath(__file__)) + '/../../paper/result/default/'
Avalanches_human = np.load(path_data + 'avalanches.npy', allow_pickle=True)
avalanches_patterns_all = Avalanches_human[0][904:910, :]

cmap_black_white = ListedColormap(["white", "black", "white"], name='from_list', N=None)
cmap_blue_white = ListedColormap(["white", "deepskyblue", "deepskyblue"], name='from_list', N=None)
cmap_red_white = ListedColormap(["white", "salmon", "salmon"], name='from_list', N=None)
cmap_red_magenta = ListedColormap(["white", "salmon", "magenta", "magenta"], name='from_list', N=3)
cmap_bleu_green = ListedColormap(["blueviolet", "lightblue", "cyan", "cyan"], name='from_list', N=3)
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
ax.set_title("Null model 1\n$\\bf{shuffle}$\n$\\bf{cluster\ label}$", {"fontsize": titlefont_size}, pad=2)
ax.set_axis_off()

gs_shuffle = GridSpec(4, 6, figure=fig, height_ratios=[0.9, 0.1, 1., 1.], hspace=0.05, wspace=0.25)
avalanches_patterns_1 = np.load('avalanches_example.npy')
range_region_label = np.arange(0, avalanches_patterns_1.shape[0], 1)
phate_operator = phate.PHATE(n_components=2, n_jobs=8, n_pca=None, decay=1.2, gamma=-1, knn=1, knn_dist='cosine', mds_dist='cosine')
Y_phate_example = phate_operator.fit_transform(avalanches_patterns_1)
cluster_phate = KMeans(n_clusters=3, random_state=12).fit_predict(Y_phate_example)
# avalanches
ax = fig.add_subplot(gs_shuffle[0, :2])
ax.imshow(avalanches_patterns_1.T, vmin=0.0, vmax=2.0, cmap=cmap_black_white, origin='lower')
ax.set_xticks(np.arange(0.5, len(cluster_phate)))
ax.set_xticklabels([])
ax.grid(axis='x')
ax.set_yticks([])
ax.set_ylim(ymax=len(range_region_label)-0.5, ymin=-0.5)
ax.set_ylabel('ROIs', {"fontsize": labelfont_size})
ax.tick_params(which='both', labelsize=tickfont_size)
# label
cluster_phate_1 = [0, 1, 0, 1, 1, 0, 2, 2, 1]
# cluster_phate_1 = cluster_phate
data = np.expand_dims(cluster_phate_1, 1).T
# plot label
ax = fig.add_subplot(gs_shuffle[1, :2])
ax.imshow(data, cmap=cmap_red_magenta, vmin=0, vmax=2)
for (j, i), label in np.ndenumerate(data):
    plt.text(i, j+0.2, np.around(label, 2), ha='center', va='center', fontsize=tickfont_size)
ax.set_yticks([])
ax.set_xticks(np.arange(0, len(cluster_phate_1))+0.5)
ax.set_xticklabels([])
ax.set_xlabel('# avalanche pattern', {"fontsize": labelfont_size}, labelpad=0)
ax.set_ylabel('# cluster', {"fontsize": labelfont_size}, rotation=0, loc='bottom', labelpad=45)
ax.grid(axis='x')

# matrix significant
gs_significant = GridSpec(4, 4, figure=fig, height_ratios=[1.0, 0.1, 0.9, 1.],
                          width_ratios=[0.50, 2.0, 0.2, 0.6])
ax = fig.add_subplot(gs_significant[2, 0])
ax.imshow((pvalue_cluster_all[:, 2, :, :20].swapaxes(0, 1)[0]-pvalue_cluster_all[:, 3, :, :20].swapaxes(0, 1)[0]).T,
           aspect='equal', vmin=-1.0, vmax=1.0, cmap=cmap_red_blue, origin='lower')
ax.tick_params('both', pad=1)
ax.set_xticks([0, 3, 6])
ax.set_xticklabels([1, 4, 7])
ax.set_xlabel('# cluster  ', {"fontsize": labelfont_size})
ax.set_ylabel('ROIs', {"fontsize": labelfont_size}, labelpad=0)
ax.set_yticks([])
ax.set_title('                 Significant cluster affiliation', {"fontsize": titlefont_size}, pad=13)
ax.tick_params(which='both', labelsize=tickfont_size)
for i in range(7):
    ax.annotate('...', xy=(i-1., 20.1), fontsize=titlefont_size, annotation_clip=False, rotation=90, va='bottom', ha='center')
########################### legend significant ##################################
gs_null_legend = GridSpec(8, 5, figure=fig, height_ratios=[0.9, 0.1, 0.6, 0.1, 0.1, 0.1, 0.4, 1.],
                          width_ratios=[0.25, 1., 1.25, 0.2, 0.6], bottom=0.005)
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


# entropy
gs_entropy = GridSpec(3, 5, figure=fig, height_ratios=[1, 1, 0.8], width_ratios=[0.2, 0.15, 0.75, 1.9, 1.], wspace=0.05)
d = .015  # how big to make the diagonal lines in axes coordinates
ax = fig.add_subplot(gs_entropy[2, 2])
y, x, _ = ax.hist(entropy_values_all, bins=10, histtype='step', color='black')
ax.set_ylim(ymax=0.3*nb_randomize_1+500)
ax.tick_params('both', pad=1)
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
                                 arrowstyle='->,head_width=.2', mutation_scale=20)
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
ax.tick_params('x', pad=0.8)
kwargs.update(transform=ax.transAxes)  # switch to the bottom axes
ax.plot((0.95-d, 1.05+d), (-d, +d), **kwargs)
ax.plot((0.95-d, 1.05+d), (1-d, 1+d), **kwargs)
ax.set_title('                            Estimation distribution', {"fontsize": titlefont_size})
ax.set_xlabel('                          entropy (nats)', {"fontsize": labelfont_size})
plt.subplots_adjust(wspace=0.02)
ax.tick_params('both', labelsize=tickfont_size)


################################### null model 3 ###########################
# example 1
ax = fig.add_subplot(gs_null_1[0, 2])
ax.set_title('Null model 3\n$\\bf{shuffle\ active\ region\ by}$\n$\\bf{avalanche\ pattern}$', {"fontsize": titlefont_size}, pad=2, y=1)
ax.set_axis_off()
# plot shuffles
ax = fig.add_subplot(gs_shuffle[0, 4:])
avalanches_patterns_shuffle = []
for av in avalanches_patterns_1:
    av_copy = np.copy(av)
    np.random.shuffle(av_copy)
    avalanches_patterns_shuffle.append(av_copy)
ax.imshow(np.concatenate([avalanches_patterns_shuffle]).T, vmin=0.0, vmax=2.0, cmap=cmap_black_white, origin='lower')
ax.set_xticks(np.arange(0.5, len(cluster_phate)))
ax.set_xticklabels([])
ax.grid(axis='x')
ax.set_yticks([])
ax.set_ylim(ymax=len(range_region_label)-0.5, ymin=-0.5)
ax.tick_params(which='both', labelsize=tickfont_size)

# plot label
ax = fig.add_subplot(gs_shuffle[1, 4:])
data =np.expand_dims([1, 0, 1, 2, 0, 2, 1, 0, 1], 1).T
ax.imshow(data, cmap=cmap_bleu_green, vmin=0, vmax=2)
for (j, i), label in np.ndenumerate(data):
    plt.text(i, j+0.2, np.around(label, 2), ha='center', va='center', fontsize=tickfont_size)
ax.set_yticks([])
ax.set_xticks(np.arange(0, len(cluster_phate_1))+0.5)
ax.set_xticklabels([])
ax.set_xlabel('# avalanche pattern', {"fontsize": labelfont_size}, labelpad=0)
ax.set_ylabel('# cluster', {"fontsize": labelfont_size}, rotation=0, loc='bottom', labelpad=45)
ax.grid(axis='x')

# matrix significant
ax = fig.add_subplot(gs_significant[2, 2])
ax.imshow((pvalue_cluster_all_null_model[:, 2, :, :20].swapaxes(0, 1)[0]-pvalue_cluster_all_null_model[:, 3, :, :20].swapaxes(0, 1)[0]).T,
          vmin=-1.0, vmax=1.0, cmap=cmap_red_blue, origin='lower')
ax.tick_params('both', pad=1)
ax.set_xticks([0, 3, 6])
ax.set_xticklabels([1, 4, 7])
ax.set_xlabel('# cluster  ', {"fontsize": labelfont_size})
ax.set_ylabel('ROIs', {"fontsize": labelfont_size}, labelpad=0)
ax.set_yticks([])
ax.set_title('                    Significant cluster affiliation', {"fontsize": titlefont_size}, pad=13)
ax.tick_params(which='both', labelsize=tickfont_size)
for i in range(7):
    ax.annotate('...', xy=(i-1., 20.1), fontsize=titlefont_size, annotation_clip=False, rotation=90, va='bottom', ha='center')
########################### legend significant ##################################
gs_null_legend = GridSpec(8, 6, figure=fig, height_ratios=[0.9, 0.1, 0.6, 0.1, 0.1, 0.1, 0.4, 1.],
                          width_ratios=[0.25, 1., 1.25, 0.2, 0.4, 0.2], bottom=0.005)
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

# entropy
gs_entropy_1 = GridSpec(3, 3, figure=fig, width_ratios=[1., 1., 0.6], height_ratios=[1., 1.15, 0.85])
ax = fig.add_subplot(gs_entropy_1[2, 2])
y, x, _ = ax.hist(entropy_values_null_model, bins=10, histtype='step', color='black')
ax.set_ylim(ymax=0.3*nb_randomize_2+5)
arr_1 = mpatches.FancyArrowPatch((entropy(data_patient.ravel(), base=None), 8.),
                                 (entropy(data_patient.ravel(), base=None), 0.0),
                                 color='r',
                                 arrowstyle='->,head_width=.2', mutation_scale=20)
arr_1.set_clip_on(False)
ax.add_patch(arr_1)
ax.annotate('data', (0.5, 1.), xycoords=arr_1, ha='center', va='bottom', annotation_clip=False, color='red')
ax.yaxis.set_major_formatter(FuncFormatter(to_percent(n=nb_randomize_2)))
ax.set_xlim(xmax=(entropy(data_patient.ravel(), base=None)+0.1))
ax.set_ylabel('% of distribution', {"fontsize": labelfont_size}, labelpad=0)
ax.set_xlabel('entropy (nats)', {"fontsize": labelfont_size})
ax.tick_params(which='both', labelsize=tickfont_size)
ax.set_title('Estimation distribution', {"fontsize": titlefont_size})


############################### null model 2 ########################
ax = fig.add_subplot(gs_null_1[0, 1])
ax.set_title('Null model 2\n$\\bf{shuffle\ avalanche}$\n$\\bf{pattern\ order}$', {"fontsize": titlefont_size}, pad=2, y=1)
ax.set_axis_off()

ax = fig.add_subplot(gs_shuffle[0, 2:4])
avalanches_patterns_copy = np.copy(avalanches_patterns_1)
indexes = np.arange(0, len(avalanches_patterns_1), 1)
np.random.shuffle(indexes)
avalanches_patterns_copy = avalanches_patterns_copy[indexes]
ax.imshow(avalanches_patterns_copy.T, vmin=0.0, vmax=2.0, cmap=cmap_black_white, origin='lower')
ax.set_xticks(np.arange(0.5, len(cluster_phate)))
ax.set_xticklabels([])
ax.grid(axis='x')
ax.set_yticks([])
ax.set_ylim(ymax=len(range_region_label)-0.5, ymin=-0.5)
ax.tick_params(which='both', labelsize=tickfont_size)

# plot label
ax = fig.add_subplot(gs_shuffle[1, 2:4])
ax.imshow([data[0][indexes]], cmap=cmap_red_magenta, vmin=0, vmax=2)
for (j, i), label in np.ndenumerate([data[0][indexes]]):
    plt.text(i, j+0.2, np.around(label, 2), ha='center', va='center', fontsize=tickfont_size)
ax.set_yticks([])
ax.set_xticks(np.arange(0, len(cluster_phate_1))+0.5)
ax.set_xticklabels([])
ax.set_xlabel('# avalanche pattern', {"fontsize": labelfont_size}, labelpad=0)
ax.set_ylabel('# cluster', {"fontsize": labelfont_size}, rotation=0, loc='bottom', labelpad=45)
ax.grid(axis='x')

# transitions
gs_shuffle_2 = GridSpec(4, 6, figure=fig, height_ratios=[0.6, 0.1, 1.9, 1.],
                        width_ratios=[1.3, 1.3, 0.7, 1.2, 1.2, 1.2])
ax = fig.add_subplot(gs_shuffle_2[2, 2])
transitions = np.load(path + "/transition_all.npy")
im_transition = ax.imshow(transitions*100, vmin=0, vmax=35, aspect='equal')
ax.set_xticks([0.1, 3.1, 6.1])
ax.set_xticklabels([1, 4, 7])
ax.set_yticks([0.1, 3.1, 6.1])
ax.set_yticklabels([1, 4, 7])
ax.set_title('Observed CTM', {"fontsize": titlefont_size}, pad=10)
ax.set_ylabel('# cluster', {"fontsize": labelfont_size}, labelpad=0)
ax.set_xlabel('# cluster', {"fontsize": labelfont_size}, labelpad=0)
ax.tick_params(which='both', labelsize=tickfont_size)
cax = fig.add_axes([0.4, 0.31, 0.10, 0.01])
colorbar_transition = fig.colorbar(im_transition, cax=cax, orientation='horizontal')
colorbar_transition.ax.xaxis.set_tick_params(labelsize=tickfont_size)
colorbar_transition.ax.set_xlabel('% of transitions', {"fontsize": labelfont_size}, labelpad=2)
colorbar_transition.ax.xaxis.set_ticks_position('top')
colorbar_transition.set_ticks([0, 35])
colorbar_transition.set_ticklabels([0, 35])
colorbar_transition.ax.tick_params(axis='x', pad=0)

nb = 3
for i in range(nb):
    gs_null_3 = GridSpec(7, 4, figure=fig, height_ratios=[i*0.8, 3., 1., 1.5, 1., 2., 1-i*0.8],
                         width_ratios=[0.5-i*0.1, 0.4, 3., i*0.1])
    ax = fig.add_subplot(gs_null_3[3, 2])
    transitions = np.load(path + "/transition_all"+str(i)+".npy")
    im_transition = ax.imshow(transitions, vmin=0.0, vmax=0.35, aspect='equal')
    if i == nb-1:
        ax.set_xticks([0.1, 3.1, 6.1])
        ax.set_xticklabels([1, 4, 7])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_ylabel('# cluster', {"fontsize": labelfont_size}, labelpad=0)
        ax.set_xlabel('# cluster', {"fontsize": labelfont_size}, labelpad=0)
        ax.tick_params(which='both', labelsize=tickfont_size)
    elif i == 0:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Null CTM    ', {"fontsize": titlefont_size}, pad=9)
    else:
        # ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])

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
ax = fig.add_subplot(gs_shuffle_2[3, 2:4])
im_2 = ax.imshow(pvalue_all, vmin=significatif, vmax=1 - significatif, cmap=cmap_red_blue)
ax.set_xticks([0.1, 3.1, 6.1])
ax.set_xticklabels([1, 4, 7])
ax.set_yticks([0.1, 3.1, 6.1])
ax.set_yticklabels([1, 4, 7])
ax.set_ylabel('# cluster', {"fontsize": labelfont_size}, labelpad=0)
ax.set_xlabel('# cluster    ', {"fontsize": labelfont_size})
ax.tick_params(which='both', labelsize=tickfont_size)
ax.set_title('Significant CTM', {"fontsize": titlefont_size}, pad=9)
gs_shuffle_3 = GridSpec(4, 6, figure=fig, height_ratios=[0.9, 0.1, 1.25, 0.75],
                        width_ratios=[1., 1., 1.25, 0.75, 1., 1.])


plt.annotate('A', (0., 0.975), xycoords='figure fraction', fontsize=titlefont_size, weight='bold')
plt.annotate('B', (0., 0.545), xycoords='figure fraction', fontsize=titlefont_size, weight='bold')
plt.annotate('C', (0., 0.25), xycoords='figure fraction', fontsize=titlefont_size, weight='bold')

plt.annotate('D', (0.35, 0.975), xycoords='figure fraction', fontsize=titlefont_size, weight='bold')
plt.annotate('E', (0.35, 0.545), xycoords='figure fraction', fontsize=titlefont_size, weight='bold')
plt.annotate('G', (0.43, 0.25), xycoords='figure fraction', fontsize=titlefont_size, weight='bold')
plt.annotate('F', (0.53, 0.545), xycoords='figure fraction', fontsize=titlefont_size, weight='bold')


plt.annotate('H', (0.69, 0.975), xycoords='figure fraction', fontsize=titlefont_size, weight='bold')
plt.annotate('I', (0.69, 0.545), xycoords='figure fraction', fontsize=titlefont_size, weight='bold')
plt.annotate('J', (0.69, 0.25), xycoords='figure fraction', fontsize=titlefont_size, weight='bold')

plt.subplots_adjust(left=0.05, right=0.99, bottom=0.07, top=0.89, hspace=0.6)

plt.savefig('figure_3/figure_5_pre.png')
plt.savefig('figure_3/figure_5_pre.svg')
plt.savefig('figure_3/figure_5_pre.eps')
plt.savefig('figure_3/figure_5_pre.tiff', dpi=800)
plt.show()

