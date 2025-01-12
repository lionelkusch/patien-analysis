import os
import numpy as np
import matplotlib.pyplot as plt
from pipeline_phate_clustering.null_model_sensibility_analysis.get_result_null_model import get_color_map
from matplotlib.gridspec import GridSpec
import scipy.io as io
from pipeline_phate_clustering.functions_helper.plot_brain import get_region_select

path = os.path.dirname(os.path.realpath(__file__)) + '/../../matlab/library/'
region_select = get_region_select()
region_names = np.concatenate(io.loadmat(path +'/ROI_MNI_V4_List.mat')['ROI']['Nom_L'][0][:90])
# region_names = []
# for name in np.concatenate(io.loadmat(path +'/ROI_MNI_V4_List.mat')['ROI']['Nom_L'][0]):
#     if name in region_select:
#         region_names.append(name)


titlefont_size = 12.0
tickfont_size = 8.0
labelfont_size = 10.0
letter_font_size = 12
cmap_red_blue = get_color_map()
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
pvalue_cluster_all_null_model = np.array(pvalue_cluster_all_null_model)


fig = plt.figure(figsize=(6.4, 8.4))
gs = GridSpec(1, 2, figure=fig)
ax = fig.add_subplot(gs[0, 0])
ax.imshow((pvalue_cluster_all[:, 2, :, :].swapaxes(0, 1)[0]-pvalue_cluster_all[:, 3, :, :].swapaxes(0, 1)[0]).T,
          vmin=-1.0, vmax=1.0, cmap=cmap_red_blue, origin='lower')
ax.tick_params('both', pad=1)
ax.set_xticks([0, 3, 6])
ax.set_xticklabels([1, 4, 7])
ax.set_xlabel('# cluster')
ax.set_ylabel('ROIs', {"fontsize": labelfont_size}, labelpad=0)
ax.set_yticks(np.arange(0, len(region_names), 1)[::2])
ax.set_yticklabels(region_names[::2], {"fontsize": 7})
ax.set_title('Null model 1', {"fontsize": titlefont_size}, pad=13)
ax.tick_params(which='both', labelsize=tickfont_size)
ax.annotate('A', xy=(-1., 1.025), xycoords='axes fraction', weight='bold', fontsize=labelfont_size)

ax = fig.add_subplot(gs[0, 1])
ax.imshow((pvalue_cluster_all_null_model[:, 2, :, :].swapaxes(0, 1)[0]-pvalue_cluster_all_null_model[:, 3, :, :].swapaxes(0, 1)[0]).T,
          vmin=-1.0, vmax=1.0, cmap=cmap_red_blue, origin='lower')
ax.tick_params('both', pad=1)
ax.set_xticks([0, 3, 6])
ax.set_xticklabels([1, 4, 7])
ax.set_xlabel('# cluster')
ax.set_ylabel('ROIs', {"fontsize": labelfont_size}, labelpad=0)
ax.set_yticks(np.arange(0, len(region_names), 1)[1::2])
ax.set_yticklabels(region_names[1::2], {"fontsize": 7})
ax.set_title('Null model 3', {"fontsize": titlefont_size}, pad=13)
ax.tick_params(which='both', labelsize=tickfont_size)
ax.annotate('B', xy=(-1., 1.025), xycoords='axes fraction', weight='bold', fontsize=labelfont_size)


########################### legend significant ##################################
gs_null_legend = GridSpec(5, 3, figure=fig,
                          height_ratios=[1, 0.3, 0.3, 0.3, 1.],
                          width_ratios=[1., 1., 0.1]
                          )
ax = fig.add_subplot(gs_null_legend[1, 2])
ax.imshow([[-1.0]], vmin=-1.0, vmax=1.0, cmap=cmap_red_blue, origin='lower')
ax.set_yticks([0])
ax.set_yticklabels(['significant\nactive'], fontdict={'fontsize': labelfont_size})
ax.set_xticks([])
ax.set_xlim(xmin=0, xmax=0.5)
ax = fig.add_subplot(gs_null_legend[2, 2])
ax.imshow([[0.0]], vmin=-1.0, vmax=1.0, cmap=cmap_red_blue, origin='lower')
ax.set_yticks([0])
ax.set_yticklabels(['no\nsignificant'], fontdict={'fontsize': labelfont_size})
ax.set_xticks([])
ax.set_xlim(xmin=0, xmax=0.5)
ax = fig.add_subplot(gs_null_legend[3, 2])
ax.imshow([[1.0]], vmin=-1.0, vmax=1.0, cmap=cmap_red_blue, origin='lower')
ax.set_yticks([0])
ax.set_yticklabels(['significant\ninactive'], fontdict={'fontsize': labelfont_size})
ax.set_xticks([])
ax.set_xlim(xmin=0, xmax=0.5)

plt.subplots_adjust(top=0.96, bottom=0.05, left=0.05, right=0.98, wspace=0., hspace=0.)
plt.savefig('figure/figure_SP16.png')
plt.savefig('figure/figure_SP16.svg')
plt.show()