import os
import h5py
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap, Normalize
import matplotlib.patches as patches
from matplotlib import cm
from sklearn.cluster import KMeans
from pipeline_phate_clustering.functions_helper.load_data import go_avalanches
from pipeline_phate_clustering.functions_helper.plot_brain_test import multiview_pyvista, get_region_select
from pipeline_phate_clustering.functions_helper.plot_brain import get_brain_mesh
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams['svg.fonttype'] = 'none'

label_size = 12.0
tickfont_size = 10.0
label_col_size = 8.0
linewidth = 0.5

range_time = (900, 1400)
regions_select = (0, 10)

# Preparation data for the pipeline
path_data = os.path.dirname(os.path.realpath(__file__)) + '/../../data/'
f = h5py.File(path_data + 'serie_Melbourne.mat', 'r')
struArray = f['D']
Nsubs = 44
nregions = 90
selected_subjects = ['5']
data_subject = []
for i in range(Nsubs):
    name = '%d' % i
    if name in selected_subjects:
        data_subject.append(np.swapaxes(f[struArray[i, 0]][:nregions, :], 0, 1))
data_subject = np.array(data_subject)

# zscore data
data_zscore = []
for data in data_subject:
    data_zscore.append(stats.zscore(data))
data_zscore = np.array(data_zscore)

# compute the avalanches for each patient
avalanches_threshold = 3
avalanches_direction = 0
avalanches_binsize = 1
subjects_index = []
data_avalanches = []
for data in data_subject:
    Avalanches_human = go_avalanches(data, thre=avalanches_threshold, direc=avalanches_direction,
                                     binsize=avalanches_binsize)
    avalanches = []
    for kk1 in range(len(Avalanches_human['ranges'])):
        begin = Avalanches_human['ranges'][kk1][0]
        end = Avalanches_human['ranges'][kk1][1]
        avalanches.append(Avalanches_human['Zbin'][begin:end, :])
    data_avalanches.append(np.concatenate(avalanches))
data_avalanches_1 = Avalanches_human['Zbin'][range_time[0]:range_time[1], regions_select[0]:regions_select[1]]
index = np.where(np.sum(data_avalanches_1, axis=1))[0]
begin = np.concatenate([[index[0]], index[np.where(np.diff(index) > 1)[0]+1]])
end = np.concatenate([index[np.where(np.diff(index) > 1)[0]], [index[-1]]])
avalanches = []
for begin_avalanche, end_avalanche in zip(begin, end):
    if begin_avalanche - end_avalanche == 0:
        avalanches.append(np.array(np.sum(data_avalanches_1[begin_avalanche:end_avalanche+1, :], axis=0) >= 1, dtype=int))
    else:
        avalanches.append(np.array(np.sum(data_avalanches_1[begin_avalanche:end_avalanche, :], axis=0) >= 1, dtype=int))
avalanches = np.concatenate([avalanches])
np.save('avalanches_example.npy',avalanches)

# data from all subjects
path = os.path.dirname(os.path.realpath(__file__)) + "/../../paper/result/default/"
avalanches_pattern = np.load(path + "/avalanches.npy", allow_pickle=True)
kmeans_nb_cluster = 7
kmeans_seed = 123
PHATE_n_pca = 5
Y_phate = np.load(path + "/Phate.npy")
cluster_phate = KMeans(n_clusters=kmeans_nb_cluster, random_state=kmeans_seed).fit_predict(Y_phate)
cluster_vector = []
for i in range(kmeans_nb_cluster):
    cluster_vector.append(np.mean(np.concatenate(avalanches_pattern)[np.where(cluster_phate == i)], axis=0)[:78])
cluster_vector = np.array(cluster_vector)
transitions = np.load(path + "/transition_all.npy")

colormap = plt.get_cmap('Accent', kmeans_nb_cluster)
range_region = np.arange(0, 10, 1)
range_region_label = np.arange(0, 11, 1)

fig = plt.figure(figsize=(6.8, 6.8))
gs = GridSpec(3, 6, figure=fig)

ax1 = fig.add_subplot(gs[0, :3])
data = data_zscore[0][range_time[0]:range_time[1], regions_select[0]:regions_select[1]]
max = np.max(data)
min = np.min(data)
data = np.arange(0, 10, 1)*(max-min)+data
mean_zscore_up = np.arange(0, 10, 1)*(max-min) + 3
mean_zscore_low = np.arange(0, 10, 1)*(max-min) - 3
ax1.plot(data, color='black', linewidth=linewidth)
ax1.hlines(mean_zscore_up, 0, range_time[1]-range_time[0], color='red', linewidth=linewidth*0.5)
ax1.hlines(mean_zscore_low, 0, range_time[1]-range_time[0], color='red', linewidth=linewidth*0.5)
ax1.set_xlim(xmin=0, xmax=range_time[1]-range_time[0])
ax1.set_yticks(range_region_label[0::4]*(max-min))
ax1.set_yticklabels(range_region_label[0::4])
ax1.set_ylabel('ROI', {"fontsize": label_size})
ax1.set_xticks([0, (range_time[1]-range_time[0])/2, (range_time[1]-range_time[0])])
ax1.set_xticklabels([0, (range_time[1]-range_time[0])/2, (range_time[1]-range_time[0])])
ax1.set_xlabel('time in ms', fontdict={'fontsize': label_size})
ax1.tick_params(axis='both', labelsize=tickfont_size)
ax1.annotate('A', xy=(-0.1, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
ax1.set_title('Reconstructed source normalized\nMEG signals', {"fontsize": label_size})

ax2 = fig.add_subplot(gs[0, 3:])
cmap = ListedColormap(["white", "black", "white"], name='from_list', N=None)
im = ax2.imshow(data_avalanches_1.T, interpolation='none', aspect='auto', cmap=cmap, vmin=0.0, vmax=2.0, origin='lower')
ax2.set_xticks([0, (range_time[1]-range_time[0])/2, (range_time[1]-range_time[0])])
ax2.set_xlabel('time in ms', {"fontsize": label_size}, labelpad=0)
ax2.set_yticks(range_region_label[0::4])
ax2.set_yticklabels(range_region_label[0::4])
ax2.set_ylabel('ROI', {"fontsize": label_size})
ax2.annotate('B', xy=(-0.1, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
ax2.set_title('Avalanches', {"fontsize": label_size})
# Create a Rectangle patch
for begin_avalanche, end_avalanche in zip(begin, end):
    rect = patches.Rectangle((begin_avalanche, -0.5), end_avalanche-begin_avalanche+1,
                             regions_select[1]-regions_select[0]+0.5,
                             linewidth=0.0, edgecolor='none', facecolor='grey', alpha=0.5)
    ax2.add_patch(rect)
# ax2.set_yticks(range_region_label[0::4]+0.5)
# ax2.set_yticklabels(range_region_label[0::4])
ax2.tick_params(axis='both', labelsize=tickfont_size)
ax2.set_ylim(ymin=0)

gs_av = GridSpec(3, 4, figure=fig, width_ratios=[0.8, 1., 1., 0.01])
ax3 = fig.add_subplot(gs_av[1, 0])
ax3.imshow(avalanches.T, interpolation='none', cmap=cmap, vmin=0.0, vmax=2.0, origin='lower')
ax3.set_yticks(range_region_label[0::4]+0.5)
ax3.set_yticklabels(range_region_label[0::4])
ax3.set_ylim(ymin=0, ymax=len(range_region_label)-1)
ax3.set_xticks(np.array([0, np.int(np.floor(avalanches.shape[0]/2)), avalanches.shape[0]-1])+0.5)
ax3.set_xticklabels([0, np.int(np.floor(avalanches.shape[0]/2)), avalanches.shape[0]-1])
ax3.set_xlabel('#avalanche pattern', {"fontsize": label_size}, labelpad=0)
ax3.set_ylabel('ROI', {"fontsize": label_size})
ax3.annotate('C', xy=(-0.3, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
# ax3.set_title('Avalanche patterns', {"fontsize": label_size})

ax4 = fig.add_subplot(gs_av[1, 1])
ax4.scatter(Y_phate[:, 0],
            Y_phate[:, 1],
            color='black', s=0.05)
ax4.set_xticks([])
ax4.set_yticks([])
ax4.set_ylabel('PHATE dim 2', {"fontsize": label_size})
ax4.set_xlabel('PHATE dim 1', {"fontsize": label_size})
ax4.annotate('D', xy=(-0.12, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
ax4.set_title('PHATE reduction', {"fontsize": label_size})
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)

ax5 = fig.add_subplot(gs_av[1, 2])
kmeans_plot=ax5.scatter(Y_phate[:, 0],
            Y_phate[:, 1],
            c=cluster_phate, s=0.05, cmap=colormap, vmin=0,  vmax=7)
ax5.set_xticks([])
ax5.set_yticks([])
ax5.set_ylabel('PHATE dim 2', {"fontsize": label_size})
ax5.set_xlabel('PHATE dim 1', {"fontsize": label_size})
ax5.annotate('E', xy=(-0.12, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
ax5.set_title('K-means', {"fontsize": label_size})
ax5.spines['right'].set_visible(False)
ax5.spines['top'].set_visible(False)
ax = fig.add_axes([0.92, 0.39, 0.01, 0.21])
colbar = fig.colorbar(kmeans_plot, cax=ax, orientation='vertical')
colbar.set_ticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
colbar.set_ticklabels([1, 2, 3, 4, 5, 6, 7])
colbar.ax.yaxis.set_tick_params(pad=0.1, labelsize=tickfont_size)
colbar.ax.set_ylabel('#cluster', {"fontsize": label_size}, labelpad=0)

gs_brain = GridSpec(6, 4, figure=fig)
gs_brain_middle = GridSpec(3, 6, figure=fig, height_ratios=[0.8, 0.8, 1.0])
lhfaces, rhfaces, vertices, region_map = get_brain_mesh()
region_selects = get_region_select()
axs6 = [fig.add_subplot(gs_brain[4, 0]), fig.add_subplot(gs_brain[4, 1]),
         fig.add_subplot(gs_brain[5, 0]), fig.add_subplot(gs_brain[5, 1]), fig.add_subplot(gs_brain_middle[2, 1])]
multiview_pyvista(axs6, vertices, lhfaces, rhfaces, region_map, cluster_vector[5], region_select=region_selects, cmap='plasma')
axs6[0].annotate('F', xy=(-0.5, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
ax = fig.add_axes([0.15, 0.06, 0.3, 0.01])
colbar = fig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=0., vmax=1.), cmap=plt.get_cmap('plasma')), cax=ax, orientation='horizontal')
colbar.set_ticks([-1., 0.0, 1.0])
colbar.set_ticklabels([-1.0, 0.0, 1.0])
colbar.ax.xaxis.set_tick_params(pad=0.1, labelsize=tickfont_size)
colbar.ax.set_xlabel('% activity in a cluster', {"fontsize": label_size}, labelpad=0)

gs_ctm = GridSpec(3, 2, figure=fig, width_ratios=[0.5, 1.0], height_ratios=[1., 1., 1.35])
ax7 = fig.add_subplot(gs_ctm[2, 1])
im_transition = ax7.imshow(transitions)
divider = make_axes_locatable(ax7)
cax = divider.append_axes("right", size="5%", pad=0.05)
colorbar_transition = fig.colorbar(im_transition, cax=cax)
colorbar_transition.ax.yaxis.set_tick_params(pad=0.1, labelsize=tickfont_size)
colorbar_transition.ax.set_ylabel('% of transition', {"fontsize": label_size}, labelpad=2)
ax7.set_xticks([0.1, 3.1, 6.1])
ax7.set_xticklabels([0, 3, 6])
ax7.set_yticks([0.1, 3.1, 6.1])
ax7.set_yticklabels([0, 3, 6])
ax7.annotate('G', xy=(-0.2, 1.), xycoords='axes fraction', weight='bold', fontsize=label_size)
ax7.set_title('CTM', {"fontsize": label_size})
ax7.set_ylabel('#cluster', {"fontsize": label_size})
ax7.set_xlabel('#cluster', {"fontsize": label_size})



plt.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.08, hspace=0.56, wspace=0.80)
plt.savefig('figure/figure_1_pre_pipeline.png')
plt.savefig('figure/figure_1_pre_pipeline.svg')
plt.show()