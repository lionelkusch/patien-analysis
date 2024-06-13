import os
import h5py
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
from sklearn.cluster import KMeans
from phate import PHATE
from pipeline_phate_clustering.functions_helper.load_data import go_avalanches


label_size = 12.0
tickfont_size = 10.0
label_col_size = 8.0
linewidth = 0.5

range_time = (900, 1900)
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

Y_phate = PHATE(n_components=2, n_jobs=8, decay=1.5, n_pca=None, gamma=1, knn=1, knn_dist='cosine',\
                mds_dist='cosine').fit_transform(avalanches)
cluster_phate = KMeans(n_clusters=3, random_state=123).fit_predict(Y_phate)
colormap = plt.get_cmap('Accent', 3)

fig = plt.figure(figsize=(6.8, 6.8))
gs = GridSpec(5, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 0.5, 0.9, 0.18, 0.09])
ax1 = fig.add_subplot(gs[0, 0])
data = data_subject[0][range_time[0]:range_time[1], regions_select[0]:regions_select[1]]
max = np.max(data)
min = np.min(data)
range_region = np.arange(0, 10, 1)
range_region_label = np.arange(0, 11, 1)
data = range_region*(max-min)+data
ax1.plot(data, color='black', linewidth=linewidth)
ax1.set_xlim(xmin=0, xmax=range_time[1]-range_time[0])
ax1.set_yticks(range_region_label[0::4]*(max-min))
ax1.set_yticklabels(range_region_label[0::4])
ax1.set_ylabel('RIO', {"fontsize": label_size})
ax1.set_xticks([0, (range_time[1]-range_time[0])/2, (range_time[1]-range_time[0])])
ax1.set_xticklabels([0, (range_time[1]-range_time[0])/2, (range_time[1]-range_time[0])])
ax1.set_xlabel('time in ms', fontdict={'fontsize': label_size})
ax1.tick_params(axis='both', labelsize=tickfont_size)
ax1.annotate('A', xy=(-0.12, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)

ax2 = fig.add_subplot(gs[0, 1])
data = data_zscore[0][range_time[0]:range_time[1], regions_select[0]:regions_select[1]]
max = np.max(data)
min = np.min(data)
data = np.arange(0, 10, 1)*(max-min)+data
mean_zscore_up = np.arange(0, 10, 1)*(max-min) + 3
mean_zscore_low =  np.arange(0, 10, 1)*(max-min) - 3
ax2.plot(data, color='black', linewidth=linewidth)
ax2.hlines(mean_zscore_up, 0, range_time[1]-range_time[0], color='red', linewidth=linewidth*0.5)
ax2.hlines(mean_zscore_low, 0, range_time[1]-range_time[0], color='red', linewidth=linewidth*0.5)
ax2.set_xlim(xmin=0, xmax=range_time[1]-range_time[0])
ax2.set_yticks(range_region_label[0::4]*(max-min))
ax2.set_yticklabels(range_region_label[0::4])
ax2.set_ylabel('RIO', {"fontsize": label_size})
ax2.set_xticks([0, (range_time[1]-range_time[0])/2, (range_time[1]-range_time[0])])
ax2.set_xticklabels([0, (range_time[1]-range_time[0])/2, (range_time[1]-range_time[0])])
ax2.set_xlabel('time in ms', fontdict={'fontsize': label_size})
ax2.tick_params(axis='both', labelsize=tickfont_size)
ax2.annotate('B', xy=(-0.12, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)

ax3 = fig.add_subplot(gs[1, :])
cmap = ListedColormap(["white", "black", "white"], name='from_list', N=None)
im = ax3.imshow(data_avalanches_1.T, interpolation='none', aspect='auto', cmap=cmap, vmin=0.0, vmax=2.0, origin='lower')
ax3.set_xticks([0, (range_time[1]-range_time[0])/2, (range_time[1]-range_time[0])])
ax3.set_xlabel('time in ms', {"fontsize": label_size}, labelpad=0)
ax3.set_ylabel('RIO', {"fontsize": label_size})
ax3.annotate('C', xy=(-0.05, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)

# Create a Rectangle patch
for begin_avalanche, end_avalanche in zip(begin, end):
    rect = patches.Rectangle((begin_avalanche, -0.5), end_avalanche-begin_avalanche+1,
                             regions_select[1]-regions_select[0]+0.5,
                             linewidth=0.0, edgecolor='none', facecolor='grey', alpha=0.5)
    ax3.add_patch(rect)
ax3.set_yticks(range_region_label[0::4]+0.5)
ax3.set_yticklabels(range_region_label[0::4])
ax3.tick_params(axis='both', labelsize=tickfont_size)
ax3.set_ylim(ymin=0)

ax4 = fig.add_subplot(gs[2, 1])
ax4.imshow(avalanches.T, interpolation='none', cmap=cmap, vmin=0.0, vmax=2.0, origin='lower')
ax4.set_yticks(range_region_label[0::4]+0.5)
ax4.set_yticklabels(range_region_label[0::4])
ax4.set_ylim(ymin=0, ymax=len(range_region_label)-1)
ax4.set_xticks(np.array([0, np.int(np.floor(avalanches.shape[0]/2)), avalanches.shape[0]-1])+0.5)
ax4.set_xticklabels([0, np.int(np.floor(avalanches.shape[0]/2)), avalanches.shape[0]-1])
ax4.set_xlabel('avalanche patterns', {"fontsize": label_size}, labelpad=0)
ax4.set_ylabel('RIO', {"fontsize": label_size})
ax4.annotate('D', xy=(-0.22, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)

ax5 = fig.add_subplot(gs[3, 1])
im_phate = ax5.imshow(Y_phate.T)
ax5.set_yticks([0, 1])
ax5.set_xticks(np.arange(0, len(cluster_phate)-1)+0.5)
ax5.set_xticklabels([])
# ax5.set_xticks(np.array([0, np.int(np.floor(Y_phate.shape[0]/2)), Y_phate.shape[0]-1]))
# ax5.set_xticklabels([0, np.int(np.floor(Y_phate.shape[0]/2)), Y_phate.shape[0]-1])
ax5.annotate('E', xy=(-0.22, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)

ax = fig.add_axes([0.94, 0.15, 0.01, 0.05])
colbar = fig.colorbar(im_phate, cax=ax, orientation='vertical')
colbar.ax.tick_params(axis='both', labelsize=label_col_size)
colbar.set_ticks([0.25, 0.0, -0.20])
colbar.set_ticklabels([0.25, 0.0, -0.20])

ax6 = fig.add_subplot(gs[2:, 0])
im = ax6.scatter(Y_phate[:, 0], Y_phate[:, 1], c=cluster_phate, s=10.0, cmap=colormap, vmin=0,  vmax=2)
ax6.set_xticks([])
ax6.set_yticks([])
ax6.set_ylabel('PHATE dim 2', {"fontsize": label_size})
ax6.set_xlabel('PHATE dim 1', {"fontsize": label_size})
ax6.grid(axis='x')
ax6.annotate('G', xy=(-0.1, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)

ax = fig.add_axes([0.4, 0.03, 0.27, 0.02])
colbar = fig.colorbar(im, cax=ax, orientation='horizontal')
colbar.set_ticks([0.5, 1.5, 2.5])
colbar.set_ticklabels([0, 1, 2])

ax7 = fig.add_subplot(gs[4, 1])
im_cluster = ax7.imshow(np.expand_dims(cluster_phate, 0), cmap=colormap)
# ax7.set_xticks(np.array([0, np.int(np.floor(Y_phate.shape[0]/2)), Y_phate.shape[0]-1]))
# ax7.set_xticklabels([0, np.int(np.floor(Y_phate.shape[0]/2)), Y_phate.shape[0]-1])
ax7.set_xticks(np.arange(0, len(cluster_phate)-1)+0.5)
ax7.set_xticklabels([])
ax7.set_yticks([])
ax7.grid(axis='x')
ax7.annotate('F', xy=(-0.22, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)


plt.subplots_adjust(left=0.07, right=0.97, top=0.98, bottom=0.06, hspace=0.56, wspace=0.16)
plt.savefig('figure/figure_5_pre.png')
plt.show()