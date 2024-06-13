import os
import h5py
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
from pipeline_phate_clustering.functions_helper.load_data import go_avalanches


label_size = 12.0
tickfont_size = 10.0
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
avalanches_bin = []
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

fig = plt.figure(figsize=(2.4, 3.6))
gs = GridSpec(4, 1, figure=fig, height_ratios=[1, 1, 0.5, 0.5])
ax1 = fig.add_subplot(gs[0])
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
ax1.set_xticklabels([])
ax1.tick_params(axis='both', labelsize=tickfont_size)
ax1.annotate('A', xy=(-0.2, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)

ax2 = fig.add_subplot(gs[1])
data = data_zscore[0][range_time[0]:range_time[1], regions_select[0]:regions_select[1]]
max = np.max(data)
min = np.min(data)
data = np.arange(0, 10, 1)*(max-min)+data
ax2.plot(data, color='black', linewidth=linewidth)
ax2.set_xlim(xmin=0, xmax=range_time[1]-range_time[0])
ax2.set_yticks(range_region_label[0::4]*(max-min))
ax2.set_yticklabels(range_region_label[0::4])
ax2.set_ylabel('RIO', {"fontsize": label_size})
ax2.set_xticks([0, (range_time[1]-range_time[0])/2, (range_time[1]-range_time[0])])
ax2.set_xticklabels([])
ax2.tick_params(axis='both', labelsize=tickfont_size)
ax2.annotate('B', xy=(-0.2, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)

ax3 = fig.add_subplot(gs[2])
data = Avalanches_human['Zbin'][range_time[0]:range_time[1], regions_select[0]:regions_select[1]]
cmap = ListedColormap(["white", "black", "white"], name='from_list', N=None)
im = ax3.imshow(data.T, interpolation='none', aspect='auto', cmap=cmap, vmin=0.0, vmax=2.0, origin='lower')
ax3.set_xticks([0, (range_time[1]-range_time[0])/2, (range_time[1]-range_time[0])])
ax3.set_xlabel('time in ms', {"fontsize": label_size}, labelpad=0)
ax3.set_ylabel('RIO', {"fontsize": label_size})
ax3.annotate('C', xy=(-0.2, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)

# Create a Rectangle patch
index = np.where(np.sum(data, axis=1))[0]
begin = np.concatenate([[index[0]], index[np.where(np.diff(index) > 1)[0]+1]])
end = np.concatenate([index[np.where(np.diff(index) > 1)[0]], [index[-1]]])
for begin_avalanche, end_avalanche in zip(begin, end):
    time_avalanches = end_avalanche-begin_avalanche+1
    rect = patches.Rectangle((begin_avalanche-0.5, -0.5), time_avalanches,
                             regions_select[1]-regions_select[0]+0.5,
                             linewidth=0.0, edgecolor='none', facecolor='grey', alpha=0.5)
    ax3.add_patch(rect)
ax3.set_ylim(ymin=0, ymax=10)
ax3.set_yticks(range_region_label[0::4]+0.5)
ax3.set_yticklabels(range_region_label[0::4])
ax3.tick_params(axis='both', labelsize=tickfont_size)
ax3.set_ylim(ymin=0)

ax4 = fig.add_subplot(gs[3])
avalanches = []
for begin_avalanche, end_avalanche in zip(begin, end):
    avalanches.append(np.array(np.sum(data[begin_avalanche:end_avalanche+1, :], axis=0)>=1, dtype=int))
    print(begin_avalanche, end_avalanche, avalanches[-1])
avalanches = np.concatenate([avalanches])
np.save('avalanches_pattern_example_fig_1.npy', avalanches)
ax4.imshow(avalanches.T, interpolation='none', cmap=cmap, vmin=0.0, vmax=2.0, origin='lower')
ax4.set_yticks(range_region_label[0::4]+0.5)
ax4.set_yticklabels(range_region_label[0::4])
ax4.set_ylim(ymax=len(range_region_label)-1, ymin=0)
ax4.set_xticks(np.array([0, np.int(np.floor(avalanches.shape[0]/2)), avalanches.shape[0]])+0.5)
ax4.set_xticklabels([0, np.int(np.floor(avalanches.shape[0]/2)), avalanches.shape[0]-1])
ax4.set_xlabel('      avalanche patterns', {"fontsize": label_size}, labelpad=0)
ax4.set_ylabel('RIO', {"fontsize": label_size})
ax4.set_aspect(aspect='equal', anchor='SW')
ax4.annotate('D', xy=(-0.85, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)

plt.subplots_adjust(left=0.18, top=0.975, right=0.94, hspace=0.5)
plt.savefig('figure/figure_1a.png', density=800)
plt.savefig('figure/figure_1a.svg', density=800)
plt.show()