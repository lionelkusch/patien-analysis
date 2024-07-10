import os
import h5py
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pipeline_phate_clustering.functions_helper.load_data import go_avalanches
from helper_function import add_arrows
from matplotlib.gridspec import GridSpec


label_size = 12.0
tickfont_size = 10.0
axis_show = True

# Preparation data for the pipeline
path_data = os.path.dirname(os.path.realpath(__file__)) + '/../../data/'
f = h5py.File(path_data + 'serie_Melbourne.mat', 'r')
struArray = f['D']
Nsubs = 44
nregions = 90
selected_subjects = ['43', '39', '38', '35', '34', '29', '26', '21', '20', '19', '18', '17', '15', '13', '9', '8', '6',
                     '5']
data_subjects = []
for i in range(Nsubs, 1, -1):
    name = '%d' % i
    if name in selected_subjects:
        data_subjects.append(np.swapaxes(f[struArray[i, 0]][:nregions, :], 0, 1))
data_subjects = np.array(data_subjects)

# zscore data
data_zscore = []
for data in data_subjects:
    data_zscore.append(stats.zscore(data))
data_zscore = np.array(data_zscore)

# compute the avalanches for each patient
avalanches_threshold = 3
avalanches_direction = 0
avalanches_binsize = 1
avalanches_bin = []
subjects_index = []
data_avalanches = []
data_avalanches_bin = []
for data in data_subjects:
    Avalanches_human = go_avalanches(data, thre=avalanches_threshold,
                                     direc=avalanches_direction, binsize=avalanches_binsize)
    avalanches = []
    avalanches_bin = []
    for kk1 in range(len(Avalanches_human['ranges'])):
        begin = Avalanches_human['ranges'][kk1][0]
        end = Avalanches_human['ranges'][kk1][1]
        avalanches.append(Avalanches_human['Zbin'][begin:end, :])
        sum = np.sum(Avalanches_human['Zbin'][begin:end, :], 0)
        out = np.zeros(nregions)
        out[np.where(sum >= 1)] = 1
        avalanches_bin.append(out)
    data_avalanches.append(np.concatenate(avalanches))
    data_avalanches_bin.append(np.concatenate([avalanches_bin]))
data_avalanches = np.array(data_avalanches)
data_avalanches_bin = np.array(data_avalanches_bin)
pca_choice = 5
print('process PCA data')
PCA_fit_data_subject = PCA(n_components=pca_choice).fit_transform(np.concatenate(data_subjects))
print('process PCA data normalized')
PCA_fit_data_zscore = PCA(n_components=pca_choice).fit_transform(np.concatenate(data_zscore))
print('process PCA avalanche')
PCA_fit_data_avalanche = PCA(n_components=pca_choice).fit_transform(np.concatenate(data_avalanches))
print('process PCA avalanche pattern')
PCA_fit_data_avalanche_pattern = PCA(n_components=pca_choice).fit_transform(np.concatenate(data_avalanches_bin))

letter = ['A', 'B', 'C', 'D', 'E']

fig = plt.figure(figsize=(6.8, 3.4))
gs_1 = GridSpec(2, 8, figure=fig, width_ratios=[1., 1., 0.1, 1., 1., 0.1, 1., 1.])
for index_data, (PCA_fit, data_subject, index_ax, title, x_position_title) in enumerate([
    (PCA_fit_data_zscore, data_zscore, 1, 'Normalized data', 0.04),
    (PCA_fit_data_avalanche, data_avalanches, 3, 'Avalanches', 0.45),
    (PCA_fit_data_avalanche_pattern, data_avalanches_bin, 5, 'Avalanche patterns', 0.7)]):
    ax1 = fig.add_subplot(gs_1[0, index_data*3])
    patient_time = []
    begin = 0
    for index, data in enumerate(data_subject):
        end = begin + len(data)
        ax1.scatter(PCA_fit[begin:end, 0], PCA_fit[begin:end, 1], c=np.arange(len(data)), s=0.8)
        patient_time.append((begin, end))
        begin = end
        ax1.axis('off')
    ax1.annotate(title, xy=(x_position_title, 0.95), xycoords='figure fraction', fontsize=label_size, weight='bold')
    ax1.annotate(letter[index_data], xy=(-0.27, 1.1), xycoords='axes fraction', weight='bold', fontsize=label_size)
    ax1.set_title('pooled data')
    for nb_ax_x, nb_ax_y, nb_patient in [(1, 0, 0), (0, 1, 1), (1, 1, 2)]:
        ax = fig.add_subplot(gs_1[nb_ax_y, index_data*3+nb_ax_x])
        scatter_plot = ax.scatter(PCA_fit[patient_time[nb_patient][0]:patient_time[nb_patient][1], 0],
               PCA_fit[patient_time[nb_patient][0]:patient_time[nb_patient][1], 1],
               c=np.arange(patient_time[nb_patient][1]-patient_time[nb_patient][0]), s=0.8)
        ax.axis('off')
        if nb_patient == 1:
            add_arrows(PCA_fit[patient_time[nb_patient][0]:patient_time[nb_patient][1], 0],
                       PCA_fit[patient_time[nb_patient][0]:patient_time[nb_patient][1], 1],
                       ax, "PCA1", "PCA2")
        ax.set_title('#'+str(nb_patient))

ax = fig.add_axes([0.25, 0.08, 0.5, 0.02])
colbar_time = fig.colorbar(scatter_plot, cax=ax, orientation='horizontal')
colbar_time.ax.yaxis.tick_left()
colbar_time.set_ticks([0, patient_time[2][1]-patient_time[2][0] - 1])
colbar_time.set_ticklabels(['t=0', 't=end'])
colbar_time.ax.xaxis.set_tick_params(pad=0.1, labelsize=tickfont_size)
colbar_time.ax.set_xlabel('time evolution', {"fontsize": label_size}, labelpad=-7)

plt.subplots_adjust(left=0.035, top=0.85, bottom=0.18, right=0.99, hspace=0.25)
plt.savefig('figure/figure_2_pre.png')
plt.savefig('figure/figure_2_pre.svg')
plt.show()
