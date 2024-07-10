import os
import h5py
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from helper_function import add_arrows
from pipeline_phate_clustering.functions_helper.load_data import go_avalanches
from matplotlib import cm
from matplotlib.colors import Normalize


label_size = 12.0
tickfont_size = 10.0

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

letter = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']

for index_PCA, (name, PCA_fit, data_subject) in enumerate([
    ('time_PCA_Data_zscore', PCA_fit_data_zscore, data_zscore),
    ('time_PCA_avalanche', PCA_fit_data_avalanche, data_avalanches),
    ('time_PCA_avalanche_pattern', PCA_fit_data_avalanche_pattern, data_avalanches_bin)]):

    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot(4, 5, 1)
    patient_time = []
    begin = 0
    for index, data in enumerate(data_subject):
        end = begin + len(data)
        ax1.scatter(PCA_fit[begin:end, 0], PCA_fit[begin:end, 1], c=np.arange(len(data)), s=0.8)
        patient_time.append((begin, end))
        begin = end
    ax1.axis('off')
    ax1.annotate(letter[0], xy=(-0.1, 0.9), xycoords='axes fraction', weight='bold', fontsize=label_size)
    ax1.set_title('pooled data')
    add_arrows(PCA_fit[:, 0], PCA_fit[:, 1], ax1, "PCA1", "PCA2")
    for nb_patient in range(len(data_subjects)):
        ax = plt.subplot(4, 5, nb_patient+2)
        scatter_plot = ax.scatter(PCA_fit[patient_time[nb_patient][0]:patient_time[nb_patient][1], 0],
                   PCA_fit[patient_time[nb_patient][0]:patient_time[nb_patient][1], 1],
                   c=np.arange(patient_time[nb_patient][1]-patient_time[nb_patient][0]), s=0.8)
        ax.axis('off')
        ax.annotate(letter[nb_patient+1], xy=(-0.1, 0.9), xycoords='axes fraction', weight='bold', fontsize=label_size)
        ax.set_title('#' + str(nb_patient))
    ax = fig.add_axes([0.25, 0.03, 0.5, 0.02])
    colbar_time = fig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=0., vmax=1.)), cax=ax, orientation='horizontal')
    colbar_time.ax.yaxis.tick_left()
    colbar_time.set_ticks([0, patient_time[2][1]-patient_time[2][0] - 1])
    colbar_time.set_ticklabels(['t=0', 't=end'])
    colbar_time.ax.xaxis.set_tick_params(pad=0.1, labelsize=tickfont_size)
    colbar_time.ax.set_xlabel('time evolution', {"fontsize": label_size}, labelpad=-7)

    plt.subplots_adjust(left=0.04, right=0.97, top=0.97, bottom=0.06, wspace=0.6, hspace=0.6)
    plt.savefig('figure/SP_'+str(2+index_PCA)+'_'+name+'.png')
    # plt.show()

