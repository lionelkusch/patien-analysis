import os
import h5py
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pipeline_phate_clustering.functions_helper.load_data import go_avalanches
from pipeline_phate_clustering.functions_helper.plot import cnames
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize

label_size = 12.0
tickfont_size = 10.0
list_color = np.array(list(cnames.values()))[:18]
cmap = ListedColormap(list_color, name='from_list', N=18)
colorbar_patient = ScalarMappable(Normalize(0, 18), cmap=cmap)
letter = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']

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

Y_phate_data = np.load(path_data + "/../paper/result/no_avalanche/data_euclidean/Phate.npy")
Y_phate_data_normalized = np.load(path_data + "/../paper/result/no_avalanche/data_normalized_euclidean/Phate.npy")
Y_phate_avalanches = np.load(path_data + "/../paper/result/no_avalanche/avalanches/Phate.npy")
Y_phate_avalanches_euclidean = np.load(path_data + "/../paper/result/no_avalanche/avalanches_2/Phate.npy")
Y_phate_avalanches_pattern = np.load(path_data + "/../paper/result/default/Phate.npy")


for index_data_set, (param, dataset) in enumerate([
        ( [(9, 6), 0.09, 0.96, 0.915, 0.05, 0.6, 0.45, 'PCA'],
            [
             ('PCA_data', 'PCA data', PCA_fit_data_subject, data_subjects),
             ('PCA_normalized_data', 'PCA normalized data', PCA_fit_data_zscore, data_zscore),
             ('PCA_avalanches', 'PCA avalanches', PCA_fit_data_avalanche, data_avalanches),
             ('PCA_avalanche_patterns', 'PCA avalanche patterns', PCA_fit_data_avalanche_pattern, data_avalanches_bin)]),
        ([(9, 9), 0.09, 0.96, 0.93, 0.05, 0.85, 0.45, 'PHATE'],
            [('PHATE_data', 'PHATE data', Y_phate_data, data_subjects),
             ('PHATE_normalized_data', 'PHATE normalized data', Y_phate_data_normalized, data_zscore),
             ('PHATE_avalanches', 'PHATE avalanches', Y_phate_avalanches, data_avalanches),
             ('PHATE_euclidean_avalanches', 'PHATE with euclidean measure of avalanches   ',
             Y_phate_avalanches_euclidean, data_avalanches),
             ('PHATE_avalanche_patterns', 'PHATE avalanche patterns', Y_phate_avalanches_pattern, data_avalanches_bin)])
            ]):
    fig = plt.figure(figsize=param[0])
    col_bars = []
    for index_data, (name_fig, name, data, data_patient_nb) in enumerate(dataset):
        print(int(np.ceil(len(dataset)/2)), index_data*2)
        ax_title = fig.add_subplot(int(np.ceil(len(dataset)/2)), 2, index_data+1)
        ax_title.set_title(name, fontsize=label_size, pad=25)
        ax_title.axis('off')
        ax_1 = fig.add_subplot(int(np.ceil(len(dataset)/2)), 4, index_data*2+1)
        ax_1.set_title('time evolution', {"fontsize": label_size}, pad=12)
        if index_data % 2 == 0:
            ax_1.annotate(letter[index_data], xy=(-0.15, 0.9), xycoords='axes fraction', weight='bold', fontsize=label_size)
        else:
            ax_1.annotate(letter[index_data], xy=(-0.5, 0.9), xycoords='axes fraction', weight='bold', fontsize=label_size)
        ax_2 = fig.add_subplot(int(np.ceil(len(dataset)/2)), 4, index_data*2+2)
        ax_2.set_title('subject', {"fontsize": label_size}, pad=12)
        begin = 0
        for index, avalanche in enumerate(data_patient_nb):
            end = begin + len(avalanche)
            time_axes = ax_1.scatter(data[begin:end, 0], data[begin:end, 1], c=np.arange(len(avalanche)), s=0.8)
            ax_2.scatter(data[begin:end, 0], data[begin:end, 1], c=list(cnames.values())[index], s=0.8)
            begin = end
        if index_data % 2 == 0:
            ax_1.yaxis.tick_right()
            divider = make_axes_locatable(ax_1)
            cax = divider.append_axes("left", size="5%", pad=0.2)
            colbar_time = fig.colorbar(time_axes, cax=cax)
            colbar_time.ax.yaxis.tick_left()
            colbar_time.set_ticks([0, len(data_patient_nb[-1]) - 1])
            colbar_time.set_ticklabels(['begin\nrecording', 'end\nrecording'])
            col_bars.append(colbar_time)
        if index_data % 2 == 1:
            divider = make_axes_locatable(ax_2)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            colbar = fig.colorbar(colorbar_patient, cax=cax)
            colbar.set_ticks([0, 5, 10, 15, 18])
            col_bars.append(colbar)
    plt.subplots_adjust(left=param[1], right=param[2], top=param[3], bottom=param[4], wspace=param[5], hspace=param[6])
    plt.savefig('figure/SP_'+str(10+index_data_set)+'_'+param[7]+'.png')
    # plt.show()
