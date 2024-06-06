import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from pipeline_phate_clustering.functions_helper.load_data import go_avalanches
from pipeline_phate_clustering.functions_helper.plot import plot_figure_2D, plot_figure_2D_patient, \
    plot_figure_2D_patient_unique, plot_figure_2D_patient_unique_time, plot_figure_2D_3D
import h5py
import scipy.io as io

# Preparation data
path_data = os.path.dirname(os.path.realpath(__file__)) + '/../../data/'
f = h5py.File(path_data + 'serie_Melbourne.mat', 'r')
struArray = f['D']
patients_data = {}
Nsubs = 44
nregions = 90
for i in range(Nsubs):
    patients_data['%d' % i] = np.swapaxes(f[struArray[i, 0]][:nregions, :], 0, 1)
selected_subjects = ['43', '39', '38', '35', '34', '29', '26', '21', '20', '19', '18', '17', '15', '13', '9', '8', '6',
                     '5']
# compute the avalanches for each patient
avalanches_threshold=3
avalanches_direction=0
avalanches_binsize=1
avalanches_bin = []
subjects_index = []
nb_regions = patients_data[selected_subjects[0]].shape[1]
data = []
for subject in selected_subjects:
    Avalanches_human = go_avalanches(patients_data[subject], thre=avalanches_threshold,
                                     direc=avalanches_direction, binsize=avalanches_binsize)
    avalanches = []
    for kk1 in range(len(Avalanches_human['ranges'])):
                begin = Avalanches_human['ranges'][kk1][0]
                end = Avalanches_human['ranges'][kk1][1]
                avalanches.append(Avalanches_human['Zbin'][begin:end, :])
    data.append(np.concatenate(avalanches))
pca_choice = 15

pca = PCA(n_components=90)
pca.fit(np.concatenate(data))
print(pca.explained_variance_ratio_)

cumulative = []
count = 0
for i in range(90):
    count += pca.explained_variance_ratio_[i]
    cumulative.append(count)
plt.figure()
plt.plot(np.arange(1, 91, 1), pca.explained_variance_ratio_, alpha=0.5)
plt.plot(np.arange(1, 91, 1), pca.explained_variance_ratio_, 'x')
plt.vlines(pca_choice, ymin=0.0, ymax=pca.explained_variance_ratio_[pca_choice-1], color='r', alpha=0.5)
plt.hlines(pca.explained_variance_ratio_[pca_choice-1], xmin=1, xmax=pca_choice, color='r', alpha=0.5)
plt.savefig(path_data+'/../paper/result/PCA/avalanches/pca_info.svg')
plt.figure()
plt.plot(np.arange(1, 91, 1), pca.explained_variance_ratio_, alpha=0.5)
plt.plot(np.arange(1, 91, 1), pca.explained_variance_ratio_, 'x')
plt.vlines(pca_choice, ymin=0.0, ymax=pca.explained_variance_ratio_[pca_choice-1], color='r', alpha=0.5)
plt.hlines(pca.explained_variance_ratio_[pca_choice-1], xmin=1, xmax=pca_choice, color='r', alpha=0.5)
plt.yscale("log")
plt.savefig(path_data+'/../paper/result/PCA/avalanches/pca_info_log.svg')
plt.figure()
plt.plot(np.arange(1, 91, 1), cumulative, alpha=0.5)
plt.plot(np.arange(1, 91, 1), cumulative, 'x')
plt.vlines(pca_choice, ymin=0.0, ymax=cumulative[pca_choice-1], color='r', alpha=0.5)
plt.hlines(cumulative[pca_choice-1], xmin=1, xmax=pca_choice, color='r', alpha=0.5)
plt.savefig(path_data+'/../paper/result/PCA/avalanches/pca_cumulative.svg')
# plt.show()

PCA_fit = PCA(n_components=pca_choice).fit_transform(np.concatenate(data))
cluster = KMeans(n_clusters=7, random_state=123).fit_predict(PCA_fit)
clusters_vectors = []
for i in range(7):
    index = np.where(cluster == i)
    clusters_vectors.append(np.mean(np.concatenate(data)[index], axis=0))
io.savemat(path_data + '/../paper/result/PCA/avalanches/vector_cluster.mat',
           {'cluster_vector':np.concatenate([clusters_vectors])}
           )
plot_figure_2D(PCA_fit, '', cluster)
plt.savefig(path_data+'/../paper/result/PCA/avalanches/PCA_cluster_in_2D.png')
plot_figure_2D_patient(PCA_fit, '', data)
plt.savefig(path_data+'/../paper/result/PCA/avalanches/PCA_cluster_for_patient.png')
plot_figure_2D_patient_unique(PCA_fit, '', data)
plt.savefig(path_data+'/../paper/result/PCA/avalanches/PCA_cluster_for_patient_unique.png')
plot_figure_2D_patient_unique_time(PCA_fit, '', data)
plt.savefig(path_data+'/../paper/result/PCA/avalanches/PCA_cluster_for_patient_time.png')
plot_figure_2D_3D(PCA_fit, '', cluster)
plt.savefig(path_data+'/../paper/result/PCA/avalanches/PCA_cluster_3D.png')
plt.close('all')

# plt.show()
