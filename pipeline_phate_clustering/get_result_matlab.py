import scipy.io as io
import numpy as np
import os
import h5py


# Preparation data for the pipeline
path_data = os.path.dirname(os.path.realpath(__file__)) + '/../data/'
f = h5py.File(path_data + 'serie_Melbourne.mat', 'r')
struArray = f['D']
patients_data = {}
Nsubs = 44
nregions = 90
for i in range(Nsubs):
    patients_data['%d' % i] = np.swapaxes(f[struArray[i, 0]][:nregions, :], 0, 1)

# data from pipeline
path_saving = "/home/kusch/Documents/project/patient_analyse/paper/result/default/"
avalanches_bin = np.load(path_saving + '/avalanches.npy', allow_pickle=True)
Y_phate = np.load(path_saving + "/Phate.npy")
transition = np.load(path_saving + "/transition.npy")
histograms_patient = np.load(path_saving + "/histograms_patient.npy")

# cluster_data
PHATE_knn = 7
PHATE_decay = 1.0
path_cluster = "/home/kusch/Documents/project/patient_analyse/paper/cluster_measure/sensibility_analysis/PHATE_KNN_"+str(PHATE_knn)+'_decay_'+str(PHATE_decay)+'/'
data_cluster = np.load(path_cluster + '/measure_cluster.npy', allow_pickle=True)

dic_data = {'source_reconstruction_MEG': patients_data,
    'avalanches_binarize': avalanches_bin,
    'PHATE_position': Y_phate,
    'transition_matrix': transition,
    'histogram': histograms_patient,
    'cluster':[[i[1], i[0]] for i in data_cluster]}


io.savemat(path_saving + '/data_with_cluster.mat', dic_data)
