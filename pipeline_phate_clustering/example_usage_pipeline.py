import os
import h5py
import numpy as np
from pipeline_phate_clustering.pipeline import pipeline

# Preparation data for the pipeline
path_data = os.path.dirname(os.path.realpath(__file__)) + '/../data/'
f = h5py.File(path_data + 'serie_Melbourne.mat', 'r')
struArray = f['D']
data = {}
Nsubs = 44
nregions = 90
for i in range(Nsubs):
    data['%d' % i] = np.swapaxes(f[struArray[i, 0]][:nregions, :], 0, 1)

# Data format : { 'patient_name' : time_series of source reconstruction array[time, regions] , fixed number of regions}

# remove suject 11,15,20
selected_subjects = ['43', '39', '38', '35', '34', '29', '26', '21', '20', '19', '18', '17', '15', '13', '9', '8', '6', '5']
path_saving = "/home/kusch/Documents/project/patient_analyse/paper/result/default/"

pipeline(path_saving, data, selected_subjects,
             avalanches_threshold=3, avalanches_direction=0, avalanches_binsize=1,
             PHATE_n_pca=5, PHATE_knn=5, PHATE_decay=1.0, PHATE_knn_dist='cosine',
             PHATE_gamma=-1.0, PHATE_mds_dist='cosine', PHATE_n_components=3, PHATE_n_jobs=8,
             kmeans_nb_cluster=5, kmeans_seed=123, update=False, save_for_matlab=True, plot=True, plot_save=True)