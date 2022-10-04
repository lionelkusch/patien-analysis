import numpy as np
import scipy.io as io
from select_data import get_data_selected_patient_1
datas = get_data_selected_patient_1()
print(datas[4], )
data = io.loadmat('/home/kusch/Documents/project/patient_analyse/projection_data/first_projection/all_subject_Y_phate_knn_dist_cosine_mds_dist_cosine_nb_comp_3_nb_pca_5_gamma_-1.0.mat')

histograms_patient = np.empty((len(subjects), nb_cluster))
transition = np.empty((len(subjects), nb_cluster, nb_cluster))
for index_patient, cluster_k in enumerate(cluster_patient_data):
    hist = np.histogram(cluster_k, bins=nb_cluster, range=(0, 12))
    histograms_patient[index_patient, :] = hist[0]
    next_step = cluster_k[1:]
    step = cluster_k[:-1]
    for i in range(nb_cluster):
        data = next_step[np.where(step == i)]
        percentage_trans = np.bincount(data) / len(data)
        if len(percentage_trans) < nb_cluster:
            percentage_trans = np.concatenate([percentage_trans, np.zeros(nb_cluster - percentage_trans.shape[0])])
        transition[index_patient, i, :] = percentage_trans
