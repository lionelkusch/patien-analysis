import os
import math
import numpy as np
import matplotlib.pyplot as plt

import scipy.io as io
import h5py

import phate
from sklearn.cluster import KMeans
import os

from functions.load_data import go_avalanches
from functions.plot import plot_figure_2D, plot_figure_2D_patient
from select_data import get_data_seelcted_patient_1

avalanches_bin, avalanches_sum, out, out_sum = get_data_seelcted_patient_1()

# # save one subject
# io.savemat('subject_'+str(subject)+'.mat',{'source_reconstruction_MEG':data,
#                                            'avalanches_binarize':out,
#                                            'avalanches_sum':out_sum})

# # all data
# knn_dist = 'cosine'
# mds_dist = 'cosine'
# knn_dist_name = knn_dist
# for n_components in range(2, 6):
#     for n_pca in [5]:
#         for gamma in [-1.0, 0.0, 1.0]:
#             if not os.path.exists("../projection_data/all_subject_Y_phate_knn_dist_" + knn_dist_name
#                                   + "_mds_dist_" + mds_dist + "_nb_comp_" + str(n_components) + "_nb_pca_" + str(n_pca)
#                                   + "_gamma_" + str(gamma) + ".npy"):
#                 phate_operator = phate.PHATE(n_components=n_components, n_jobs=-2, decay=1.0, n_pca=n_pca,
#                                              gamma=gamma, knn=5, knn_dist=knn_dist, mds_dist=mds_dist)
#                 Y_phate = phate_operator.fit_transform(avalanches_bin)
#                 np.save("all_subject_Y_phate_knn_dist_" + knn_dist_name + "_mds_dist_" + mds_dist
#                         + "_nb_comp_" + str(n_components) + "_nb_pca_" + str(n_pca) + "_gamma_" + str(gamma) + ".npy",
#                         Y_phate)

# for knn_dist_name, mds_dist, n_components, n_pca, gamma, nb_cluster in [('cosine', 'cosine', 2, 5, 1.0, 10),
#                                                                         ('cosine', 'cosine', 2, 5, 0.0, 10),
#                                                                         ('cosine', 'cosine', 2, 5, -1.0, 10)]:
#     file = "../projection_data/all_subject_Y_phate_knn_dist_" + knn_dist_name + "_mds_dist_" + mds_dist \
#            + "_nb_comp_" + str(n_components) + "_nb_pca_" + str(n_pca) + "_gamma_" + str(gamma)
#     data = np.load(file + '.npy')
#     plot_figure_2D(data, file, KMeans(n_clusters=nb_cluster, random_state=123).fit_predict(data))
#
# plt.show()

knn_dist = 'cosine'
mds_dist = 'cosine'
knn_dist_name = knn_dist
nb_cluster = 5
for n_components in range(2, 6):
    for n_pca in [5]:
        for gamma in [-1.0, 0.0, 1.0]:
            file = "../projection_data/all_subject_Y_phate_knn_dist_" + knn_dist_name + "_mds_dist_" + mds_dist \
                   + "_nb_comp_" + str(n_components) + "_nb_pca_" + str(n_pca) + "_gamma_" + str(gamma) + ".npy"
            data = np.load(file)
            plot_figure_2D(data, file, KMeans(n_clusters=nb_cluster, random_state=123).fit_predict(data))
            plot_figure_2D_patient(data, file, avalanches_sum)
plt.show()

# Cluster Result
cluster = KMeans(n_clusters=nb_cluster, random_state=123).fit_predict(data)
file = "../projection_data/all_subject_Y_phate_knn_dist_cosine_mds_dist_cosine_nb_comp_2_nb_pca_5_gamma_-1.0.npy"
save = False
plot = True
cluster_patient_data = []
begin = 0
for avalanche in avalanches_bin:
    end = begin + len(avalanche)
    cluster_patient_data.append(cluster[begin:end])
    begin = end
cluster_patient = np.empty((len(avalanches_bin), nb_cluster))
transition = np.empty((len(subjects), nb_cluster, nb_cluster))
histograms_patient = np.empty((len(subjects), nb_cluster))
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
if plot:
    for index_patient, cluster_k in enumerate(cluster_patient_data):
        fig, axs = plt.subplots(1, 2)
        axs[0].hist(cluster_k, bins=nb_cluster, range=(0, 12))
        im = axs[1].imshow(transition[index_patient], vmin=0.0)
        fig.colorbar(im)
if save:
    data = np.load(file)
    io.savemat('../projection_data/cluster_18_patients.mat',
               {'avalanches_binarize': np.concatenate(avalanches_bin),
                'cluster_index': KMeans(n_clusters=nb_cluster, random_state=123).fit_predict(data),
                'PHATE_position': data,
                'transition_matrix': transition,
                'histogram': histograms_patient})
plt.show()