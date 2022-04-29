import os
import numpy as np
from sklearn.cluster import KMeans
import phate
import matplotlib.pyplot as plt
from functions.plot import plot_figure_2D, plot_figure_2D_patient, plot_figure_2D_patient_unique, plot_figure_2D_patient_unique_time

from select_data import get_data_all_patient
avalanches_bin, avalanches_sum = get_data_all_patient()
path_data = os.path.dirname(os.path.realpath(__file__)) + '/../data/'

# all data
knn_dist = 'cosine'
mds_dist = 'cosine'
knn_dist_name = knn_dist
gamma = -1.0
for n_components in range(2, 5):
    for n_pca in range(5, 10):
        file = "../projection_data/all_patient/all_subject_Y_phate_knn_dist_" + knn_dist_name \
               + "_mds_dist_" + mds_dist + "_nb_comp_" + str(n_components) + "_nb_pca_" + str(n_pca) \
               + "_gamma_" + str(gamma) + ".npy"
        print(file)
        if not os.path.exists(file):
            phate_operator = phate.PHATE(n_components=n_components, n_jobs=-2, decay=1.0, n_pca=n_pca,
                                         gamma=gamma, knn=5, knn_dist=knn_dist, mds_dist=mds_dist)
            Y_phate = phate_operator.fit_transform(np.concatenate(avalanches_bin))
            np.save(file, Y_phate)

# plot result
for n_components in range(2, 5):
    for n_pca in range(5, 10):
        file = "../projection_data/all_patient/all_subject_Y_phate_knn_dist_" + knn_dist_name + "_mds_dist_" + mds_dist \
               + "_nb_comp_" + str(n_components) + "_nb_pca_" + str(n_pca) + "_gamma_" + str(gamma)
        data = np.load(file + ".npy")
        plot_figure_2D(data, file, KMeans(n_clusters=n_pca + 1, random_state=123).fit_predict(data))
        plt.savefig(file + '_cluster.png')
        plot_figure_2D_patient(data, file, avalanches_bin)
        plt.savefig(file + '_patient.png')
        plot_figure_2D_patient_unique(data, file, avalanches_bin)
        plt.savefig(file + '_patient_unique.png')
        plot_figure_2D_patient_unique_time(data, file, avalanches_bin)
        plt.savefig(file + '_patient_unique_time.png')
        plt.close('all')


avalanches_bin, avalanches_sum = get_data_all_patient(nregions=116)
path_data = os.path.dirname(os.path.realpath(__file__)) + '/../data/'

# all data
for n_components in range(2, 5):
    for n_pca in range(5, 10):
        file = "../projection_data/all_patient_region/all_subject_Y_phate_knn_dist_" + knn_dist_name \
               + "_mds_dist_" + mds_dist + "_nb_comp_" + str(n_components) + "_nb_pca_" + str(n_pca) \
               + "_gamma_" + str(gamma) + ".npy"
        print(file)
        if not os.path.exists(file):
            phate_operator = phate.PHATE(n_components=n_components, n_jobs=-2, decay=1.0, n_pca=n_pca,
                                         gamma=gamma, knn=5, knn_dist=knn_dist, mds_dist=mds_dist)
            Y_phate = phate_operator.fit_transform(np.concatenate(avalanches_bin))
            np.save(file, Y_phate)


# plot result
for n_components in range(2, 5):
    for n_pca in range(5, 10):
        file = "../projection_data/all_patient_region/all_subject_Y_phate_knn_dist_" + knn_dist_name + "_mds_dist_" + mds_dist \
               + "_nb_comp_" + str(n_components) + "_nb_pca_" + str(n_pca) + "_gamma_" + str(gamma)
        data = np.load(file + ".npy")
        plot_figure_2D(data, file, KMeans(n_clusters=n_pca + 1, random_state=123).fit_predict(data))
        plt.savefig(file + '_cluster.png')
        plot_figure_2D_patient(data, file, avalanches_bin)
        plt.savefig(file + '_patient.png')
        plot_figure_2D_patient_unique(data, file, avalanches_bin)
        plt.savefig(file + '_patient_unique.png')
        plot_figure_2D_patient_unique_time(data, file, avalanches_bin)
        plt.savefig(file + '_patient_unique_time.png')
        plt.close('all')
