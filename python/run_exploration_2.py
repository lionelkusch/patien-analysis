import os
import numpy as np
from sklearn.cluster import KMeans
import phate
import matplotlib.pyplot as plt
from functions.plot import plot_figure_2D, plot_figure_2D_patient, plot_figure_2D_patient_unique

# from select_data import get_data_selected_patient_1
# avalanches_bin, avalanches_sum, out, out_sum = get_data_selected_patient_1()
path_data = os.path.dirname(os.path.realpath(__file__)) + '/../data/'
avalanches_bin = np.load(path_data + '/avalanches_selected_patient.npy', allow_pickle=True)
avalanches_bin = np.concatenate(avalanches_bin)

# all data
knn_dist = 'cosine'
mds_dist = 'cosine'
knn_dist_name = knn_dist
gamma = -1.0
for n_components in range(4, 15):
    for n_pca in range(1, 15):
        file = "../projection_data/first_projection/all_subject_Y_phate_knn_dist_" + knn_dist_name \
               + "_mds_dist_" + mds_dist + "_nb_comp_" + str(n_components) + "_nb_pca_" + str(n_pca) \
               + "_gamma_" + str(gamma) + ".npy"
        print(file)
        if not os.path.exists(file):
            phate_operator = phate.PHATE(n_components=n_components, n_jobs=-2, decay=1.0, n_pca=n_pca,
                                         gamma=gamma, knn=5, knn_dist=knn_dist, mds_dist=mds_dist)
            Y_phate = phate_operator.fit_transform(avalanches_bin)
            np.save(file, Y_phate)


# plot result
for n_components in range(4, 15):
    for n_pca in range(1, 15):
        file = "../projection_data/first_projection/all_subject_Y_phate_knn_dist_" + knn_dist_name + "_mds_dist_" + mds_dist \
               + "_nb_comp_" + str(n_components) + "_nb_pca_" + str(n_pca) + "_gamma_" + str(gamma)
        data = np.load(file+ ".npy")
        plot_figure_2D(data, file, KMeans(n_clusters=n_pca + 1, random_state=123).fit_predict(data))
        plt.savefig(file + '_cluster.png')
        plot_figure_2D_patient(data, file, np.load(path_data + '/avalanches_selected_patient.npy', allow_pickle=True))
        plt.savefig(file + '_patient.png')
        plot_figure_2D_patient_unique(data, file, np.load(path_data + '/avalanches_selected_patient.npy', allow_pickle=True))
        plt.savefig(file + '_patient_unique.png')
