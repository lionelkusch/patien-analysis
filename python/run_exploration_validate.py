import os
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import phate
import matplotlib.pyplot as plt
from functions.plot import plot_figure_2D, plot_figure_2D_patient, plot_figure_2D_patient_unique, plot_figure_3D, plot_figure_2D_patient_unique_time
import scipy.io as io

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
parameters = [(3,6), (3,5), (2,5)]

# plot result
for n_components, n_pca in parameters:
        file = "../projection_data/first_projection/all_subject_Y_phate_knn_dist_" + knn_dist_name + "_mds_dist_" + mds_dist \
               + "_nb_comp_" + str(n_components) + "_nb_pca_" + str(n_pca) + "_gamma_" + str(gamma)
        data = np.load(file + ".npy")
        # plot_figure_2D(data, file, KMeans(n_clusters=10, random_state=123).fit_predict(data))
        # # plot_figure_2D(data, file, DBSCAN(eps=0.0005, min_samples=100).fit_predict(data))
        # plt.show()
        # plt.savefig(file + '_cluster.png')
        # avalanches = np.load(path_data + '/avalanches_selected_patient.npy', allow_pickle=True)
        # plot_figure_2D_patient(data, file, avalanches)
        # plt.savefig(file + '_patient.png')
        # plot_figure_2D_patient_unique(data, file, avalanches)
        # plt.savefig(file + '_patient_unique.png')
        # plot_figure_2D_patient_unique_time(data, file, avalanches)
        # plt.savefig(file + '_patient_unique_time.png')
        # plt.close('all')
        io.savemat('../projection_data/'+file+'.mat',
               {'avalanches_binarize':avalanches_bin,
                'cluster_index':KMeans(n_clusters=n_pca, random_state=123).fit_predict(data),
                'PHATE_position':data,
                # 'transition_matrix':transition,
                # 'histogram':histograms_patient
                })
