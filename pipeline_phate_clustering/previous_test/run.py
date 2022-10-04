import os
import math
import numpy as np
import matplotlib.pyplot as plt

import scipy.io as io
import scipy
import h5py

import phate
from sklearn.cluster import KMeans
import os

from pipeline_phate_clustering.functions_helper.load_data import go_avalanches
from pipeline_phate_clustering.functions_helper.plot import plot_figure_2D, plot_figure_2D_patient, plot_figure_2D_patient_unique, plot_figure_2D_patient_unique_time, plot_figure_2D_3D
from select_data import get_data_selected_patient_1

avalanches_bin, avalanches_sum, out, out_sum = get_data_selected_patient_1()

subjects = ['43', '39', '38', '35', '34', '29', '26', '21', '20', '19', '18', '17', '15', '13', '9', '8', '6', '5']
plt.figure()
plt.bar(np.arange(0, 90, 1), np.sum(np.concatenate(avalanches_bin), axis=0))
plt.ylabel('number of time region\nis part of avalanches')
plt.xlabel('region')
plt.savefig('../projection_data/figure_patient/sum_regions.png')

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
for n_components in [3]:
    for n_pca in [5]:
        for gamma in [-1.0]:
            file = "../projection_data/first_projection/all_subject_Y_phate_knn_dist_" + knn_dist_name + "_mds_dist_" + mds_dist \
                   + "_nb_comp_" + str(n_components) + "_nb_pca_" + str(n_pca) + "_gamma_" + str(gamma) + ".npy"
            data = np.load(file)
            # plot_figure_2D(data, file, KMeans(n_clusters=nb_cluster, random_state=123).fit_predict(data))
            # plot_figure_2D_patient(data, file, avalanches_sum)
            plot_figure_2D_patient_unique(data, file, avalanches_bin)
            plot_figure_2D_patient_unique_time(data, file, avalanches_bin)
            plot_figure_2D_3D(data, file, KMeans(n_clusters=nb_cluster, random_state=123).fit_predict(data))
plt.show()

# Cluster Result
cluster = KMeans(n_clusters=nb_cluster, random_state=123).fit_predict(data)
file = "../projection_data/first_projection/all_subject_Y_phate_knn_dist_cosine_mds_dist_cosine_nb_comp_3_nb_pca_5_gamma_-1.0.npy"
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
transition_all = np.empty((len(subjects), nb_cluster, nb_cluster))
transition_zscore = np.empty((len(subjects), nb_cluster, nb_cluster))
transition_zscore_1 = np.empty((len(subjects), nb_cluster, nb_cluster))
histograms_patient = np.empty((len(subjects), nb_cluster))
for index_patient, cluster_k in enumerate(cluster_patient_data):
    hist = np.histogram(cluster_k, bins=nb_cluster, range=(0, 12))
    histograms_patient[index_patient, :] = hist[0]
    next_step = cluster_k[1:]
    step = cluster_k[:-1]
    for i in range(nb_cluster):
        data = next_step[np.where(step == i)]
        percentage_trans = np.bincount(data)
        if len(percentage_trans) < nb_cluster:
            percentage_trans = np.concatenate([percentage_trans, np.zeros(nb_cluster - percentage_trans.shape[0])])
        transition[index_patient, i, :] = percentage_trans / len(data)
        transition_all[index_patient, i, :] = percentage_trans / len(next_step)
    transition_zscore[index_patient, :, :] = (scipy.stats.zscore(transition_all[index_patient].ravel())).reshape(
        transition_all[index_patient, :, :].shape)
    tmp = np.copy(transition_all[index_patient])
    np.fill_diagonal(tmp, 0)
    transition_zscore_1[index_patient, :, :] = (scipy.stats.zscore(tmp.ravel())).reshape(tmp.shape)
if plot:
    max_transition = np.max(transition)
    min_transition = np.min(transition)
    max_transition_all = np.max(transition_all)
    min_transition_all = np.min(transition_all)
    max_transition_zscore = np.max(transition_zscore)
    min_transition_zscore = np.min(transition_zscore)
    max_transition_zscore_1 = np.max(transition_zscore)
    min_transition_zscore_1 = np.min(transition_zscore)
    print(max_transition, max_transition_all, max_transition_zscore, max_transition_zscore_1)
    print(min_transition, min_transition_all, min_transition_zscore, min_transition_zscore_1)
    for index_patient, cluster_k in enumerate(cluster_patient_data):
        fig, axs = plt.subplots(2, 3, figsize=(20, 10))
        fig.suptitle('patient :'+str(index_patient))
        plt.subplot(131).hist(cluster_k, bins=nb_cluster, range=(0, nb_cluster), linewidth=0.5, edgecolor="black")
        im_1 = axs[0, 1].imshow(transition[index_patient])
        for (j, i), label in np.ndenumerate(transition[index_patient]):
            axs[0, 1].text(i, j, np.around(label, 4), ha='center', va='center')
        axs[0, 1].autoscale(False)
        fig.colorbar(im_1, ax=axs[0, 1])
        im_1 = axs[0, 2].imshow(transition_all[index_patient])
        for (j, i), label in np.ndenumerate(transition_all[index_patient]):
            axs[0, 2].text(i, j, np.around(label, 4), ha='center', va='center')
        axs[0, 2].autoscale(False)
        fig.colorbar(im_1, ax=axs[0, 2])
        im_2 = axs[1, 1].imshow(transition_zscore[index_patient])
        for (j, i), label in np.ndenumerate(transition_zscore[index_patient]):
            axs[1, 1].text(i, j, np.around(label, 4), ha='center', va='center')
        axs[1, 1].autoscale(False)
        fig.colorbar(im_2, ax=axs[1, 1])
        im_2 = axs[1, 2].imshow(transition_zscore_1[index_patient])
        for (j, i), label in np.ndenumerate(transition_zscore_1[index_patient]):
            axs[1, 2].text(i, j, np.around(label, 4), ha='center', va='center')
        axs[1, 2].autoscale(False)
        fig.colorbar(im_2, ax=axs[1, 2])
        plt.savefig('../projection_data/figure_patient/'+str(index_patient)+'.pdf')
        plt.close('all')

    for name_file, title, data in [('tr', 'transition matrices in percentage by cluster', transition),
                        ('tr_all', 'transition matrices in percentage of total transition', transition),
                        ('zscore_all', 'zcore of percentage of total transition', transition_zscore),
                        ('zscore_all_d', 'zcore of percentage of total transition without the diagonal', transition_zscore_1)]:
        nb_x = 5
        nb_y = 4
        fig, axs = plt.subplots(nb_x, nb_y, figsize=(10, 20))
        fig.suptitle(title)
        for index_patient, cluster_k in enumerate(cluster_patient_data):
            # im = axs[int(index_patient % nb_x), int(index_patient / nb_y)].imshow(data[index_patient], vmin=np.min(data), vmax=np.max(data))
            print()
            im = axs[int(index_patient % nb_x), int(index_patient / nb_x)].imshow(data[index_patient])
            axs[int(index_patient % nb_x), int(index_patient / nb_x)].set_title('patient : ' + str(index_patient))
            axs[int(index_patient % nb_x), int(index_patient / nb_x)].autoscale(False)
            fig.colorbar(im, ax=axs[int(index_patient % nb_x), int(index_patient / nb_x)])
            # for (j, i), label in np.ndenumerate(data[index_patient]):
            #     axs[int(index_patient % nb_x), int(index_patient / nb_y)].text(i, j, np.around(label, 2), ha='center', va='center')
        for index_no_patient in range(len(cluster_patient_data), nb_x * nb_y):
            axs[int(index_no_patient % nb_x), int(index_no_patient / nb_x)].imshow(np.ones_like(data[0]) * np.NAN)
            axs[int(index_no_patient % nb_x), int(index_no_patient / nb_x)].autoscale(False)
        plt.subplots_adjust(left=0.0, right=1.0, wspace=0., top=0.94, bottom=0.03, hspace=0.3)
        plt.savefig('../projection_data/figure_patient/'+name_file+'.pdf')
        plt.close('all')

if save:
    data = np.load(file)
    io.savemat('../projection_data/test_2.mat',
               {'avalanches_binarize': np.concatenate(avalanches_bin),
                'cluster_index': KMeans(n_clusters=nb_cluster, random_state=123).fit_predict(data),
                'PHATE_position': data,
                'transition_matrix': transition,
                'histogram': histograms_patient})
plt.show()
