import os
import numpy as np
import json
import phate
from sklearn.cluster import KMeans
from scipy import stats
import matplotlib.pyplot as plt

from pipeline_phate_clustering.functions_helper.load_data import go_avalanches
from pipeline_phate_clustering.functions_helper.plot import plot_figure_2D, plot_figure_2D_patient, \
    plot_figure_2D_patient_unique, plot_figure_2D_patient_unique_time, plot_figure_2D_3D


def pipeline_no_avalanche(path_saving, patients_data, selected_subjects,
                          normalized_data = False,
                          PHATE_n_pca=5, PHATE_knn=5, PHATE_decay=1.0, PHATE_knn_dist='cosine',
             PHATE_gamma=-1.0, PHATE_mds_dist='cosine', PHATE_n_components=3, PHATE_n_jobs=-1,
             kmeans_nb_cluster=5, kmeans_seed=123, save_for_matlab=False, plot=False, plot_save=True,
             update_avalanches=False, update_Phate=False, update_transition=False
             ):
    """
    Pipeline for extracting unsupervised clustering from resting state MEG data on raw data or normalized data
    The Pipeline is composed on 3 steps:
        1) use PATHE for dimension reduction of the data
        2) Cluster the output using Kmeans

    :param path_saving: path for saving data
    :param patients_data: patient data (dictionary of 'patient name':[time, region])
    :param selected_subjects: selection of patient for the pipeline
    :param PHATE_n_pca: the number of component used for the PCA
    :param PHATE_knn: the number of neighboor for the k'neighboor algorithms
    :param PHATE_decay: the decay of the connection with other nodes
    :param PHATE_knn_dist: the type of distance use for the k'neighboor algorithms
    :param PHATE_gamma: the type of distance used diffusion matrices
    :param PHATE_mds_dist: the distance used for reduction dimension using mds algorithms
    :param PHATE_n_components: number of dimension of reduce representation
    :param PHATE_n_jobs: number of CPU used for running the jobs
    :param kmeans_nb_cluster: number of cluster
    :param kmeans_seed: seed for the starting points
    :param update_avalanches: update files from avalanches
    :param update_Phate: update the file from Phate
    :param update_transition: update the file from transition
    :param save_for_matlab: save the data for matlab
    :param plot: plot the result
    :param plot_save: save the plot in figure folder
    :return:
    """
    parameters = {
        'normalized': normalized_data,
        'selected_subjects': selected_subjects,
        'PHATE_n_pca': PHATE_n_pca,
        'PHATE_knn': PHATE_knn,
        'PHATE_decay': PHATE_decay,
        'PHATE_knn_dist': PHATE_knn_dist,
        'PHATE_gamma': PHATE_gamma,
        'PHATE_mds_dist': PHATE_mds_dist,
        'PHATE_n_components': PHATE_n_components,
        'PHATE_n_jobs': PHATE_n_jobs,
        'kmeans_nb_cluster': kmeans_nb_cluster,
        'kmeans_seed': kmeans_seed
    }
    update_Phate = update_avalanches or update_Phate
    update_transition = update_avalanches or update_Phate or update_transition

    with open(path_saving + '/parameters.json', 'w+') as f:
        json.dump(parameters, f)

    if plot and plot_save and not os.path.exists(path_saving + '/figure/'):
        os.mkdir(path_saving + '/figure/')

    # get data
    data = []
    for subject in selected_subjects:
        if normalized_data:
            data.append(stats.zscore(patients_data[subject]))
        else:
            data.append(patients_data[subject])
    data = np.array(data)
    # all data
    if update_Phate or not os.path.exists(path_saving + "/Phate.npy"):
        phate_operator = phate.PHATE(n_components=PHATE_n_components, n_jobs=PHATE_n_jobs, decay=PHATE_decay,
                                     n_pca=PHATE_n_pca, verbose=10,
                                     gamma=PHATE_gamma, knn=PHATE_knn, knn_dist=PHATE_knn_dist, mds_dist=PHATE_mds_dist)
        Y_phate = phate_operator.fit_transform(np.concatenate(data))
        np.save(path_saving + "/Phate.npy", Y_phate)
    else:
        Y_phate = np.load(path_saving + "/Phate.npy")

    # Cluster Result
    cluster = KMeans(n_clusters=kmeans_nb_cluster, random_state=kmeans_seed).fit_predict(Y_phate)

    if plot:
        plot_figure_2D(Y_phate, '', cluster)
        if plot_save:
            plt.savefig(path_saving + '/figure/cluster_in_2D.png');
            plt.close('all')
        plot_figure_2D_patient(Y_phate, '', data)
        if plot_save:
            plt.savefig(path_saving + '/figure/cluster_for_patient.png');
            plt.close('all')
        plot_figure_2D_patient_unique(Y_phate, '', data)
        if plot_save:
            plt.savefig(path_saving + '/figure/cluster_for_patient_unique.png');
            plt.close('all')
        plot_figure_2D_patient_unique_time(Y_phate, '', data)
        if plot_save:
            plt.savefig(path_saving + '/figure/cluster_for_patient_time.png');
            plt.close('all')
        plot_figure_2D_3D(Y_phate, '', cluster)
        if plot_save:
            plt.savefig(path_saving + '/figure/cluster_3D.png');
            plt.close('all')

    # compute transition matrix
    if update_transition or not os.path.exists(path_saving + "/transition.npy"):
        cluster_patient_data = []
        begin = 0
        for avalanche in data:
            end = begin + len(avalanche)
            cluster_patient_data.append(cluster[begin:end])
            begin = end
        histograms_region = []
        for j in range(kmeans_nb_cluster):
            histograms_region.append(np.sum(np.concatenate(data)[np.where(cluster == j)], axis=0))

        transition = np.empty((len(selected_subjects), kmeans_nb_cluster, kmeans_nb_cluster))
        histograms_patient = np.empty((len(selected_subjects), kmeans_nb_cluster))
        for index_patient, cluster_k in enumerate(cluster_patient_data):
            hist = np.histogram(cluster_k, bins=kmeans_nb_cluster, range=(0, kmeans_nb_cluster))
            histograms_patient[index_patient, :] = hist[0]
            next_step = cluster_k[1:]
            step = cluster_k[:-1]
            for i in range(kmeans_nb_cluster):
                data = next_step[np.where(step == i)]
                percentage_trans = np.bincount(data)
                if len(percentage_trans) < kmeans_nb_cluster:
                    percentage_trans = np.concatenate(
                        [percentage_trans, np.zeros(kmeans_nb_cluster - percentage_trans.shape[0])])
                transition[index_patient, i, :] = percentage_trans / len(data)
        transition_all = np.empty((kmeans_nb_cluster, kmeans_nb_cluster))
        cluster_all = np.concatenate(cluster_patient_data)
        hist_all = np.histogram(cluster_all, bins=kmeans_nb_cluster, range=(0, kmeans_nb_cluster))[0]
        next_step = cluster_all[1:]
        step = cluster_all[:-1]
        for i in range(kmeans_nb_cluster):
            data = next_step[np.where(step == i)]
            percentage_trans = np.bincount(data)
            if len(percentage_trans) < kmeans_nb_cluster:
                percentage_trans = np.concatenate(
                    [percentage_trans, np.zeros(kmeans_nb_cluster - percentage_trans.shape[0])])
            transition_all[i, :] = percentage_trans / len(data)

        np.save(path_saving + "/transition_all.npy", transition_all)
        transition = np.array(transition)
        np.save(path_saving + "/transition.npy", transition)
        histograms_patient = np.array(histograms_patient)
        np.save(path_saving + "/histograms_patient.npy", histograms_patient)
        histograms_region = np.array(histograms_region)
        np.save(path_saving + "/histograms_region.npy", histograms_region)
        cluster_patient_data = np.array(cluster_patient_data, dtype=object)
        np.save(path_saving + "/cluster_patient_data.npy", cluster_patient_data)
    else:
        transition = np.load(path_saving + "/transition.npy")
        histograms_patient = np.load(path_saving + "/histograms_patient.npy")
        histograms_region = np.load(path_saving + "/histograms_region.npy")
        cluster_patient_data = np.load(path_saving + "/cluster_patient_data.npy", allow_pickle=True)

    if plot:
        fig, axs = plt.subplots(1, 1, figsize=(20, 10))
        plt.imshow(histograms_region / histograms_region.max(axis=0).reshape(1, len(data[0][0])))
        if plot_save:
            plt.savefig(path_saving + '/figure/vector_cluster.pdf');
            plt.close('all')

        for index_patient, cluster_k in enumerate(cluster_patient_data):
            fig, axs = plt.subplots(1, 2, figsize=(20, 10))
            fig.suptitle('patient :' + str(index_patient))
            axs[0].hist(cluster_k, bins=kmeans_nb_cluster, range=(0, kmeans_nb_cluster), linewidth=0.5,
                        edgecolor="black")
            im_1 = axs[1].imshow(transition[index_patient])
            for (j, i), label in np.ndenumerate(transition[index_patient]):
                axs[1].text(i, j, np.around(label, 4), ha='center', va='center')
            axs[1].autoscale(False)
            fig.colorbar(im_1, ax=axs[1])
            if plot_save:
                plt.savefig(path_saving + '/figure/' + str(index_patient) + '.pdf');
                plt.close('all')

        nb_x = int(np.sqrt(len(selected_subjects))) + 1
        nb_y = int(len(selected_subjects) / np.sqrt(len(selected_subjects)))
        fig, axs = plt.subplots(nb_x, nb_y, figsize=(10, 20))
        for index_patient, cluster_k in enumerate(cluster_patient_data):
            im = axs[int(index_patient % nb_x), int(index_patient / nb_x)].imshow(transition[index_patient])
            axs[int(index_patient % nb_x), int(index_patient / nb_x)].set_title('patient : ' + str(index_patient))
            axs[int(index_patient % nb_x), int(index_patient / nb_x)].autoscale(False)
            fig.colorbar(im, ax=axs[int(index_patient % nb_x), int(index_patient / nb_x)])
        for index_no_patient in range(len(cluster_patient_data), nb_x * nb_y):
            axs[int(index_no_patient % nb_x), int(index_no_patient / nb_x)].imshow(
                np.ones_like(transition[0]) * np.NAN)
            axs[int(index_no_patient % nb_x), int(index_no_patient / nb_x)].autoscale(False)
        plt.subplots_adjust(left=0.0, right=0.97, wspace=0.3, top=0.94, bottom=0.03, hspace=0.3)
        if plot_save:
            plt.savefig(path_saving + '/figure/transition.pdf');
            plt.close('all')

        fig, axs = plt.subplots(nb_x, nb_y, figsize=(10, 20))
        for index_patient, cluster_k in enumerate(cluster_patient_data):
            transition_patient = transition[index_patient]
            np.fill_diagonal(transition_patient, 0.)
            im = axs[int(index_patient % nb_x), int(index_patient / nb_x)].imshow(transition_patient)
            axs[int(index_patient % nb_x), int(index_patient / nb_x)].set_title('patient : ' + str(index_patient))
            axs[int(index_patient % nb_x), int(index_patient / nb_x)].autoscale(False)
            fig.colorbar(im, ax=axs[int(index_patient % nb_x), int(index_patient / nb_x)])
        for index_no_patient in range(len(cluster_patient_data), nb_x * nb_y):
            axs[int(index_no_patient % nb_x), int(index_no_patient / nb_x)].imshow(
                np.ones_like(transition[0]) * np.NAN)
            axs[int(index_no_patient % nb_x), int(index_no_patient / nb_x)].autoscale(False)
        plt.subplots_adjust(left=0.0, right=0.97, wspace=0.3, top=0.94, bottom=0.03, hspace=0.3)
        if plot_save:
            plt.savefig(path_saving + '/figure/transition_no_diag.pdf');
            plt.close('all')

    if save_for_matlab:
        import scipy.io as io
        io.savemat(path_saving + '/data.mat',
                   {'source_reconstruction_MEG': patients_data,
                    'avalanches_binarize': data,
                    'cluster_index': KMeans(n_clusters=kmeans_nb_cluster,
                                            random_state=kmeans_seed).fit_predict(Y_phate),
                    'PHATE_position': Y_phate,
                    'transition_matrix': transition,
                    'histogram': histograms_patient})
    if plot and not plot_save:
        plt.show()

if __name__=='__main__':
    import os
    import h5py
    import numpy as np

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

    path_saving = "/home/kusch/Documents/project/patient_analyse/paper/result/no_avalanche/data_normalized_euclidiean/"
    pipeline_no_avalanche(path_saving, data, selected_subjects,
                          normalized_data=True,
                          PHATE_n_pca=6, PHATE_knn=10, PHATE_decay=100.0, PHATE_knn_dist='euclidean',
                          PHATE_gamma=-1.0, PHATE_mds_dist='euclidean', PHATE_n_components=3, PHATE_n_jobs=8,
                          kmeans_nb_cluster=7, kmeans_seed=123,
                          update_avalanches=True, update_Phate=True, update_transition=True,
                          save_for_matlab=True, plot=True, plot_save=True)

    path_saving = "/home/kusch/Documents/project/patient_analyse/paper/result/no_avalanche/data_normalized_euclidean_2/"
    pipeline_no_avalanche(path_saving, data, selected_subjects,
                          normalized_data=True,
                          PHATE_n_pca=6, PHATE_knn=50, PHATE_decay=35.0, PHATE_knn_dist='euclidean',
                          PHATE_gamma=-1.0, PHATE_mds_dist='euclidean', PHATE_n_components=3, PHATE_n_jobs=8,
                          kmeans_nb_cluster=7, kmeans_seed=123,
                          update_avalanches=True, update_Phate=True, update_transition=True,
                          save_for_matlab=True, plot=True, plot_save=True)

    path_saving = "/home/kusch/Documents/project/patient_analyse/paper/result/no_avalanche/data_normalized_euclidean_cosine/"
    pipeline_no_avalanche(path_saving, data, selected_subjects,
                          normalized_data=True,
                          PHATE_n_pca=6, PHATE_knn=50, PHATE_decay=35.0, PHATE_knn_dist='euclidean',
                          PHATE_gamma=-1.0, PHATE_mds_dist='cosine', PHATE_n_components=3, PHATE_n_jobs=8,
                          kmeans_nb_cluster=7, kmeans_seed=123,
                          update_avalanches=True, update_Phate=True, update_transition=True,
                          save_for_matlab=True, plot=True, plot_save=True)

    path_saving = "/home/kusch/Documents/project/patient_analyse/paper/result/no_avalanche/data_euclidean/"
    pipeline_no_avalanche(path_saving, data, selected_subjects,
                          normalized_data=False,
                          PHATE_n_pca=6, PHATE_knn=10, PHATE_decay=1e16, PHATE_knn_dist='euclidean',
                          PHATE_gamma=-1.0, PHATE_mds_dist='euclidean', PHATE_n_components=3, PHATE_n_jobs=8,
                          kmeans_nb_cluster=7, kmeans_seed=123,
                          update_avalanches=True, update_Phate=True, update_transition=True,
                          save_for_matlab=True, plot=True, plot_save=True)


