import os
import numpy as np
import phate
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from pipeline_phate_clustering.functions_helper.plot import plot_figure_2D, plot_figure_2D_patient, \
    plot_figure_2D_patient_unique, plot_figure_2D_patient_unique_time, plot_figure_2D_3D


def null_model_transition(path_saving, nb_randomize=100, seed=123, plot_save=False):
    """
    null model of transition
    :param path_saving: path of result of default paper
    :param nb_randomize: number of randomization
    :param plot_save: plot or not the result
    :param seed: seed of the random generator
    :return:
    """
    cluster_patient_data = np.load(path_saving + "/cluster_patient_data.npy", allow_pickle=True)
    rng = np.random.default_rng(seed=seed)
    nb_patient = len(cluster_patient_data)
    kmeans_nb_cluster = np.max(cluster_patient_data[0]) + 1
    for nb_rand in range(nb_randomize):
        # shuffle data
        for i in range(cluster_patient_data.shape[0]):
            rng.shuffle(cluster_patient_data[i])

        transition = np.empty((nb_patient, kmeans_nb_cluster, kmeans_nb_cluster))
        histograms_patient = np.empty((nb_patient, kmeans_nb_cluster))
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
        transition = np.array(transition)
        np.save(path_saving + "/transition" + str(nb_rand) + ".npy", transition)
        histograms_patient = np.array(histograms_patient)
        np.save(path_saving + "/histograms" + str(nb_rand) + ".npy", histograms_patient)
        cluster_patient_data = np.array(cluster_patient_data)
        np.save(path_saving + "/cluster_patient_data" + str(nb_rand) + ".npy", cluster_patient_data)

        if plot_save:
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
                plt.savefig(path_saving + '/figure/' + str(nb_rand) + '_' + str(index_patient) + '.pdf')
                plt.close('all')

            nb_x = int(np.sqrt(nb_patient)) + 1
            nb_y = int(nb_patient / np.sqrt(nb_patient))
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
            plt.savefig(path_saving + '/figure/transition' + str(nb_rand) + '.pdf')
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
            plt.savefig(path_saving + '/figure/transition_no_diag' + str(nb_rand) + '.pdf')
            plt.close('all')


def null_model_transition_all(path_saving, nb_randomize=100, seed=123, plot_save=False):
    """
    null model of transition with concatenation of patients
    :param path_saving: path of result of default paper
    :param nb_randomize: number of randomization
    :param plot_save: plot or not the result
    :param seed: seed of the random generator
    :return:
    """
    cluster_patient_data = np.load(path_saving + "/cluster_patient_data.npy", allow_pickle=True)
    rng = np.random.default_rng(seed=seed)
    cluster_patient_data = np.concatenate(cluster_patient_data)
    kmeans_nb_cluster = np.max(cluster_patient_data) + 1
    for nb_rand in range(nb_randomize):
        # shuffle data
        rng.shuffle(cluster_patient_data)

        transition = np.empty((kmeans_nb_cluster, kmeans_nb_cluster))
        next_step = cluster_patient_data[1:]
        step = cluster_patient_data[:-1]
        for i in range(kmeans_nb_cluster):
            data = next_step[np.where(step == i)]
            percentage_trans = np.bincount(data)
            if len(percentage_trans) < kmeans_nb_cluster:
                percentage_trans = np.concatenate(
                    [percentage_trans, np.zeros(kmeans_nb_cluster - percentage_trans.shape[0])])
            transition[i, :] = percentage_trans / len(data)
        transition = np.array(transition)
        np.save(path_saving + "/transition_all" + str(nb_rand) + ".npy", transition)

        if plot_save:
            fig = plt.figure(figsize=(20, 10))
            fig.suptitle('patient : all')
            ax = plt.subplot(121)
            ax.hist(cluster_patient_data, bins=kmeans_nb_cluster, range=(0, kmeans_nb_cluster), linewidth=0.5,
                    edgecolor="black")
            ax = plt.subplot(223)
            im_1 = ax.imshow(transition)
            for (j, i), label in np.ndenumerate(transition):
                ax.text(i, j, np.around(label, 4), ha='center', va='center')
            ax.autoscale(False)
            fig.colorbar(im_1, ax=ax)
            np.fill_diagonal(transition, 0.0)
            ax = plt.subplot(224)
            im_1 = ax.imshow(transition)
            for (j, i), label in np.ndenumerate(transition):
                ax.text(i, j, np.around(label, 4), ha='center', va='center')
            ax.autoscale(False)
            fig.colorbar(im_1, ax=ax)
            plt.savefig(path_saving + '/figure/transition_' + str(nb_rand) + '_all.pdf')
            plt.close('all')


def null_model_cluster_regions(path_saving, nb_randomize=100, plot_save=False, seed=123):
    """
    null model of cluster region
    :param path_saving: path of result of default paper
    :param nb_randomize: number of randomization
    :param plot_save: plot or not the result
    :param seed: seed of the random generator
    :return:
    """
    avalanches_bin = np.load(path_saving + '/avalanches.npy', allow_pickle=True)
    cluster_patient_data = np.load(path_saving + "/cluster_patient_data.npy", allow_pickle=True)
    rng = np.random.default_rng(seed=seed)
    kmeans_nb_cluster = np.max(cluster_patient_data[0]) + 1
    for nb_rand in range(nb_randomize):
        # shuffle data
        for i in range(cluster_patient_data.shape[0]):
            rng.shuffle(cluster_patient_data[i])
        cluster = np.concatenate(cluster_patient_data)
        histograms_region = []
        for j in range(kmeans_nb_cluster):
            histograms_region.append(np.sum(np.concatenate(avalanches_bin)[np.where(cluster == j)], axis=0))
        histograms_region = np.array(histograms_region)
        np.save(path_saving + "/histograms_region_" + str(nb_rand) + ".npy", histograms_region)
        plt.subplots(1, 1, figsize=(20, 10))
        plt.imshow(histograms_region / histograms_region.max(axis=0).reshape(1, len(avalanches_bin[0][0])))
        if plot_save:
            plt.savefig(path_saving + '/figure/vector_cluster' + str(nb_rand) + '.pdf')
            plt.close('all')


def null_model_data(path_saving, avalanches_path=None,
                    PHATE_n_pca=5, PHATE_knn=5, PHATE_decay=1.0, PHATE_knn_dist='cosine',
                    PHATE_gamma=-1.0, PHATE_mds_dist='cosine', PHATE_n_components=3, PHATE_n_jobs=-1,
                    kmeans_nb_cluster=7, kmeans_seed=123, plot=False, plot_save=True,
                    update_Phate=False, update_transition=False,
                    nb_randomize=100, seed=123
                    ):
    """
    Check the effect of shuffle the active regions in each avalanches pattern

    Pipeline for extracting unsupervised clustering from resting state MEG data
    The Pipeline is composed on 3 steps:
        1) compute the avalanches from time series
        2) use PATHE for dimension reduction of the data
        3) Cluster the output using Kmeans

    :param path_saving: path for saving data
    :param avalanches_path: path for already computed avalanches
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
    :param update_Phate: update the file from Phate
    :param update_transition: update the file from transition
    :param plot: plot the result
    :param plot_save: save the plot in figure folder
    :param nb_randomize: number of randomize test
    :param seed: seed for the randomisation, important for the reproducibility
    :return:
    """
    update_transition = update_Phate or update_transition

    if avalanches_path is not None:
        avalanches_bin = np.load(avalanches_path, allow_pickle=True)
    else:
        avalanches_bin = np.load(path_saving + '/avalanches.npy', allow_pickle=True)
    nb_regions = len(avalanches_bin[0][0])

    if not os.path.exists(path_saving + '/null_model/'):
        os.mkdir(path_saving + '/null_model/')
    path_saving = path_saving + '/null_model/'

    if plot and plot_save and not os.path.exists(path_saving + '/figure/'):
        os.mkdir(path_saving + '/figure/')

    rng = np.random.default_rng(seed=seed)
    for nb_rand in range(nb_randomize):
        for i in range(avalanches_bin.shape[0]):
            rng.shuffle(avalanches_bin[i], axis=1)

        # all data
        if update_Phate or not os.path.exists(path_saving + "/" + str(nb_rand) + "_Phate.npy"):
            phate_operator = phate.PHATE(n_components=PHATE_n_components, n_jobs=PHATE_n_jobs, decay=PHATE_decay,
                                         n_pca=PHATE_n_pca,
                                         gamma=PHATE_gamma, knn=PHATE_knn, knn_dist=PHATE_knn_dist,
                                         mds_dist=PHATE_mds_dist)
            Y_phate = phate_operator.fit_transform(np.concatenate(avalanches_bin))
            np.save(path_saving + "/" + str(nb_rand) + "_Phate.npy", Y_phate)
        else:
            Y_phate = np.load(path_saving + "/" + str(nb_rand) + "_Phate.npy")

        # Cluster Result
        cluster = KMeans(n_clusters=kmeans_nb_cluster, random_state=kmeans_seed).fit_predict(Y_phate)

        if plot:
            plot_figure_2D(Y_phate, '', cluster)
            if plot_save:
                plt.savefig(path_saving + '/figure/' + str(nb_rand) + 'cluster_in_2D.png')
                plt.close('all')
            plot_figure_2D_patient(Y_phate, '', avalanches_bin)
            if plot_save:
                plt.savefig(path_saving + '/figure/' + str(nb_rand) + 'cluster_for_patient.png')
                plt.close('all')
            plot_figure_2D_patient_unique(Y_phate, '', avalanches_bin)
            if plot_save:
                plt.savefig(path_saving + '/figure/' + str(nb_rand) + 'cluster_for_patient_unique.png')
                plt.close('all')
            plot_figure_2D_patient_unique_time(Y_phate, '', avalanches_bin)
            if plot_save:
                plt.savefig(path_saving + '/figure/' + str(nb_rand) + 'cluster_for_patient_time.png')
                plt.close('all')
            plot_figure_2D_3D(Y_phate, '', cluster)
            if plot_save:
                plt.savefig(path_saving + '/figure/' + str(nb_rand) + 'cluster_3D.png')
                plt.close('all')

        # compute transition matrix
        if update_transition or not os.path.exists(path_saving + "/" + str(nb_rand) + "_transition.npy"):
            cluster_patient_data = []
            begin = 0
            for avalanche in avalanches_bin:
                end = begin + len(avalanche)
                cluster_patient_data.append(cluster[begin:end])
                begin = end
            histograms_region = []
            for j in range(kmeans_nb_cluster):
                histograms_region.append(np.sum(np.concatenate(avalanches_bin)[np.where(cluster == j)], axis=0))

            transition = np.empty((len(avalanches_bin), kmeans_nb_cluster, kmeans_nb_cluster))
            histograms_patient = np.empty((len(avalanches_bin), kmeans_nb_cluster))
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
            transition = np.array(transition)
            np.save(path_saving + "/" + str(nb_rand) + "_transition.npy", transition)
            histograms_patient = np.array(histograms_patient)
            np.save(path_saving + "/" + str(nb_rand) + "_histograms_patient.npy", histograms_patient)
            histograms_region = np.array(histograms_region)
            np.save(path_saving + "/" + str(nb_rand) + "_histograms_region.npy", histograms_region)
            cluster_patient_data = np.array(cluster_patient_data)
            np.save(path_saving + "/" + str(nb_rand) + "_cluster_patient_data.npy", cluster_patient_data)
        else:
            transition = np.load(path_saving + "/" + str(nb_rand) + "_transition.npy")
            histograms_patient = np.load(path_saving + "/" + str(nb_rand) + "_histograms_patient.npy")
            histograms_region = np.load(path_saving + "/" + str(nb_rand) + "_histograms_region.npy")
            cluster_patient_data = np.load(path_saving + "/" + str(nb_rand) + "_cluster_patient_data.npy",
                                           allow_pickle=True)

        if plot:
            plt.subplots(1, 1, figsize=(20, 10))
            plt.imshow(histograms_region / histograms_region.max(axis=0).reshape(1, nb_regions))
            if plot_save:
                plt.savefig(path_saving + '/figure/' + str(nb_rand) + 'vector_cluster.pdf')
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
                    plt.savefig(path_saving + '/figure/' + str(nb_rand) + '_' + str(index_patient) + '.pdf')
                    plt.close('all')

            nb_x = int(np.sqrt(len(avalanches_bin))) + 1
            nb_y = int(len(avalanches_bin) / np.sqrt(len(avalanches_bin)))
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
                plt.savefig(path_saving + '/figure/' + str(nb_rand) + 'transition.pdf')
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
                plt.savefig(path_saving + '/figure/' + str(nb_rand) + 'transition_no_diag.pdf')
                plt.close('all')

        if plot and not plot_save:
            plt.show()


def sanitary_check_data(path_saving, avalanches_path=None,
                    PHATE_n_pca=5, PHATE_knn=5, PHATE_decay=1.0, PHATE_knn_dist='cosine',
                    PHATE_gamma=-1.0, PHATE_mds_dist='cosine', PHATE_n_components=3, PHATE_n_jobs=-1,
                    kmeans_nb_cluster=7, kmeans_seed=123, plot=False, plot_save=True,
                    update_Phate=False, update_transition=False,
                    nb_randomize=100, seed=123
                    ):
    """
    Check the effect of changing order of avalanches

    Pipeline for extracting unsupervised clustering from resting state MEG data
    The Pipeline is composed on 3 steps:
        1) compute the avalanches from time series
        2) use PATHE for dimension reduction of the data
        3) Cluster the output using Kmeans

    :param path_saving: path for saving data
    :param avalanches_path: path for already computed avalanches
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
    :param update_Phate: update the file from Phate
    :param update_transition: update the file from transition
    :param plot: plot the result
    :param plot_save: save the plot in figure folder
    :param nb_randomize: number of randomize test
    :param seed: seed for the randomisation, important for the reproducibility
    :return:
    """
    update_transition = update_Phate or update_transition

    if avalanches_path is not None:
        avalanches_bin = np.load(avalanches_path, allow_pickle=True)
    else:
        avalanches_bin = np.load(path_saving + '/avalanches.npy', allow_pickle=True)
    nb_regions = len(avalanches_bin[0][0])
    avalanches_bin_cont = np.concatenate(avalanches_bin)

    if not os.path.exists(path_saving + '/sanitary_check/'):
        os.mkdir(path_saving + '/sanitary_check/')
    path_saving = path_saving + '/sanitary_check/'

    if plot and plot_save and not os.path.exists(path_saving + '/figure/'):
        os.mkdir(path_saving + '/figure/')

    rng = np.random.default_rng(seed=seed)
    for nb_rand in range(nb_randomize):
        rng.shuffle(avalanches_bin_cont)

        # all data
        if update_Phate or not os.path.exists(path_saving + "/" + str(nb_rand) + "_Phate.npy"):
            phate_operator = phate.PHATE(n_components=PHATE_n_components, n_jobs=PHATE_n_jobs, decay=PHATE_decay,
                                         n_pca=PHATE_n_pca,
                                         gamma=PHATE_gamma, knn=PHATE_knn, knn_dist=PHATE_knn_dist,
                                         mds_dist=PHATE_mds_dist)
            Y_phate = phate_operator.fit_transform(avalanches_bin_cont)
            np.save(path_saving + "/" + str(nb_rand) + "_Phate.npy", Y_phate)
        else:
            Y_phate = np.load(path_saving + "/" + str(nb_rand) + "_Phate.npy")

        # Cluster Result
        cluster = KMeans(n_clusters=kmeans_nb_cluster, random_state=kmeans_seed).fit_predict(Y_phate)

        if plot:
            plot_figure_2D(Y_phate, '', cluster)
            if plot_save:
                plt.savefig(path_saving + '/figure/' + str(nb_rand) + 'cluster_in_2D.png')
                plt.close('all')
            plot_figure_2D_patient(Y_phate, '', avalanches_bin)
            if plot_save:
                plt.savefig(path_saving + '/figure/' + str(nb_rand) + 'cluster_for_patient.png')
                plt.close('all')
            plot_figure_2D_patient_unique(Y_phate, '', avalanches_bin)
            if plot_save:
                plt.savefig(path_saving + '/figure/' + str(nb_rand) + 'cluster_for_patient_unique.png')
                plt.close('all')
            plot_figure_2D_patient_unique_time(Y_phate, '', avalanches_bin)
            if plot_save:
                plt.savefig(path_saving + '/figure/' + str(nb_rand) + 'cluster_for_patient_time.png')
                plt.close('all')
            plot_figure_2D_3D(Y_phate, '', cluster)
            if plot_save:
                plt.savefig(path_saving + '/figure/' + str(nb_rand) + 'cluster_3D.png')
                plt.close('all')

        # compute transition matrix
        if update_transition or not os.path.exists(path_saving + "/" + str(nb_rand) + "_transition.npy"):
            cluster_patient_data = []
            begin = 0
            for avalanche in avalanches_bin:
                end = begin + len(avalanche)
                cluster_patient_data.append(cluster[begin:end])
                begin = end
            histograms_region = []
            for j in range(kmeans_nb_cluster):
                histograms_region.append(np.sum(avalanches_bin_cont[np.where(cluster == j)], axis=0))

            transition = np.empty((len(avalanches_bin), kmeans_nb_cluster, kmeans_nb_cluster))
            histograms_patient = np.empty((len(avalanches_bin), kmeans_nb_cluster))
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
            transition = np.array(transition)
            np.save(path_saving + "/" + str(nb_rand) + "_transition.npy", transition)
            histograms_patient = np.array(histograms_patient)
            np.save(path_saving + "/" + str(nb_rand) + "_histograms_patient.npy", histograms_patient)
            histograms_region = np.array(histograms_region)
            np.save(path_saving + "/" + str(nb_rand) + "_histograms_region.npy", histograms_region)
            cluster_patient_data = np.array(cluster_patient_data)
            np.save(path_saving + "/" + str(nb_rand) + "_cluster_patient_data.npy", cluster_patient_data)
        else:
            transition = np.load(path_saving + "/" + str(nb_rand) + "_transition.npy")
            histograms_patient = np.load(path_saving + "/" + str(nb_rand) + "_histograms_patient.npy")
            histograms_region = np.load(path_saving + "/" + str(nb_rand) + "_histograms_region.npy")
            cluster_patient_data = np.load(path_saving + "/" + str(nb_rand) + "_cluster_patient_data.npy",
                                           allow_pickle=True)

        if plot:
            plt.subplots(1, 1, figsize=(20, 10))
            plt.imshow(histograms_region / histograms_region.max(axis=0).reshape(1, nb_regions))
            if plot_save:
                plt.savefig(path_saving + '/figure/' + str(nb_rand) + 'vector_cluster.pdf')
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
                    plt.savefig(path_saving + '/figure/' + str(nb_rand) + '_' + str(index_patient) + '.pdf')
                    plt.close('all')

            nb_x = int(np.sqrt(len(avalanches_bin))) + 1
            nb_y = int(len(avalanches_bin) / np.sqrt(len(avalanches_bin)))
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
                plt.savefig(path_saving + '/figure/' + str(nb_rand) + 'transition.pdf')
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
                plt.savefig(path_saving + '/figure/' + str(nb_rand) + 'transition_no_diag.pdf')
                plt.close('all')

        if plot and not plot_save:
            plt.show()


if __name__ == '__main__':
    path_data = os.path.dirname(os.path.realpath(__file__)) + '/../../'
    null_model_cluster_regions(path_saving=path_data+"/paper/result/default/",
                               plot_save=True, nb_randomize=10000)
    null_model_transition(path_saving=path_data+"/paper/result/default/",
                          plot_save=True)
    null_model_transition_all(path_saving=path_data+"/paper/result/default/",
                              plot_save=True)
    null_model_data(path_saving=path_data+"/paper/result/default/", avalanches_path=None,
                    PHATE_n_pca=5, PHATE_knn=5, PHATE_decay=1.0, PHATE_knn_dist='cosine',
                    PHATE_gamma=-1.0, PHATE_mds_dist='cosine', PHATE_n_components=3, PHATE_n_jobs=1,
                    kmeans_nb_cluster=7, kmeans_seed=123, plot=True, plot_save=True,
                    update_Phate=False, update_transition=False,
                    nb_randomize=100, seed=123)
    sanitary_check_data(path_saving=path_data+"/paper/result/default/", avalanches_path=None,
                        PHATE_n_pca=5, PHATE_knn=5, PHATE_decay=1.0, PHATE_knn_dist='cosine',
                        PHATE_gamma=-1.0, PHATE_mds_dist='cosine', PHATE_n_components=3, PHATE_n_jobs=1,
                        kmeans_nb_cluster=7, kmeans_seed=123, plot=True, plot_save=True,
                        update_Phate=False, update_transition=False,
                        nb_randomize=100, seed=123)
