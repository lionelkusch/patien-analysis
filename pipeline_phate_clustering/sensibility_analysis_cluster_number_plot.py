import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# from sklearn.mixture import GaussianMixture
# import scipy.cluster.hierarchy as hier; heirachical_graph=hier.dendrogram(hier.linkage(X, method='ward')) #method='ward' uses the Ward variance minimization algorithm

def generate_cluster_precision(path_data, path_saving, range_n_clusters, range_kmeans_seed, nrefs):
    """
    compute the elbow, the silhouette and the gap statistic measure for different number of cluster and seed
    :param path_data: path of the Phate data
    :param path_saving: path of folder for saving result
    :param range_n_clusters: array fo cluster to test
    :param range_kmeans_seed: array fo seed to test
    :param nrefs: number of random data for the gap statistic
    :return:
    """
    data = np.load(path_data + "/Phate.npy")
    result = []
    for n_clusters in range_n_clusters:
        for kmeans_seed in range_kmeans_seed:
            clusterer = KMeans(n_clusters=n_clusters, random_state=kmeans_seed)
            cluster_labels = clusterer.fit_predict(data)
            silhouette_avg = silhouette_score(data, cluster_labels)

            # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
            refDisps = []
            for i in range(nrefs):
                # Create new random reference set
                randomReference = np.random.random_sample(size=data.shape)
                # Fit to it
                km = KMeans(n_clusters=n_clusters, random_state=kmeans_seed).fit(randomReference).inertia_
                refDisps.append(km)

            result.append((cluster_labels,  # label for specific k
                           n_clusters, kmeans_seed,  # parameter
                           clusterer.inertia_,  # elbow
                           silhouette_avg, silhouette_samples(data, cluster_labels),  # silhouette
                           np.mean(np.log(refDisps)) - np.log(clusterer.inertia_), [refDisps]  # gap statistic
                           ))
            print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg,
                  " elbow :", clusterer.inertia_,
                  " gap statistic :", np.log(np.mean(refDisps)) - np.log(clusterer.inertia_))
    np.save(path_saving + '/measure_cluster.npy', result)

def plot_cluster_analysis(path):
    """
    plot the result of the measure of clustering
    :param path: path of the measure_cluster
    :return:
    """
    # get data
    data = np.load(path + '/measure_cluster.npy', allow_pickle=True)
    nb_clusters = data[:, 1]
    elbow = data[:, 3]
    silhouette_avg = data[:, 4]

    # plot elbow measure
    plt.figure()
    plt.plot(nb_clusters, elbow, '.-')
    plt.grid()
    plt.ylabel('elbow')
    plt.savefig(path+'/elbow.png')

    # plot average silhouette
    plt.figure()
    plt.plot(nb_clusters, silhouette_avg, '.-')
    plt.grid()
    plt.ylabel('silhouette_avg')
    plt.savefig(path+'/silhouette_avg.png')

    # plot gap statistic
    ref_ = np.concatenate(data[:, 7])
    mean_ref = np.mean(np.log(ref_), axis=1)
    std_ref = np.std(np.log(ref_), axis=1)
    gap_stat_values = mean_ref - np.log(np.array(elbow, dtype=float))
    plt.figure()
    plt.plot(nb_clusters, gap_stat_values, '.-')
    previous = 0.0
    for index, (nb_cluster, gap_stat_value) in enumerate(zip(nb_clusters, gap_stat_values)):
        # https://rpubs.com/Yoann/587951
        # https://towardsdatascience.com/k-means-clustering-and-the-gap-statistics-4c5d414acd29
        # https://www.researchgate.net/post/How-to-interpret-the-output-of-Gap-Statistics-method-for-clustering
        # https://hastie.su.domains/Papers/gap.pdf
        # https://stats.stackexchange.com/questions/95290/how-should-i-interpret-gap-statistic
        plt.plot([nb_cluster, nb_cluster],
                 [gap_stat_value + std_ref[index], gap_stat_value - std_ref[index]], 'g')
        if index + 1 != len(nb_clusters) \
                and gap_stat_value > gap_stat_values[index + 1] + std_ref[index + 1] \
                and gap_stat_value > previous:
            previous = gap_stat_value
            plt.plot(nb_cluster, gap_stat_value, 'r*')
    plt.grid()
    plt.ylabel('gap_statistic')
    plt.savefig(path+'/gap_statistic.png')


def plot_cluster_all_gap_statistic(path_saving_root, path_data_root, range_PHATE_knn, range_PHATE_decay):
    nb_PHATE_knn = len(range_PHATE_knn)
    nb_PHATE_decay = len(range_PHATE_decay)
    fig, axis = plt.subplots(nb_PHATE_decay, nb_PHATE_knn, figsize=(20,20))
    for index_knn, PHATE_knn in enumerate(range_PHATE_knn):
        for index_decay, PHATE_decay in enumerate(range_PHATE_decay):
            path_data = path_data_root + '/PHATE_KNN_'+str(PHATE_knn)+'_decay_'+str(PHATE_decay)+'/'
            print(path_data)
            # get data
            data = np.load(path_data + '/measure_cluster.npy', allow_pickle=True)
            nb_clusters = data[:, 1]
            elbow = data[:, 3]

            # plot gap statistic
            ref_ = np.concatenate(data[:, 7])
            mean_ref = np.mean(np.log(ref_), axis=1)
            std_ref = np.std(np.log(ref_), axis=1)
            gap_stat_values = mean_ref - np.log(np.array(elbow, dtype=float))
            axis[index_decay, index_knn].plot(nb_clusters, gap_stat_values, '.-')
            previous = 0.0
            for index, (nb_cluster, gap_stat_value) in enumerate(zip(nb_clusters, gap_stat_values)):
                # https://rpubs.com/Yoann/587951
                # https://towardsdatascience.com/k-means-clustering-and-the-gap-statistics-4c5d414acd29
                # https://www.researchgate.net/post/How-to-interpret-the-output-of-Gap-Statistics-method-for-clustering
                # https://hastie.su.domains/Papers/gap.pdf
                # https://stats.stackexchange.com/questions/95290/how-should-i-interpret-gap-statistic
                axis[index_decay, index_knn].plot([nb_cluster, nb_cluster], [gap_stat_value+std_ref[index], gap_stat_value-std_ref[index]], 'g')
                if index+1 != len(nb_clusters)\
                    and gap_stat_value > gap_stat_values[index+1] + std_ref[index+1]\
                    and gap_stat_value > previous:
                    previous = gap_stat_value
                    axis[index_decay, index_knn].plot(nb_cluster, gap_stat_value, 'r*')
            axis[index_decay, index_knn].grid()
    plt.subplots_adjust(left=0.06, bottom=0.05, right=0.97, top=0.96, )
    # plt.show()
    plt.savefig(path_saving_root+'/gap_statistic_all.png')

    fig, axis = plt.subplots(nb_PHATE_decay, nb_PHATE_knn, figsize=(20,20))
    for index_knn, PHATE_knn in enumerate(range_PHATE_knn):
        for index_decay, PHATE_decay in enumerate(range_PHATE_decay):
            path_data = path_data_root + '/PHATE_KNN_'+str(PHATE_knn)+'_decay_'+str(PHATE_decay)+'/'
            print(path_data)
            # get data
            data = np.load(path_data + '/measure_cluster.npy', allow_pickle=True)
            nb_clusters = data[:, 1]
            silhouette_avg = data[:, 4]
            axis[index_decay, index_knn].plot(nb_clusters, silhouette_avg, '.-')
            axis[index_decay, index_knn].grid()
    plt.subplots_adjust(left=0.06, bottom=0.05, right=0.97, top=0.96, )
    # plt.show()
    plt.savefig(path_saving_root+'/silhouette_all.png')

    fig, axis = plt.subplots(nb_PHATE_decay, nb_PHATE_knn, figsize=(20,20))
    for index_knn, PHATE_knn in enumerate(range_PHATE_knn):
        for index_decay, PHATE_decay in enumerate(range_PHATE_decay):
            path_data = path_data_root + '/PHATE_KNN_'+str(PHATE_knn)+'_decay_'+str(PHATE_decay)+'/'
            print(path_data)
            # get data
            data = np.load(path_data + '/measure_cluster.npy', allow_pickle=True)
            nb_clusters = data[:, 1]
            elbow = data[:, 3]
            axis[index_decay, index_knn].plot(nb_clusters, elbow, '.-')
            axis[index_decay, index_knn].grid()
    plt.subplots_adjust(left=0.06, bottom=0.05, right=0.97, top=0.96, )
    # plt.show()
    plt.savefig(path_saving_root+'/elbow_all.png')

if __name__ == "__main__":
    import os
    path_data_default = "/home/kusch/Documents/project/patient_analyse/paper/result/default/"
    path_saving_default = "/home/kusch/Documents/project/patient_analyse/paper/result/cluster_measure/default/"
    range_n_clusters = np.arange(2, 15, 1)
    kmeans_seed = 123
    range_kmeans_seed = [123]  # np.arange(123, 133, 1)
    nrefs = 100

    path_data_root = "/home/kusch/Documents/project/patient_analyse/paper/result/sensibility_analysis/"
    path_saving_root = "/home/kusch/Documents/project/patient_analyse/paper/result/cluster_measure/sensibility_analysis/"

    range_PHATE_knn = [2, 3, 4, 5, 6, 7, 10]
    range_PHATE_decay = [0.1, 1.0, 10.0, 100.0]
    for PHATE_knn in range_PHATE_knn:
        for PHATE_decay in range_PHATE_decay:
            path_saving = path_saving_root + '/PHATE_KNN_'+str(PHATE_knn)+'_decay_'+str(PHATE_decay)+'/'
            path_data = path_data_root + '/PHATE_KNN_'+str(PHATE_knn)+'_decay_'+str(PHATE_decay)+'/'
            if os.path.exists(path_data + "/Phate.npy"):
                print(path_saving)
                if not os.path.exists(path_saving):
                    os.mkdir(path_saving)
                if not os.path.exists(path_saving+'measure_cluster.npy'):
                    generate_cluster_precision(path_data, path_saving, range_n_clusters, range_kmeans_seed, nrefs)
                plot_cluster_analysis(path_saving)
                plt.close('all')
    plot_cluster_all_gap_statistic(path_saving_root, path_saving_root, range_PHATE_knn, range_PHATE_decay)