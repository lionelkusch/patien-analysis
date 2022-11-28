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

            result.append((n_clusters, kmeans_seed, # parameter
                           clusterer.inertia_, # elbow
                           silhouette_avg, silhouette_samples(data, cluster_labels), # silhouette
                           np.log(np.mean(refDisps)) - np.log(clusterer.inertia_) # gap statistic
                           ))
            print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg,
                  " elbow :", clusterer.inertia_,
                  " gap statistic :", np.log(np.mean(refDisps)) - np.log(clusterer.inertia_))
    np.save(path_saving + '/measure_cluster.npy', result)

def plot_cluster_analisys(path):
    """
    plot the result of the measure of clustering
    :param path: path of the measure_cluster
    :return:
    """
    # get data
    data = np.load(path + 'result_silhouette.npy', allow_pickle=True)
    nb_cluster = data[:, 0]
    elbow = data[:, 2]
    silhouette_avg = data[:, 3]
    gap_statistic = data[:, 5]

    # plot elbow measure
    plt.figure()
    plt.plot(nb_cluster, elbow, '.-')
    plt.grid()
    plt.ylabel('elbow')
    plt.savefig(path+'/elbow.png')

    # plot average silhouette
    plt.figure()
    plt.plot(nb_cluster, silhouette_avg, '.-')
    plt.grid()
    plt.ylabel('silhouette_avg')
    plt.savefig(path+'/silhouette_avg.png')

    # plot gap statistic
    plt.figure()
    plt.plot(nb_cluster, gap_statistic, '.-')
    plt.grid()
    plt.ylabel('gap_statistic')
    plt.savefig(path+'/gap_statistic.png')

if __name__ == "__main__":
    path_data = "/home/kusch/Documents/project/patient_analyse/paper/result/default/"
    path_saving = "/home/kusch/Documents/project/patient_analyse/paper/cluster_measure/default/"
    range_n_clusters = np.arange(2, 15, 1)
    kmeans_seed = 123
    range_kmeans_seed = [123]  # np.arange(123, 133, 1)
    nrefs = 100

    generate_cluster_precision(path_data, path_saving, range_n_clusters, range_kmeans_seed, nrefs)
    plot_cluster_analisys(path_saving)