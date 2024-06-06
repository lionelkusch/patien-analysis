import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from pipeline_phate_clustering.functions_helper.plot import plot_figure_2D, plot_figure_2D_patient, \
    plot_figure_2D_patient_unique, plot_figure_2D_patient_unique_time, plot_figure_2D_3D
from sklearn.metrics import silhouette_samples, silhouette_score
import scipy.io as io

# from select_data import get_data_selected_patient_1
# avalanches_bin, avalanches_sum, out, out_sum = get_data_selected_patient_1()
path_data = os.path.dirname(os.path.realpath(__file__)) + '/../../data/'
avalanches_bin = np.load(path_data+'/avalanches_selected_patient.npy', allow_pickle=True)
pca_choice = 5

pca = PCA(n_components=90)
pca.fit(np.concatenate(avalanches_bin))
print(pca.explained_variance_ratio_)

cumulative = []
count = 0
for i in range(90):
    count += pca.explained_variance_ratio_[i]
    cumulative.append(count)

plt.figure()
plt.plot(np.arange(1, 91, 1), pca.explained_variance_ratio_, alpha=0.5)
plt.plot(np.arange(1, 91, 1), pca.explained_variance_ratio_, 'x')
plt.vlines(pca_choice, ymin=0.0, ymax=pca.explained_variance_ratio_[pca_choice-1], color='r', alpha=0.5)
plt.hlines(pca.explained_variance_ratio_[pca_choice-1], xmin=1, xmax=pca_choice, color='r', alpha=0.5)
plt.savefig(path_data+'/../paper/result/PCA/avalanches_pattern/pca_info.svg')
plt.figure()
plt.plot(np.arange(1, 91, 1), pca.explained_variance_ratio_, alpha=0.5)
plt.plot(np.arange(1, 91, 1), pca.explained_variance_ratio_, 'x')
plt.vlines(pca_choice, ymin=0.0, ymax=pca.explained_variance_ratio_[pca_choice-1], color='r', alpha=0.5)
plt.hlines(pca.explained_variance_ratio_[pca_choice-1], xmin=1, xmax=pca_choice, color='r', alpha=0.5)
plt.yscale("log")
plt.savefig(path_data+'/../paper/result/PCA/avalanches_pattern/pca_info_log.svg')
plt.figure()
plt.plot(np.arange(1, 91, 1), cumulative, alpha=0.5)
plt.plot(np.arange(1, 91, 1), cumulative, 'x')
plt.vlines(pca_choice, ymin=0.0, ymax=cumulative[pca_choice-1], color='r', alpha=0.5)
plt.hlines(cumulative[pca_choice-1], xmin=1, xmax=pca_choice, color='r', alpha=0.5)
plt.savefig(path_data+'/../paper/result/PCA/avalanches_pattern/pca_cumulative.svg')

PCA_fit = PCA(n_components=pca_choice).fit_transform(np.concatenate(avalanches_bin))
cluster = KMeans(n_clusters=7, random_state=123).fit_predict(PCA_fit)

clusters_vectors = []
for i in range(7):
    index = np.where(cluster==i)
    clusters_vectors.append(np.mean(np.concatenate(avalanches_bin)[index], axis=0))
io.savemat(path_data + '/../paper/result/PCA/avalanches_pattern/vector_cluster.mat',
           {'cluster_vector':np.concatenate([clusters_vectors])}
           )

plot_figure_2D(PCA_fit, '', cluster)
plt.savefig(path_data+'/../paper/result/PCA/avalanches_pattern/PCA_cluster_in_2D.png')
plot_figure_2D_patient(PCA_fit, '', avalanches_bin)
plt.savefig(path_data+'/../paper/result/PCA/avalanches_pattern/PCA_cluster_for_patient.png')
plot_figure_2D_patient_unique(PCA_fit, '', avalanches_bin)
plt.savefig(path_data+'/../paper/result/PCA/avalanches_pattern/PCA_cluster_for_patient_unique.png')
plot_figure_2D_patient_unique_time(PCA_fit, '', avalanches_bin)
plt.savefig(path_data+'/../paper/result/PCA/avalanches_pattern/PCA_cluster_for_patient_time.png')
plot_figure_2D_3D(PCA_fit, '', cluster)
plt.savefig(path_data+'/../paper/result/PCA/avalanches_pattern/PCA_cluster_3D.png')
plt.close('all')

# plt.show()


range_kmeans_seed = [123]
nrefs = 100
range_n_clusters = np.arange(2, 15, 1)
data = PCA_fit
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
np.save(path_data+'/../paper/result/PCA/avalanches_pattern/measure_cluster.npy', result)
result = np.load(path_data+'/../paper/result/PCA/avalanches_pattern/measure_cluster.npy', allow_pickle=True)

plt.figure()
plt.subplot(131)
plt.plot(result[:, 1], result[:, 3])
plt.title('elbow')
plt.subplot(132)
plt.plot(result[:, 1], result[:, 4])
plt.title('silhouette')
plt.subplot(133)
plt.plot(result[:, 1], result[:, 6])
plt.title('gap statistic')
plt.subplots_adjust(top=0.95, bottom=0.067, left=0.098, right=0.97, wspace=0.45)
plt.savefig(path_data+'/../paper/result/PCA/avalanches_pattern/measure_cluster.png')
# plt.show()

