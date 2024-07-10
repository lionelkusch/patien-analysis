import os
import numpy as np
import matplotlib.pyplot as plt

label_size = 12.0

path_data = os.path.dirname(os.path.realpath(__file__)) + '/../..//paper/result/'
path_saving = path_data + '/cluster_measure/sensibility_analysis/PHATE_KNN_5_decay_1.0/'
data = np.load(path_saving + '/measure_cluster.npy', allow_pickle=True)
nb_clusters = data[:, 1]
elbow = data[:, 3]
silhouette_avg = data[:, 4]
ref_ = np.concatenate(data[:, 7])
mean_ref = np.mean(np.log(ref_), axis=1)
std_ref = np.std(np.log(ref_), axis=1)
gap_stat_values = mean_ref - np.log(np.array(elbow, dtype=float))


path_saving = path_data + '/PCA/avalanches_pattern/'
pca_data = np.load(path_saving + '/measure_cluster.npy', allow_pickle=True)
pca_nb_clusters = pca_data[:, 1]
pca_elbow = pca_data[:, 3]
pca_silhouette_avg = pca_data[:, 4]
pca_ref_ = np.concatenate(pca_data[:, 7])
pca_mean_ref = np.mean(np.log(pca_ref_), axis=1)
pca_std_ref = np.std(np.log(pca_ref_), axis=1)
pca_gap_stat_values = pca_mean_ref - np.log(np.array(pca_elbow, dtype=float))



plt.figure(figsize=(10, 5))
ax = plt.subplot(2, 3, 1)
ax.plot(nb_clusters, elbow)
ax.set_ylabel('elbow', {"fontsize": label_size})
plt.xticks([2, 5, 10, 15])

ax = plt.subplot(2, 3, 2)
ax.plot(nb_clusters, silhouette_avg)
ax.set_xlabel('nb cluster', {"fontsize": label_size})
ax.set_ylabel('average silhouette', {"fontsize": label_size})
ax.set_title('PHATE', {"fontsize": label_size}, weight='bold')
plt.xticks([2, 5, 10, 15])

ax = plt.subplot(2, 3, 3)
ax.plot(nb_clusters, gap_stat_values)
ax.set_ylabel('gap statistic', {"fontsize": label_size})
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
    plt.xticks([2, 5, 10, 15])


ax = plt.subplot(2, 3, 4)
ax.plot(pca_nb_clusters, pca_elbow)
ax.set_ylabel('elbow', {"fontsize": label_size})
plt.xticks([2, 5, 10, 15])

ax = plt.subplot(2, 3, 5)
ax.plot(pca_nb_clusters, pca_silhouette_avg)
ax.set_xlabel('nb cluster', {"fontsize": label_size})
ax.set_ylabel('average silhouette', {"fontsize": label_size})
ax.set_title('PCA', {"fontsize": label_size}, weight='bold')
plt.xticks([2, 5, 10, 15])

ax = plt.subplot(2, 3, 6)
ax.plot(pca_nb_clusters, pca_gap_stat_values)
ax.set_ylabel('gap statistic', {"fontsize": label_size})
pca_previous = 0.0
for index, (pca_nb_cluster, pca_gap_stat_value) in enumerate(zip(pca_nb_clusters, pca_gap_stat_values)):
    # https://rpubs.com/Yoann/587951
    # https://towardsdatascience.com/k-means-clustering-and-the-gap-statistics-4c5d414acd29
    # https://www.researchgate.net/post/How-to-interpret-the-output-of-Gap-Statistics-method-for-clustering
    # https://hastie.su.domains/Papers/gap.pdf
    # https://stats.stackexchange.com/questions/95290/how-should-i-interpret-gap-statistic
    plt.plot([pca_nb_cluster, pca_nb_cluster],
             [pca_gap_stat_value + pca_std_ref[index], pca_gap_stat_value - pca_std_ref[index]], 'g')
    if index + 1 != len(pca_nb_clusters) \
            and pca_gap_stat_value > pca_gap_stat_values[index + 1] + pca_std_ref[index + 1] \
            and pca_gap_stat_value > pca_previous:
        pca_previous = pca_gap_stat_value
        plt.plot(pca_nb_cluster, pca_gap_stat_value, 'r*')
    plt.xticks([2, 5, 10, 15])

plt.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.09, wspace=0.32, hspace=0.4)
# plt.show()
plt.savefig('figure/SP_11_cluster_measure.png')
