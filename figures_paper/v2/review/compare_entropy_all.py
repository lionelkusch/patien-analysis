import numpy as np
import os
from scipy.stats import entropy
import matplotlib.pyplot as plt

# entropy
path = os.path.dirname(os.path.realpath(__file__)) + "/../../../paper/result/default/"
path_all = os.path.dirname(os.path.realpath(__file__)) + "/../../../paper/result/all_subject_melbourne/"
nb_randomize_1 = 10000
significatif = 0.05 / (90 * 7)
label_size = 12.0

# load value
data_null_model = []
for nb_rand in range(nb_randomize_1):
    data_null_model.append(np.load(path + "/histograms_region_" + str(nb_rand) + ".npy"))
data_patient = np.load(path + "/histograms_region.npy")
nb_cluster = data_patient.shape[0]
pvalue_cluster = []
entropy_values = []
for index, cluster_region in enumerate(data_patient):
    pvalue = np.sum(
        np.sum(np.array(data_null_model) > cluster_region, axis=0) / nb_randomize_1, axis=0) / nb_cluster
    significatif_high = pvalue > 1.0 - significatif
    significatif_low = pvalue < significatif
    significatif_all = np.logical_or(significatif_low, significatif_high)
    pvalue_cluster.append(
        [[pvalue], [significatif_all], [significatif_high], [significatif_low]])
for data in data_null_model:
    entropy_values.append(entropy(np.array(data).ravel(), base=None))
pvalue_cluster = np.array(pvalue_cluster)

# load value
data_null_model_all = []
for nb_rand in range(nb_randomize_1):
    data_null_model_all.append(np.load(path_all + "/histograms_region_" + str(nb_rand) + ".npy"))
data_patient_all = np.load(path + "/histograms_region.npy")
nb_cluster_all = data_patient_all.shape[0]
pvalue_cluster_all = []
entropy_values_all = []
for index, cluster_region in enumerate(data_patient_all):
    pvalue_all = np.sum(
        np.sum(np.array(data_null_model_all) > cluster_region, axis=0) / nb_randomize_1, axis=0) / nb_cluster_all
    significatif_high_all = pvalue_all > 1.0 - significatif
    significatif_low_all = pvalue_all < significatif
    significatif_all_all = np.logical_or(significatif_low_all, significatif_high_all)
    pvalue_cluster_all.append(
        [[pvalue_all], [significatif_all_all], [significatif_high_all], [significatif_low_all]])
for data in data_null_model_all:
    entropy_values_all.append(entropy(np.array(data).ravel(), base=None))
pvalue_cluster_all = np.array(pvalue_cluster_all)


plt.figure()
d = .015  # how big to make the diagonal lines in axes coordinates
y, x, _ = plt.hist(entropy_values_all, bins=10, histtype='step', color='black', label='all subjects')
y, x, _ = plt.hist(entropy_values, bins=10, histtype='step', color='blue', label='18 subjects')
plt.ylim(ymax=0.3*nb_randomize_1+500)
plt.tick_params('both', pad=1)
plt.xlabel('entropy', {"fontsize": label_size})
plt.yticks([])
plt.legend()
plt.savefig('figure/compare_entropy.png')
plt.show()