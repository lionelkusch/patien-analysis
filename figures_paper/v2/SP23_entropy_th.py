import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import io
import itertools
from pipeline_phate_clustering.functions_helper.get_entropy import get_entropy

def value_correlation(paths):
    cluster_vector = {}
    for name, path in paths:
        if os.path.exists(path + 'vector_cluster.mat'):
            cluster_vector[name] = io.loadmat(path + 'vector_cluster.mat')['cluster_vector']
        else:
            cluster_vector[name] = np.load(path + "/histograms_region.npy")
    result = {}
    for name_A, name_B in itertools.product(cluster_vector.keys(), cluster_vector.keys()):
        if name_A != name_B:
            assert np.all(cluster_vector[name_A].shape == cluster_vector[name_B].shape)
            nb_cluster = cluster_vector[name_A].shape[0]
            matrix_vector = np.concatenate((cluster_vector[name_A], cluster_vector[name_B]))
            correlation = np.corrcoef(matrix_vector)
            res = np.max(correlation[nb_cluster:, :nb_cluster], axis=0)
            result[name_A+'_'+name_B] = np.mean(res)
    return result

label_size = 12.0
tickfont_size = 10.0
label_col_size = 8.0
linewidth = 0.5


path_root = os.path.dirname(os.path.realpath(__file__)) + '/../../'
path_phate = path_root + "/paper/result/default/"

# th
th_range = np.arange(1.0, 6.0, 0.1)[10:40]
entropy_list = []
paths= [('default', path_root + '/paper/result/default/')]
for th in th_range:
    path_saving = path_phate + '/../sensibility_analysis/PHATE_th_'+str(th)
    paths.append((str(th), path_saving))
    entropy_list.append(get_entropy(path_saving, 'th_' + str(th)))

plt.figure(figsize=(6.4, 3.8))
plt.subplot(211)
for th, entropy in zip(th_range, entropy_list):
    if np.abs(th - 3.0) < 1e-4:
        plt.plot(th, entropy, 'rx')
    else:
        plt.plot(th, entropy, 'bx')
plt.ylabel('entropy', {"fontsize": label_size})
plt.gca().set_xticklabels([])
plt.tick_params('both', labelsize=tickfont_size)

plt.subplot(212)
results_all = value_correlation(paths)
result = []
for th in th_range:
    result.append(results_all['default_'+str(th)])
plt.plot(th_range, result)
plt.ylabel('maximum average\nof correlation with\ndefault result', {"fontsize": label_size})
plt.xlabel("threshold of the normalised data", {"fontsize": label_size})
plt.tick_params('both', labelsize=tickfont_size)

plt.subplots_adjust(top=0.98, bottom=0.13, left=0.20, right=0.98, hspace=0.05)
plt.savefig('figure_3/SP_21_entropy_th.png', dpi=600)
plt.show()
