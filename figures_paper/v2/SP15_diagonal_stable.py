import os
import numpy as np
import matplotlib.pyplot as plt
from pipeline_phate_clustering.null_model_sensibility_analysis.get_result_null_model import get_color_map


path = os.path.dirname(os.path.realpath(__file__)) + "/../../paper/result/default/"
significatif = 0.05
nb_randomize = 100
label_size = 12.0
label_size_min = 8.0
max_nb_cluster = 15

pvalues_prob_transisiton = np.load(path+'/model_diagonal.npy', allow_pickle=True)
diag_per_lows, diag_per_nos, diag_per_highs, no_diag_per_lows, no_diag_per_nos, no_diag_per_highs =\
    np.load(path+'/model_diagonalsignificatif.npy')


plt.figure(figsize=(5, 5))
ax = plt.gca()
index_of_cluster_nb = list(range(2, max_nb_cluster))
for index, data in enumerate(np.array([diag_per_lows, diag_per_nos, diag_per_highs,
                                       no_diag_per_lows, no_diag_per_nos, no_diag_per_highs]).T):
    ax.bar(index_of_cluster_nb[index], data[0], color=['b'], width=0.4)
    ax.bar(index_of_cluster_nb[index], data[1], color=['w'], width=0.4, bottom=data[0])
    ax.bar(index_of_cluster_nb[index], data[2], color=['r'], width=0.4, bottom=data[0]+data[1])
    ax.bar(index_of_cluster_nb[index] + 0.3, data[3], color='b', width=0.4)
    ax.bar(index_of_cluster_nb[index] + 0.3, data[4], color='w', width=0.4, bottom=data[3])
    ax.bar(index_of_cluster_nb[index] + 0.3, data[5], color='r', width=0.4, bottom=data[3] + data[4])
ax.set_xlabel('number of cluster', {"fontsize": label_size})
ax.set_ylabel('% of significant transition', {"fontsize": label_size})

plt.subplots_adjust(left=0.13, right=0.98, top=0.98, bottom=0.13)
plt.savefig('figure/SP_15_diagonal_transition.png')

plt.show()