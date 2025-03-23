import itertools
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import os

label_size = 12.0
tickfont_size = 10.0
label_col_size = 8.0
linewidth = 0.5

def compare_cluster(name_fig, paths, cluster_vector_order, bottom=0.15):
    cluster_vector = {}
    for name, path in paths:
        if os.path.exists(path + 'vector_cluster.mat'):
            cluster_vector[name] = io.loadmat(path + 'vector_cluster.mat')['cluster_vector']
        else:
            cluster_vector[name] = np.load(path + "/histograms_region.npy")
        assert len(np.unique(cluster_vector_order[name])) == len(cluster_vector_order[name])
        cluster_vector[name] = cluster_vector[name][cluster_vector_order[name]]

    matrix_vector = np.concatenate(list(cluster_vector.values()))
    correlation = np.corrcoef(matrix_vector)
    labels = np.concatenate([[name+' '+str(i) for i in cluster_vector_order[name]]for name in cluster_vector.keys()])
    plt.figure(figsize=(6.4, 4.2))
    im = plt.imshow(correlation, vmin=-1.0, vmax=1.0, cmap='coolwarm')
    col = plt.colorbar(im)
    col.ax.set_ylabel('correlation')
    col.ax.tick_params('both', labelsize=tickfont_size)
    plt.xticks(np.arange(0, matrix_vector.shape[0]), labels, rotation=90)
    plt.yticks(np.arange(0, matrix_vector.shape[0]), labels,)
    plt.tick_params('both', labelsize=tickfont_size)
    plt.subplots_adjust(top=0.98, bottom=bottom, left=0., right=1.0, )
    plt.savefig('figure_3/'+name_fig, dpi=600)
    plt.show()

if __name__ == '__main__':
    label_size = 12.0
    tickfont_size = 10.0
    label_col_size = 8.0
    linewidth = 0.5

    path_root = os.path.dirname(os.path.realpath(__file__)) + '/../../'
    path_phate = path_root + "/paper/result/default/"
    paths = [('Phate', path_root + "/paper/result/default/"),
            ('Sp Cl', path_root + "/paper/result/spectral_cosine/spectral_cosine7/")]
    compare_cluster('SP_23_Phate_Sp.png', paths,
                    {'Phate': [1, 4, 5, 2, 0, 3, 6],
                     'Sp Cl': [4, 5, 3, 2, 6, 0, 1]},
                    bottom=0.155)
    paths = [
        ('Phate 18 ', path_root + "/paper/result/default/"),
        ('Phate all', path_root + "/paper/result/all_subject_melbourne/"),
    ]
    compare_cluster('SP_24_Phate_all.png', paths,
                    {'Phate 18 ': [1, 4, 5, 2, 0, 3, 6],
                     'Phate all': [2, 3, 6, 1, 4, 0, 5]},
                    bottom=0.215,
                    )