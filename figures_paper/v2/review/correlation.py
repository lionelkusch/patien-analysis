import itertools
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import os


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
    plt.figure()
    im = plt.imshow(correlation, vmin=-1.0, vmax=1.0, cmap='coolwarm')
    plt.colorbar(im)
    plt.xticks(np.arange(0, matrix_vector.shape[0]), labels, rotation=90)
    plt.yticks(np.arange(0, matrix_vector.shape[0]), labels,)
    plt.subplots_adjust(top=0.98, bottom=bottom)
    plt.savefig('figure/'+name_fig)
    plt.show()


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


if __name__ == '__main__':
    label_size = 12.0
    tickfont_size = 10.0
    label_col_size = 8.0
    linewidth = 0.5

    path_root = os.path.dirname(os.path.realpath(__file__)) + '/../../../'
    path_phate = path_root + "/paper/result/default/"
    paths = [('PHATE', path_root + "/paper/result/default/"),
            ('Sp Cl', path_root + "/paper/result/spectral_cosine/spectral_cosine7/")]
# compare_cluster('Phate_Sp.png', paths,
#                 {'PHATE': [1, 4, 5, 2, 0, 3, 6],
    #                  'Sp Cl': [4, 5, 3, 2, 6, 0, 1]})
    print(value_correlation(paths))
    paths = [('PHATE', path_root + "/paper/result/default/"),
            ('Sp Cl', path_root + "/paper/result/spectral_cosine/spectral_cosine7/"),
            ('PCA', path_root + "/paper/result/PCA/avalanches_pattern/"),
             ]
    # compare_cluster('Phate_Sp_PCA.png', paths,
    #                 {'PHATE': [1, 4, 5, 2, 0, 3, 6],
    #                  'Sp Cl': [4, 5, 3, 2, 6, 0, 1],
    #                  'PCA':[0, 5, 1, 2, 6, 4, 3]}
    #                 )
    print(value_correlation(paths))
    paths = [
        ('PHATE 18 ', path_root + "/paper/result/default/"),
        ('PHATE all', path_root + "/paper/result/all_subject_melbourne/"),
    ]
    # compare_cluster('Phate_all.png', paths,
    #                 {'PHATE 18 ': [1, 4, 5, 2, 0, 3, 6],
    #                  'PHATE all': [2, 3, 6, 1, 4, 0, 5]},
    #                 bottom=0.2,
    #                 )
    print(value_correlation(paths))
    paths = [
        ('default', path_root + '/paper/result/default/'),
        ('th=2.7', path_root + '/paper/result/sensibility_analysis/PHATE_th_2.7000000000000015/'),
        ('th=3.0', path_root + '/paper/result/sensibility_analysis/PHATE_th_3.0000000000000018/'),
        ('th=3.2', path_root + '/paper/result/sensibility_analysis/PHATE_th_3.200000000000002/'),
    ]
    # compare_cluster('th.png', paths,
    #                 {
    #                  'default': [1, 4, 5, 2, 0, 3, 6],
    #                  'th=2.7': [2, 0, 4, 6, 1, 3, 5],
    #                  'th=3.0': [4, 1, 2, 5, 3, 0, 6],
    #                  'th=3.2': [6, 3, 4, 2, 0, 5, 1]
    #                 },
    #                 )
    print(value_correlation(paths))

    # th
    th_range = np.arange(1.0, 6.0, 0.1)
    entropy_list = []
    paths= [('default', path_root + '/paper/result/default/')]
    for th in th_range:
        paths.append((str(th), path_phate + '/../sensibility_analysis/PHATE_th_'+str(th)))

    results_all = value_correlation(paths)
    result = []
    for th in th_range:
        result.append(results_all['default_'+str(th)])
    plt.figure(figsize=(5, 4))
    plt.plot(th_range, result)
    plt.ylabel('maximum average of\ncorrelation of default result', {"fontsize": label_size})
    plt.xlabel("threshold of the normalised data", {"fontsize": label_size})
    plt.tick_params('both', labelsize=tickfont_size)
    plt.subplots_adjust(top=0.98, bottom=0.11, left=0.18, right=0.99)
    plt.savefig('figure/correlation_th.png')
    plt.show()