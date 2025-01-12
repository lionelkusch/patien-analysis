import numpy as np
from scipy.stats import entropy
import scipy.io as io

def get_entropy(path, name, base = None, nb_region=90):
    histograms_region = np.load(path + "/histograms_region.npy")
    min = histograms_region.min(axis=0).reshape(1, nb_region)
    max = histograms_region.max(axis=0).reshape(1, nb_region)
    if np.any(min < 0.0):
        cluster_vector = (histograms_region - min) / (max-min)
        print('rectify')
    else:
        cluster_vector = histograms_region / max
    print(name, ': ', entropy(cluster_vector.ravel(), base=base))


def get_entropy_pca(path, name, base = None):
    cluster_vector = io.loadmat(path+'vector_cluster.mat')['cluster_vector']
    min = cluster_vector.min(axis=0)
    if np.any(min < 0.0):
        print('rectify')
        cluster_vector -= cluster_vector.min(axis=0)
    print(name, ': ', entropy(cluster_vector.ravel(), base=base))


if __name__ == '__main__':
    import os
    path_root = os.path.dirname(os.path.realpath(__file__)) + '/../../'
    path_phate = path_root + "/paper/result/default/"
    for name, path in [
        ('Phate', path_root + "/paper/result/default/"),
        # ('avalanches', path_root + "/paper/result/no_avalanche/avalanches/"), # more neighboor
        # ('avalanches 2', path_root + "/paper/result/no_avalanche/avalanches_2/"), # euclidean distance
        ('avalanches 3', path_root + "/paper/result/no_avalanche/avalanches_3/"),
        ('data euclidean', path_root + "/paper/result/no_avalanche/data_euclidean/"),
        # ('data euclidean 1', path_root + "/paper/result/no_avalanche/data_euclidiean_1/"),
        # ('data normalized euclidean', path_root + "/paper/result/no_avalanche/data_normalized_euclidean/"),
        ('data normalized euclidean 2', path_root + "/paper/result/no_avalanche/data_normalized_euclidean_2/"), # different decay
        # ('data normalized euclidean cosine', path_root + "/paper/result/no_avalanche/data_normalized_euclidean_cosine/"),
        ('spectral 3', path_root+"/paper/result/spectral_cosine/spectral_cosine3"),
        ('spectral 4', path_root + "/paper/result/spectral_cosine/spectral_cosine4"),
        ('spectral 5', path_root + "/paper/result/spectral_cosine/spectral_cosine5"),
        ('spectral 6', path_root + "/paper/result/spectral_cosine/spectral_cosine6"),
        ('spectral 7', path_root + "/paper/result/spectral_cosine/spectral_cosine7"),
        ('spectral 8', path_root + "/paper/result/spectral_cosine/spectral_cosine8"),
        ('spectral 9', path_root + "/paper/result/spectral_cosine/spectral_cosine9"),
        ('spectral 10', path_root + "/paper/result/spectral_cosine/spectral_cosine10"),
        ('spectral 11', path_root + "/paper/result/spectral_cosine/spectral_cosine10"),
        ('spectral 12', path_root + "/paper/result/spectral_cosine/spectral_cosine10"),
        ('spectral 13', path_root + "/paper/result/spectral_cosine/spectral_cosine10"),
    ]:
        get_entropy(path, name)
    for name, path in [
        ('PCA avalanches pattern', path_root + "/paper/result/PCA/avalanches_pattern/"),
        ('PCA avalanches', path_root + "/paper/result/PCA/avalanches/"),
        ('PCA data normalized', path_root + "/paper/result/PCA/data_normalized/"),
        ('PCA data', path_root + "/paper/result/PCA/data/"),
    ]:
        get_entropy_pca(path, name)

