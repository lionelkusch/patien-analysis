import os
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import h5py

# Preparation data
path_data = os.path.dirname(os.path.realpath(__file__)) + '/../data/'
f = h5py.File(path_data + 'serie_Melbourne.mat', 'r')
struArray = f['D']
patients_data = {}
Nsubs = 44
nregions = 90
for i in range(Nsubs):
    patients_data['%d' % i] = np.swapaxes(f[struArray[i, 0]][:nregions, :], 0, 1)
path_data = os.path.dirname(os.path.realpath(__file__)) + '/../data/'
selected_subjects = ['43', '39', '38', '35', '34', '29', '26', '21', '20', '19', '18', '17', '15', '13', '9', '8', '6',
                     '5']
# get data
data = []
for subject in selected_subjects:
    data.append(stats.zscore(patients_data[subject]))
data = np.array(data)

results_all = []
range_neighboor = range(1, 20, 2)
for j in range(1, 10):
    print(j)
    if j == 0:
        data_fit = np.concatenate(data)
    else:
        data_fit = PCA(n_components=j).fit_transform(np.concatenate(data))
    knn_tree = NearestNeighbors(n_neighbors=200, algorithm='auto', metric='euclidean', n_jobs=8).fit(data_fit)
    print('fit')
    graph = knn_tree.kneighbors_graph(data_fit, n_neighbors=200, mode='distance')
    print('distance')
    distance = []
    for i in range(400):
        print(j, i)
        distance.append(np.max(graph.max(axis=0).todense()))
        index = np.squeeze(np.array(graph.argmax(axis=1)), axis=1)
        graph[range(graph.shape[0]), index]=0
        graph.eliminate_zeros()
    results_all.append(np.array(distance))

np.save(path_data+'/../paper/result/PCA/data_normalized/distance_max_2.npy', results_all)
