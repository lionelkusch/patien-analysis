import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from pipeline_phate_clustering.functions_helper.load_data import go_avalanches
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
# compute the avalanches for each patient
avalanches_threshold = 3
avalanches_direction = 0
avalanches_binsize = 1
avalanches_bin = []
subjects_index = []
data = []
for subject in selected_subjects:
    Avalanches_human = go_avalanches(patients_data[subject], thre=avalanches_threshold,
                                     direc=avalanches_direction, binsize=avalanches_binsize)
    avalanches = []
    for kk1 in range(len(Avalanches_human['ranges'])):
        begin = Avalanches_human['ranges'][kk1][0]
        end = Avalanches_human['ranges'][kk1][1]
        avalanches.append(Avalanches_human['Zbin'][begin:end, :])
    data.append(np.concatenate(avalanches))

# compute for euclidean distance
results_all = []
range_neighboor = range(1, 20, 2)
for j in range(1, 10):
    if j == 0:
        data_fit = np.concatenate(data)
    else:
        data_fit = PCA(n_components=j).fit_transform(np.concatenate(data))
    knn_tree = NearestNeighbors(n_neighbors=21, algorithm='auto', metric='euclidean', n_jobs=8).fit(data_fit)
    print('fit')
    graph = knn_tree.kneighbors_graph(data_fit, n_neighbors=200, mode='distance')
    print('distance')
    distance = []
    for i in range(200):
        print(j, i)
        distance.append(np.max(graph.max(axis=0).todense()))
        index = np.squeeze(np.array(graph.argmax(axis=1)), axis=1)
        graph[range(graph.shape[0]), index] = 0
        graph.eliminate_zeros()
    results_all.append(np.array(distance))
np.save(path_data+'/../paper/result/PCA/avalanches/distance_max_2.npy', results_all)


# compute for cosine distance
results_all = []
range_neighboor = range(1, 20, 2)
for j in range(1, 10):
    if j == 0:
        data_fit = np.concatenate(data)
    else:
        data_fit = PCA(n_components=j).fit_transform(np.concatenate(data))
    knn_tree = NearestNeighbors(n_neighbors=21, algorithm='auto', metric='cosine', n_jobs=8).fit(data_fit)
    print('fit')
    graph = knn_tree.kneighbors_graph(data_fit, n_neighbors=200, mode='distance')
    print('distance')
    distance = []
    for i in range(200):
        print(j, i)
        distance.append(np.max(graph.max(axis=0).todense()))
        index = np.squeeze(np.array(graph.argmax(axis=1)), axis=1)
        graph[range(graph.shape[0]), index] = 0
        graph.eliminate_zeros()
    results_all.append(np.array(distance))
np.save(path_data+'/../paper/result/PCA/avalanches/distance_max_cosine.npy', results_all)
