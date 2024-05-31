import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# from select_data import get_data_selected_patient_1
# avalanches_bin, avalanches_sum, out, out_sum = get_data_selected_patient_1()
path_data = os.path.dirname(os.path.realpath(__file__)) + '/../data/'
avalanches_bin = np.load(path_data+'/avalanches_selected_patient.npy', allow_pickle=True)

results_all=[]
for j in range(10):
    print(j)
    if j == 0:
        data = np.concatenate(avalanches_bin)
    else:
        data = PCA(n_components=j).fit_transform(np.concatenate(avalanches_bin))
    knn_tree = NearestNeighbors(n_neighbors=200, algorithm='auto', metric='cosine', n_jobs=8).fit(data)
    print('fit')
    graph = knn_tree.kneighbors_graph(data, n_neighbors=200, mode='distance')
    print('distance')
    distance = []
    for i in range(400):
        print(j, i)
        distance.append(np.max(graph.max(axis=0).todense()))
        index = np.squeeze(np.array(graph.argmax(axis=1)), axis=1)
        graph[range(graph.shape[0]), index]=0
        graph.eliminate_zeros()
    results_all.append(np.array(distance))

np.save(path_data+'/../paper/result/PCA/avalanches_pattern/distance_max_2.npy', results_all)
