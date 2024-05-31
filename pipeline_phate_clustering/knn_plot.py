import matplotlib.pyplot as plt
import numpy as np
import os

path_data = os.path.dirname(os.path.realpath(__file__)) + '/../paper/result/PCA/'

# get data of distance by neighbours (version 1)
result_avalanches_pattern = np.load(path_data+'/avalanches_pattern/distance_max.npy')
result_avalanches = np.load(path_data+'/avalanches/distance_max.npy')
result_data = np.load(path_data+'/data/distance_max.npy')
result_data_normalized = np.load(path_data+'/data_normalized/distance_max.npy')

for name, result, range_i, range_j in [
    ('avalanches_pattern', result_avalanches_pattern, range(1, 20, 1), range(10)),
    ('avalanches', result_avalanches, range(1, 20, 2), range(1, 10)),
    ('data', result_data, range(1, 20, 1), range(1, 10)),
    ('data_normalized', result_data_normalized, range(1, 20, 2), range(1, 10))
]:
    plt.figure()
    print(name,len(range_i), len(result[0]), len(range_j), len(result))
    for index, i in enumerate(range_j):
        print(index, i)
        plt.plot(range_i, np.mean(result[index], axis=1), label='mean PCA:'+str(i))
        plt.plot(range_i, np.max(result[index], axis=1), label='max PCA:'+str(i))
        plt.title(name+' PCA:'+str(i))
    plt.legend()

# get data of distance by neighbours (version 2)
result_avalanches_pattern = np.load(path_data + '/avalanches_pattern/distance_max_2.npy')[:,:199].T
result_avalanches = np.load(path_data + '/avalanches/distance_max_2.npy')[:,:199].T
result_data = np.load(path_data + '/data/distance_max_2.npy')[:,:199].T
result_data_normalized = np.load(path_data + '/data_normalized/distance_max_2.npy')[:,:199].T

for name, result, range_i, range_j in [
    ('avalanches_pattern', result_avalanches_pattern, np.array(range(199))+1, range(10)),
    ('avalanches', result_avalanches, np.array(range(199))+1, range(1, 10)),
    ('data_normalized', result_data_normalized, np.array(range(199))+1, range(1, 10)),
    ('data', result_data, np.array(range(199)) + 1, range(1, 10)),
]:
    plt.figure()
    print(name, len(range_i), len(result[0]), len(range_j), len(result))
    plt.plot(np.flip(range_i), result)
    plt.title(name)
    plt.xlabel('number of neighboor')
    plt.ylabel('maximal of distance')
    plt.savefig(path_data + '/' + name + '/distance_max.png')


# get data of distance by neighbours for avalanches with euclidean or cosine distance(version 1)
result_avalanches = np.load(path_data + '/avalanches/distance_max_2.npy')[:,:199].T
result_avalanches_cosine = np.load(path_data + '/avalanches/distance_max_cosine.npy')[:,:199].T

for name, result, range_i, range_j in [
    ('avalanches', result_avalanches, np.array(range(199))+1, range(1, 10)),
    ('avalanches_cosine', result_avalanches_cosine, np.array(range(199)) + 1, range(1, 10)),
]:
    plt.figure()
    print(name, len(range_i), len(result[0]), len(range_j), len(result))
    plt.plot(np.flip(range_i), result)
    plt.title(name)
    plt.xlabel('number of neighboor')
    plt.ylabel('maximal of distance')

plt.show()