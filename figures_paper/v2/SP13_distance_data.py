import matplotlib.pyplot as plt
import numpy as np
import os

path_data = os.path.dirname(os.path.realpath(__file__)) + '/../../paper/result/PCA/'
label_size = 12.0

range_i = np.array(range(199))+1
# get data of distance by neighbours (version 2)
result_avalanches_pattern = np.load(path_data + '/avalanches_pattern/distance_max_2.npy')[:, :199].T
result_avalanches = np.load(path_data + '/avalanches/distance_max_2.npy')[:, :199].T
result_data = np.load(path_data + '/data/distance_max_2.npy')[:, :199].T
result_data_normalized = np.load(path_data + '/data_normalized/distance_max_2.npy')[:, :199].T

# get data of distance by neighbours for avalanches with euclidean or cosine distance(version 1)
result_avalanches_cosine = np.load(path_data + '/avalanches/distance_max_cosine.npy')[:, :199].T

letter = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']

plt.figure(figsize=(5, 7))

plt.subplot(2, 2, 1)
plt.plot(np.flip(range_i), result_data_normalized)
plt.xlabel('neighbour of KNN')
plt.ylabel('maximal of distance')
plt.annotate(letter[1], xy=(-0.1, 0.96), xycoords='axes fraction', weight='bold', fontsize=label_size)
plt.title('normalized data\n(euclidean)', weight='bold')

plt.subplot(2, 2, 2)
plt.plot(np.flip(range_i), result_avalanches)
plt.xlabel('neighbour of KNN')
plt.ylabel('maximal of distance')
plt.annotate(letter[2], xy=(-0.1, 0.96), xycoords='axes fraction', weight='bold', fontsize=label_size)
plt.title('avalanches\n(euclidean)', weight='bold')

plt.subplot(2, 2, 3)
plt.plot(np.flip(range_i), result_avalanches_cosine)
plt.xlabel('neighbour of KNN')
plt.ylabel('maximal of distance')
plt.annotate(letter[3], xy=(-0.1, 0.96), xycoords='axes fraction', weight='bold', fontsize=label_size)
plt.title('avalanches\n(cosine)', weight='bold')

plt.subplot(2, 2, 4)
plt.plot(np.flip(range_i), result_avalanches_pattern)
plt.xlabel('neighbour of KNN')
plt.ylabel('maximal of distance')
plt.annotate(letter[4], xy=(-0.1, 0.96), xycoords='axes fraction', weight='bold', fontsize=label_size)
plt.title('avalanches pattern\n(cosine)', weight='bold')

plt.subplots_adjust(left=0.11, right=0.94, top=0.93, bottom=0.07, wspace=0.3, hspace=0.4)
plt.savefig('figure/SP_13_distance_data.png')
plt.show()
