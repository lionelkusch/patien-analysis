import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# from select_data import get_data_selected_patient_1
# avalanches_bin, avalanches_sum, out, out_sum = get_data_selected_patient_1()
path_data = os.path.dirname(os.path.realpath(__file__)) + '/../data/'
avalanches_bin = np.load(path_data+'/avalanches_selected_patient.npy', allow_pickle=True)
pca_choice = 5

pca = PCA(n_components=90)
pca.fit(np.concatenate(avalanches_bin))
print(pca.explained_variance_ratio_)

cumulative = []
count = 0
for i in range(90):
    count += pca.explained_variance_ratio_[i]
    cumulative.append(count)
plt.figure()
plt.plot(np.arange(1, 91, 1), pca.explained_variance_ratio_, alpha=0.5)
plt.plot(np.arange(1, 91, 1), pca.explained_variance_ratio_, 'x')
plt.vlines(pca_choice, ymin=0.0, ymax=pca.explained_variance_ratio_[pca_choice-1], color='r', alpha=0.5)
plt.hlines(pca.explained_variance_ratio_[pca_choice-1], xmin=1, xmax=pca_choice, color='r', alpha=0.5)
plt.savefig(path_data+'/../paper/result/pca_info.svg')
plt.figure()
plt.plot(np.arange(1, 91, 1), cumulative, alpha=0.5)
plt.plot(np.arange(1, 91, 1), cumulative, 'x')
plt.vlines(pca_choice, ymin=0.0, ymax=cumulative[pca_choice-1], color='r', alpha=0.5)
plt.hlines(cumulative[pca_choice-1], xmin=1, xmax=pca_choice, color='r', alpha=0.5)
plt.savefig(path_data+'/../paper/result/pca_cumulative.svg')
plt.show()