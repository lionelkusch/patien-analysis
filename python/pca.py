import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# from select_data import get_data_selected_patient_1
# avalanches_bin, avalanches_sum, out, out_sum = get_data_selected_patient_1()
path_data = os.path.dirname(os.path.realpath(__file__)) + '/../data/'
avalanches_bin = np.load(path_data+'/avalanches_selected_patient.npy', allow_pickle=True)

pca = PCA(n_components=90)
pca.fit(np.concatenate(avalanches_bin))
print(pca.explained_variance_ratio_)

cumulative = []
count = 0
for i in range(90):
    count += pca.explained_variance_ratio_[i]
    cumulative.append(count)
plt.figure()
plt.plot(pca.explained_variance_ratio_)
plt.plot(pca.explained_variance_ratio_, 'x')
plt.savefig(path_data+'/../projection_data/pca_info.svg')
plt.figure()
plt.plot(cumulative)
plt.plot(cumulative, 'x')
plt.savefig(path_data+'/../projection_data/pca_cumulative.svg')
plt.show()