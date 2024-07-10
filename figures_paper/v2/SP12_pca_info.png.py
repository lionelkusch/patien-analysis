import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

path_data = os.path.dirname(os.path.realpath(__file__)) + '/../../data/'
avalanches_bin = np.load(path_data+'/avalanches_selected_patient.npy', allow_pickle=True)
pca_choice = 5
label_size = 12.0

pca = PCA(n_components=90)
pca.fit(np.concatenate(avalanches_bin))
print(pca.explained_variance_ratio_)

cumulative = []
count = 0
for i in range(90):
    count += pca.explained_variance_ratio_[i]
    cumulative.append(count)


plt.figure(figsize=(10, 5))
ax = plt.subplot(1, 3, 1)
ax.plot(np.arange(1, 91, 1), pca.explained_variance_ratio_, alpha=0.5)
ax.plot(np.arange(1, 91, 1), pca.explained_variance_ratio_, 'x')
ax.vlines(pca_choice, ymin=0.0, ymax=pca.explained_variance_ratio_[pca_choice-1], color='r', alpha=0.5)
ax.hlines(pca.explained_variance_ratio_[pca_choice-1], xmin=1, xmax=pca_choice, color='r', alpha=0.5)
ax.set_title('explain variance', {"fontsize": label_size}, weight='bold')
ax.set_xlabel('components of PCA', {"fontsize": label_size})

ax = plt.subplot(1, 3, 2)
ax.plot(np.arange(1, 91, 1), pca.explained_variance_ratio_, alpha=0.5)
ax.plot(np.arange(1, 91, 1), pca.explained_variance_ratio_, 'x')
ax.vlines(pca_choice, ymin=0.0, ymax=pca.explained_variance_ratio_[pca_choice-1], color='r', alpha=0.5)
ax.hlines(pca.explained_variance_ratio_[pca_choice-1], xmin=1, xmax=pca_choice, color='r', alpha=0.5)
ax.set_yscale("log")
ax.set_title('explain variance', {"fontsize": label_size}, weight='bold')
ax.set_xlabel('components of PCA', {"fontsize": label_size})

ax = plt.subplot(1, 3, 3)
ax.plot(np.arange(1, 91, 1), cumulative, alpha=0.5)
ax.plot(np.arange(1, 91, 1), cumulative, 'x')
ax.vlines(pca_choice, ymin=0.0, ymax=cumulative[pca_choice-1], color='r', alpha=0.5)
ax.hlines(cumulative[pca_choice-1], xmin=1, xmax=pca_choice, color='r', alpha=0.5)
ax.set_title('explain variance cumulative', {"fontsize": label_size}, weight='bold')
ax.set_xlabel('components of PCA', {"fontsize": label_size})

plt.subplots_adjust(left=0.05, right=0.99, top=0.93, bottom=0.1)
plt.savefig('figure/SP_12_pca_cumulative.png')
plt.show()
