import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

path_data = os.path.dirname(os.path.realpath(__file__)) + '/../../data/'
avalanches_bin = np.load(path_data+'/avalanches_selected_patient.npy', allow_pickle=True)
pca_choice = 5
label_size = 12.0
titlefont_size = 12.0

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
ax.set_ylabel('explained variance', {"fontsize": label_size})
ax.set_xlabel('components of PCA', {"fontsize": label_size})

ax = plt.subplot(1, 3, 2)
ax.plot(np.arange(1, 91, 1), pca.explained_variance_ratio_, alpha=0.5)
ax.plot(np.arange(1, 91, 1), pca.explained_variance_ratio_, 'x')
ax.vlines(pca_choice, ymin=0.0, ymax=pca.explained_variance_ratio_[pca_choice-1], color='r', alpha=0.5)
ax.hlines(pca.explained_variance_ratio_[pca_choice-1], xmin=1, xmax=pca_choice, color='r', alpha=0.5)
ax.set_yscale("log")
ax.set_ylabel('logarithmic explained variance', {"fontsize": label_size})
ax.set_xlabel('components of PCA', {"fontsize": label_size})

ax = plt.subplot(1, 3, 3)
ax.plot(np.arange(1, 91, 1), cumulative, alpha=0.5)
ax.plot(np.arange(1, 91, 1), cumulative, 'x')
ax.vlines(pca_choice, ymin=0.0, ymax=cumulative[pca_choice-1], color='r', alpha=0.5)
ax.hlines(cumulative[pca_choice-1], xmin=1, xmax=pca_choice, color='r', alpha=0.5)
ax.set_ylabel('cumulative explained variance', {"fontsize": label_size})
ax.set_xlabel('components of PCA', {"fontsize": label_size})

plt.annotate('A', (0., 0.96), xycoords='figure fraction', fontsize=titlefont_size, weight='bold')
plt.annotate('B', (0.335, 0.96), xycoords='figure fraction', fontsize=titlefont_size, weight='bold')
plt.annotate('C', (0.675, 0.96), xycoords='figure fraction', fontsize=titlefont_size, weight='bold')

plt.subplots_adjust(left=0.07, right=0.99, top=0.99, bottom=0.1, wspace=0.28)
plt.savefig('figure_3/SP_2_pca_cumulative.png', dpi=600)
plt.show()
