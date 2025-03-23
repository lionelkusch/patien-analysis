import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import entropy
import scipy.io as io
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter

label_size = 12.0
tickfont_size = 10.0
label_col_size = 8.0
linewidth = 0.5

result = {}
path_root = os.path.dirname(os.path.realpath(__file__)) + '/../../../'
path_data = path_root + "/paper/result/default/"
path_data_null_2 = path_root + "/paper/result/default/null_model/"
nb_randomize_1 = 10000
nb_randomize_2 = 100


sum_tot = 1 #np.sum(data_phate)/7
data_phate = np.load(path_data + "/histograms_region.npy")
entropy_phate = [entropy(i/sum_tot) for i in data_phate]
data_spectral = np.load(path_root + "/paper/result/spectral_cosine/spectral_cosine7/histograms_region.npy")
entropy_spectral = [entropy(i/sum_tot) for i in data_spectral]
data_pca = io.loadmat(path_root + "/paper/result/PCA/avalanches_pattern/vector_cluster.mat")['cluster_vector']
entropy_pca = [entropy(i/sum_tot) for i in data_pca]
data_null_model_1 = []
entropy_data_null_1 = []
for nb_rand in range(nb_randomize_1):
    data_null_model_1 = np.load(path_data + "/histograms_region_" + str(nb_rand) + ".npy")
    for i in range(data_null_model_1.shape[0]):
        entropy_data_null_1.append(entropy(data_null_model_1[i, :]/sum_tot))

data_null_model_2 = []
entropy_data_null_2 = []
for nb_rand in range(nb_randomize_2):
    data_null_model_2 = np.load(path_data_null_2 + str(nb_rand) + "_histograms_region.npy")
    for i in range(data_null_model_2.shape[0]):
        entropy_data_null_2.append(entropy(data_null_model_2[i, :]/sum_tot))

fig, axs = plt.subplots(4, 1, figsize=(5, 4), gridspec_kw={'height_ratios': [1, 0.1, 0.1, 0.1]})
for name, color, data in [('null model 1', 'black', entropy_data_null_1),
             ('null model 3', 'blue', entropy_data_null_2)]:
    hist, bin_edges = np.histogram(data, bins=20)
    hist = hist / len(data)
    hist = np.concatenate(([0], hist, [0]))
    bin_edges = np.concatenate(([3.3], bin_edges))
    axs[0].step(bin_edges, hist, color=color, label=name)
axs[0].set_ylim(ymin=0)
axs[0].yaxis.set_major_formatter(PercentFormatter(1))
axs[0].legend()
axs[0].tick_params('both', pad=1)
axs[0].set_xticks([])
axs[0].set_xlim(xmin=3, xmax=5)

for i in entropy_phate:
    axs[1].vlines(i, ymin=0.0, ymax=1.0)
axs[1].set_ylim(ymin=0, ymax=1)
axs[1].set_xlim(xmin=3, xmax=5)
axs[1].set_ylabel('Phate')
axs[1].set_xticks([])

for i in entropy_spectral:
    axs[2].vlines(i, ymin=0.0, ymax=1.0)
axs[2].set_ylim(ymin=0, ymax=1)
axs[2].set_xlim(xmin=3, xmax=5)
axs[2].set_ylabel('Sp Cl\n')
axs[2].set_xticks([])

for i in entropy_pca:
    axs[3].vlines(i, ymin=0.0, ymax=1.0)
axs[3].set_ylim(ymin=0, ymax=1)
axs[3].set_xlabel('entropy', {"fontsize": label_size}, labelpad=1)
axs[3].set_xlim(xmin=3, xmax=5)
axs[3].set_ylabel('PCA')

plt.subplots_adjust(top=0.95, right=0.95)
plt.savefig('figure/entropy_null_model.png')
plt.show()
# for index, i in enumerate(entropy_phate):
#     if index == 0 or index==2:
#         arr_1 = mpatches.FancyArrowPatch((i, 0.041),
#                                          (i, 0.0),
#                                          color='r',
#                                          arrowstyle='->,head_width=.15', mutation_scale=20)
#         arr_1.set_clip_on(False)
#         plt.gca().add_patch(arr_1)
#         plt.annotate('#' + str(index), (0.5, 1.3), xycoords=arr_1, ha='center', va='bottom', annotation_clip=False,
#                      color='red')
#         plt.annotate(str(np.around(i, decimals=2)), (0.5, 1.0), xycoords=arr_1, ha='center', va='bottom',
#                      annotation_clip=False, color='red')
#     else:
#         arr_1 = mpatches.FancyArrowPatch((i, -0.015),
#                                          (i, 0.0),
#                                          color='r',
#                                          arrowstyle='->,head_width=.15', mutation_scale=20)
#         arr_1.set_clip_on(False)
#         plt.gca().add_patch(arr_1)
#         plt.annotate('#'+str(index), (0.5, -1.4), xycoords=arr_1, ha='center', va='bottom', annotation_clip=False, color='red')
#         plt.annotate(' '+str(np.around(i, decimals=2)), (0.5, -0.8), xycoords=arr_1, ha='center', va='bottom', annotation_clip=False, color='red')



