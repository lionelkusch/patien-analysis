import matplotlib.pyplot as plt
import numpy as np
import os
from pipeline_phate_clustering.functions_helper.get_entropy import get_entropy, get_entropy_pca

label_size = 12.0
tickfont_size = 10.0
label_col_size = 8.0
linewidth = 0.5

result = {}
path_root = os.path.dirname(os.path.realpath(__file__)) + '/../../../'
path_phate = path_root + "/paper/result/default/"
for name, path in [
    ('PHATE\navalanche patterns', path_root + "/paper/result/default/"),
    ('PHATE\navalanches', path_root + "/paper/result/no_avalanche/avalanches_3/"),
    ('PHATE\ndata normalized', path_root + "/paper/result/no_avalanche/data_normalized_euclidean_2/"), # different decay
    ('PHATE\ndata', path_root + "/paper/result/no_avalanche/data_euclidean/"),
    ('Spectral\nclustering 7', path_root + "/paper/result/spectral_cosine/spectral_cosine7/"),
]:
    result[name] = get_entropy(path, name)
for name, path in [
    ('PCA\navalanches pattern', path_root + "/paper/result/PCA/avalanches_pattern/"),
    ('PCA\navalanches', path_root + "/paper/result/PCA/avalanches/"),
    ('PCA\ndata normalized', path_root + "/paper/result/PCA/data_normalized/"),
    ('PCA\ndata', path_root + "/paper/result/PCA/data/"),
]:
    result[name] = get_entropy_pca(path, name)

values = np.array(list(result.values()))
plt.figure(figsize=(5, 4))
plt.plot(values[0], 'xr'),
plt.plot(np.arange(1, len(values)), values[1:], 'xb'),
plt.xticks(np.arange(0, values.shape[0]), result.keys(), rotation=90 )
plt.ylabel('entropy', {"fontsize": label_size})
plt.tick_params('both', labelsize=tickfont_size)
plt.subplots_adjust(top=0.98, bottom=0.36, left=0.11, right=0.99 )
# plt.show()
plt.savefig('figure/entropy_pca_phate.png')

# spectral
result = {}
path_root = os.path.dirname(os.path.realpath(__file__)) + '/../../../'
path_phate = path_root + "/paper/result/default/"
for name, path in [
    ('PHATE', path_root + "/paper/result/default/"),
    ('3', path_root+"/paper/result/spectral_cosine/spectral_cosine3"),
    ('4', path_root + "/paper/result/spectral_cosine/spectral_cosine4"),
    ('5', path_root + "/paper/result/spectral_cosine/spectral_cosine5"),
    ('6', path_root + "/paper/result/spectral_cosine/spectral_cosine6"),
    ('7', path_root + "/paper/result/spectral_cosine/spectral_cosine7"),
    ('8', path_root + "/paper/result/spectral_cosine/spectral_cosine8"),
    ('9', path_root + "/paper/result/spectral_cosine/spectral_cosine9"),
    ('10', path_root + "/paper/result/spectral_cosine/spectral_cosine10"),
    ('11', path_root + "/paper/result/spectral_cosine/spectral_cosine10"),
    ('12', path_root + "/paper/result/spectral_cosine/spectral_cosine10"),
    ('13', path_root + "/paper/result/spectral_cosine/spectral_cosine10"),
]:
    result[name] = get_entropy(path, name)

values = np.array(list(result.values()))
plt.figure(figsize=(5, 4))
plt.plot(values[0], 'xr'),
plt.plot(np.arange(1, len(values)), values[1:], 'xb'),
plt.xticks(np.arange(0, values.shape[0]), result.keys(), rotation=90)
plt.ylabel('entropy', {"fontsize": label_size})
plt.xlabel('    spectral clustering', {"fontsize": label_size})
plt.tick_params('both', labelsize=tickfont_size)
plt.subplots_adjust(top=0.98, bottom=0.20, left=0.11, right=0.99)
# plt.show()
plt.savefig('figure/entropy_spectral.png')



# th
th_range = np.arange(1.0, 6.0, 0.1)
entropy_list = []
for th in th_range:
    path_saving = path_phate + '/../sensibility_analysis/PHATE_th_'+str(th)
    entropy_list.append(get_entropy(path_saving, 'th_' + str(th)))

plt.figure(figsize=(5, 4))
for th, entropy in zip(th_range, entropy_list):
    if np.abs(th - 3.0) < 1e-4:
        plt.plot(th, entropy, 'rx')
    else:
        plt.plot(th, entropy, 'bx')
plt.ylabel('entropy', {"fontsize": label_size})
plt.xlabel("threshold of the normalised data", {"fontsize": label_size})
plt.tick_params('both', labelsize=tickfont_size)
plt.subplots_adjust(top=0.98, bottom=0.11, left=0.13, right=0.99)
plt.savefig('figure/entropy_th.png')

