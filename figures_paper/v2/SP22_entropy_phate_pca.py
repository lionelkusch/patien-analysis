import numpy as np
import matplotlib.pyplot as plt
import os
from pipeline_phate_clustering.functions_helper.get_entropy import get_entropy, get_entropy_pca

titlefont_size = 12.0
tickfont_size = 10.0
labelfont_size = 11.0
letter_font_size = 12

# spectral
result = {}
path_root = os.path.dirname(os.path.realpath(__file__)) + '/../../'
path_phate = path_root + "/paper/result/default/"
for name, path in [
    ('PHATE\navalanche patterns', path_root + "/paper/result/default/"),
    ('PHATE\navalanches', path_root + "/paper/result/no_avalanche/avalanches_3/"),
    ('PHATE\nsource data\nnormalised', path_root + "/paper/result/no_avalanche/data_normalized_euclidean_2/"), # different decay
    ('PHATE\nsource data', path_root + "/paper/result/no_avalanche/data_euclidean/"),
    ('Spectral clustering\n(7 clusters)', path_root + "/paper/result/spectral_cosine/spectral_cosine7/"),
]:
    result[name] = get_entropy(path, name)
for name, path in [
    ('PCA\navalanches pattern', path_root + "/paper/result/PCA/avalanches_pattern/"),
    ('PCA\navalanches', path_root + "/paper/result/PCA/avalanches/"),
    ('PCA\nsource data\nnormalised', path_root + "/paper/result/PCA/data_normalized/"),
    ('PCA\nsource data', path_root + "/paper/result/PCA/data/"),
]:
    result[name] = get_entropy_pca(path, name)


values = np.array(list(result.values()))
plt.figure(figsize=(6.4, 4.2))
plt.plot(values[0], 'xr'),
plt.plot(np.arange(1, len(values)), values[1:], 'xb'),
plt.xticks(np.arange(0, values.shape[0]), result.keys(), rotation=90 )
plt.ylabel('entropy', {"fontsize": labelfont_size})
plt.tick_params('both', labelsize=tickfont_size)
plt.subplots_adjust(top=0.98, bottom=0.36, left=0.11, right=0.99)
plt.savefig('figure_3/SP_20_entropy_spectrale_phate.png', dpi=600)
plt.show()
