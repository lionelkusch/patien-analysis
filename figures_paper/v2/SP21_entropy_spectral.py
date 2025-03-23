import numpy as np
import matplotlib.pyplot as plt
import os
from pipeline_phate_clustering.functions_helper.get_entropy import get_entropy

titlefont_size = 12.0
tickfont_size = 10.0
labelfont_size = 11.0
letter_font_size = 12

# spectral
result = {}
path_root = os.path.dirname(os.path.realpath(__file__)) + '/../../'
path_phate = path_root + "/paper/result/default/"
for name, path in [
    ('PHATE (7 clusters)', path_root + "/paper/result/default/"),
    ('3 clusters', path_root+"/paper/result/spectral_cosine/spectral_cosine3"),
    ('4 clusters', path_root + "/paper/result/spectral_cosine/spectral_cosine4"),
    ('5 clusters', path_root + "/paper/result/spectral_cosine/spectral_cosine5"),
    ('6 clusters', path_root + "/paper/result/spectral_cosine/spectral_cosine6"),
    ('7 clusters', path_root + "/paper/result/spectral_cosine/spectral_cosine7"),
    ('8 clusters', path_root + "/paper/result/spectral_cosine/spectral_cosine8"),
    ('9 clusters', path_root + "/paper/result/spectral_cosine/spectral_cosine9"),
    ('10 clusters', path_root + "/paper/result/spectral_cosine/spectral_cosine10"),
    ('11 clusters', path_root + "/paper/result/spectral_cosine/spectral_cosine10"),
    ('12 clusters', path_root + "/paper/result/spectral_cosine/spectral_cosine10"),
    ('13 clusters', path_root + "/paper/result/spectral_cosine/spectral_cosine10"),
]:
    result[name] = get_entropy(path, name)

values = np.array(list(result.values()))
plt.figure(figsize=(6.4, 4.2))
plt.plot(values[0], 'xr'),
plt.plot(np.arange(1, len(values)), values[1:], 'xb'),
plt.xticks(np.arange(0, values.shape[0]), result.keys(), rotation=90 )
plt.ylabel('entropy', {"fontsize": labelfont_size})
plt.xlabel('Spectral clustering', {"fontsize": labelfont_size}, labelpad= -5)
plt.tick_params('both', labelsize=tickfont_size)
plt.subplots_adjust(top=0.98, bottom=0.36, left=0.11, right=0.99)
plt.savefig('figure_3/SP_19_entropy_spectrale_phate.png', dpi=600)
plt.show()
