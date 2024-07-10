import os
import numpy as np
import matplotlib.pyplot as plt
from pipeline_phate_clustering.null_model_sensibility_analysis.get_result_null_model import get_color_map


path = os.path.dirname(os.path.realpath(__file__)) + "/../../paper/result/default/"
significatif = 0.05
nb_randomize = 100
label_size = 12.0
label_size_min = 8.0
max_nb_cluster = 15

# load value
data_null_model = {'transition': []}
for nb_rand in range(nb_randomize):
    data_null_model['transition'].append(np.load(path + "/transition" + str(nb_rand) + ".npy"))

data_patient = {'transition': np.load(path + "/transition.npy")}
nb_cluster = data_patient['transition'].shape[1]
nb_patient = data_patient['transition'].shape[0]
cmap_red_blue = get_color_map()
pvalue = np.sum(np.array(data_null_model['transition']) > data_patient['transition'], axis=0) / nb_randomize
significatif_high = pvalue > 1.0 - significatif
significatif_low = pvalue < significatif
significatif_all = np.logical_or(significatif_low, significatif_high)
pvalue = np.array([np.concatenate(data) for data in pvalue])
significatif_high = np.array([np.concatenate(data) for data in significatif_high])
significatif_low = np.array([np.concatenate(data) for data in significatif_low])
significatif_all = np.array([np.concatenate(data) for data in significatif_all])
order_transition = np.flip(np.argsort(np.sum(significatif_high, axis=0) - np.sum(significatif_low, axis=0)))
order_patient = np.flip(np.argsort(np.sum(significatif_all, axis=1)))
name_transition = []
for input in range(nb_cluster):
    for output in range(nb_cluster):
        name_transition.append('in:' + str(input) + ' out:' + str(output))



plt.figure(figsize=(5, 10))
ax = plt.gca()
im_3 = ax.imshow(pvalue[order_patient, :][:, order_transition].T, vmin=significatif, vmax=1 - significatif,
                  cmap=cmap_red_blue)
ax.set_yticks(range(nb_cluster * nb_cluster))
ax.set_yticklabels(labels=np.array(name_transition)[order_transition], fontsize=label_size_min)
ax.set_xticks(range(nb_patient))
ax.set_xticklabels(order_patient)
ax.set_xlabel('subject', fontdict={"fontsize": label_size})
plt.subplots_adjust(left=0.0, right=1., top=0.99, bottom=0.07)
plt.savefig('figure/SP_14_order_transition.png')

plt.show()