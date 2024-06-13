import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pipeline_phate_clustering.null_model_sensibility_analysis.get_result_null_model import get_color_map


path = os.path.dirname(os.path.realpath(__file__)) + "/../../paper/result/default/"

transitions = np.load(path + "/transition_all.npy")
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


# load value
data_null_model_all = {'transition': []}
for nb_rand in range(nb_randomize):
    data_null_model_all['transition'].append(np.load(path + "/transition_all" + str(nb_rand) + ".npy"))

data_patient_all = {'transition': np.load(path + "/transition_all.npy")}
nb_cluster_all = data_patient_all['transition'].shape[1]
pvalue_all = np.sum(np.array(data_null_model_all['transition']) > data_patient_all['transition'], axis=0) / nb_randomize
significatif_high_all = pvalue_all > 1.0 - significatif
significatif_low_all = pvalue_all < significatif
significatif_all_all = np.logical_or(significatif_low_all, significatif_high_all)

pvalues_prob_transisiton = np.load(path+'/model_diagonal.npy', allow_pickle=True)
diag_per_lows, diag_per_nos, diag_per_highs, no_diag_per_lows, no_diag_per_nos, no_diag_per_highs =\
    np.load(path+'/model_diagonalsignificatif.npy')

fig = plt.figure(figsize=(6.8, 6.8))
gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1.3])
ax1 = fig.add_subplot(gs[0, 0])
im_1 = ax1.imshow(transitions)
ax1.set_xticks([0.1, 3.1, 6.1])
ax1.set_xticklabels([0, 3, 6])
ax1.set_yticks([0.1, 3.1, 6.1])
ax1.set_yticklabels([0, 3, 6])
ax1.annotate('A', xy=(-0.1, 1.03), xycoords='axes fraction', weight='bold', fontsize=label_size)
divider_1 = make_axes_locatable(ax1)
cax_1 = divider_1.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im_1, cax=cax_1)

ax2 = fig.add_subplot(gs[0, 1])
im_2 = ax2.imshow(pvalue_all, vmin=significatif, vmax=1 - significatif, cmap=cmap_red_blue)
ax2.set_xticks([0.1, 3.1, 6.1])
ax2.set_xticklabels([0, 3, 6])
ax2.set_yticks([0.1, 3.1, 6.1])
ax2.set_yticklabels([0, 3, 6])
ax2.annotate('B', xy=(-0.1, 1.03), xycoords='axes fraction', weight='bold', fontsize=label_size)

ax3 = fig.add_subplot(gs[:, 2])
im_3 = ax3.imshow(pvalue[order_patient, :][:, order_transition].T, vmin=significatif, vmax=1 - significatif,
                  cmap=cmap_red_blue)
ax3.set_yticks(range(nb_cluster * nb_cluster))
ax3.set_yticklabels(labels=np.array(name_transition)[order_transition], fontsize=label_size_min)
ax3.set_xticks(range(nb_patient)[0::5])
ax3.set_xticklabels(order_patient[0::5])
ax3.set_xlabel('subject', fontdict={"fontsize": label_size})
ax3.annotate('E', xy=(-0.1, 1.0), xycoords='axes fraction', weight='bold', fontsize=label_size)


ax4 = fig.add_subplot(gs[1, 0])
per_diag = [pvalue['data_per_diagonal_all'] for pvalue in pvalues_prob_transisiton ]
per_diag_rand = np.array([pvalue['shuffle_per_diagonal_all'] for pvalue in pvalues_prob_transisiton ])
ax4.plot(range(2, max_nb_cluster), per_diag, 'b', label='data')
ax4.plot(range(2, max_nb_cluster), per_diag_rand.mean(axis=1), 'g', label='null model')
ax4.fill_between(range(2, max_nb_cluster),
                 per_diag_rand.mean(axis=1) + per_diag_rand.std(axis=1),
                 per_diag_rand.mean(axis=1) - per_diag_rand.std(axis=1), 'g', alpha=0.5)
ax4.annotate('C', xy=(-0.1, 1.03), xycoords='axes fraction', weight='bold', fontsize=label_size)

ax5 = fig.add_subplot(gs[1, 1])
index_of_cluster_nb = list(range(2, max_nb_cluster))
for index, data in enumerate(np.array([diag_per_lows, diag_per_nos, diag_per_highs,
                                     no_diag_per_lows, no_diag_per_nos, no_diag_per_highs]).T):
    ax5.bar(index_of_cluster_nb[index], data[0], color=['b'], width=0.4)
    ax5.bar(index_of_cluster_nb[index], data[1], color=['w'], width=0.4, bottom=data[0])
    ax5.bar(index_of_cluster_nb[index], data[2], color=['r'], width=0.4, bottom=data[0]+data[1])
    ax5.bar(index_of_cluster_nb[index] + 0.3, data[3], color='b', width=0.4)
    ax5.bar(index_of_cluster_nb[index] + 0.3, data[4], color='w', width=0.4, bottom=data[3])
    ax5.bar(index_of_cluster_nb[index] + 0.3, data[5], color='r', width=0.4, bottom=data[3] + data[4])
ax5.annotate('D', xy=(-0.1, 1.03), xycoords='axes fraction', weight='bold', fontsize=label_size)

plt.subplots_adjust(left=0.05, right=0.995, top=0.995, bottom=0.05, wspace=0.38)
plt.savefig('figure/figure_4_pre.png')
# plt.show()
