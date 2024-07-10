import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pipeline_phate_clustering.null_model_sensibility_analysis.get_result_null_model import get_color_map

nb_randomize = 100
significatif = 0.05
tickfont_size = 10.0
label_size = 12
path_saving = os.path.dirname(os.path.realpath(__file__)) + '/../../paper/result/default/'
letter = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
cmap_red_blue = get_color_map()

# load value
data_null_model = {'transition': [],
                   'histogram': [],
                   'cluster_patient': []
                   }
for nb_rand in range(nb_randomize):
    data_null_model['transition'].append(np.load(path_saving + "/transition" + str(nb_rand) + ".npy"))
    data_null_model['histogram'].append(np.load(path_saving + "/histograms" + str(nb_rand) + ".npy"))
    data_null_model['cluster_patient'].append(
        np.load(path_saving + "/cluster_patient_data" + str(nb_rand) + ".npy", allow_pickle=True))

data_patient = {'transition': np.load(path_saving + "/transition.npy"),
                'histogram': np.load(path_saving + "/histograms_patient.npy"),
                'cluster_patient': np.load(path_saving + "/cluster_patient_data.npy", allow_pickle=True)
                }

# compute the pvalue of each transition
pvalue = np.sum(np.array(data_null_model['transition']) > data_patient['transition'], axis=0) / nb_randomize
significatif_high = pvalue > 1.0 - significatif
significatif_low = pvalue < significatif
significatif_all = np.logical_or(significatif_low, significatif_high)

transistion_matrix = np.array(data_patient['transition'])
transistion_matrix[np.logical_not(significatif_all)] = np.NAN
std_significant_transition = np.nanstd(transistion_matrix, axis=0)
mean_significant_transition = np.nanmean(transistion_matrix, axis=0)

# load value
all_data_null_model = {'transition': []}
for nb_rand in range(nb_randomize):
    all_data_null_model['transition'].append(np.load(path_saving + "/transition_all" + str(nb_rand) + ".npy"))
transition_all = np.load(path_saving + "/transition_all.npy")
all_pvalue = np.sum(np.array(all_data_null_model['transition']) > transition_all, axis=0) / nb_randomize
all_significatif_high = all_pvalue > 1.0 - significatif
all_significatif_low = all_pvalue < significatif
all_significatif_all = np.logical_or(all_significatif_low, all_significatif_high)

fig = plt.figure(figsize=(15, 5))
for index, (title, data) in enumerate([('mean', mean_significant_transition),
                                       ('std', std_significant_transition),
                                       ('coefficient of variation', std_significant_transition / mean_significant_transition),
                                       ]):
    ax = plt.subplot(131 + index)
    ax.set_title(title, {"fontsize": label_size})
    im = ax.imshow(data)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    for (j, i), label in np.ndenumerate(data):
        ax.text(i, j, np.around(label, 2), ha='center', va='center')
    ax.set_xlabel('# cluster', {"fontsize": label_size})
    ax.set_ylabel('# cluster', {"fontsize": label_size})
    ax.set_xticks(np.arange(7))
    ax.set_xticklabels(np.arange(7)+1)
    ax.set_yticks(np.arange(7))
    ax.set_yticklabels(np.arange(7)+1)
    ax.annotate(letter[index], xy=(-0.1, 0.9), xycoords='axes fraction', weight='bold', fontsize=label_size)
plt.subplots_adjust(top=1.0, bottom=0.0, left=0.03, right=0.97, wspace=0.21)
plt.savefig('figure/SP_6_transition_all_variability.png')

fig = plt.figure(figsize=(10, 8))
ax1 = plt.subplot(4, 5, 1)
im = ax1.imshow(all_significatif_all, cmap=cmap_red_blue)
ax1.grid()
ax1.set_yticks(np.arange(7)+0.5)
ax1.set_yticklabels([])
ax1.set_xticks(np.arange(7)+0.5)
ax1.set_xticklabels([])
ax1.set_xlabel('# cluster', {"fontsize": label_size}, labelpad=0)
ax1.set_ylabel('# cluster', {"fontsize": label_size}, labelpad=0)
ax1.annotate(letter[0], xy=(-0.15, 0.9), xycoords='axes fraction', weight='bold', fontsize=label_size)
for nb_patient, transition_patient_significatif in enumerate(pvalue):
    ax = plt.subplot(4, 5, nb_patient+2)
    ax.imshow(transition_patient_significatif, cmap=cmap_red_blue)
    ax.set_xlabel('# cluster', {"fontsize": label_size}, labelpad=0)
    ax.set_ylabel('# cluster', {"fontsize": label_size}, labelpad=0)
    ax.set_xticks(np.arange(7)+0.5)
    ax.set_xticklabels([])
    ax.set_yticks(np.arange(7)+0.5)
    ax.set_yticklabels([])
    ax.grid()
    ax.annotate(letter[nb_patient+1], xy=(-0.15, 0.9), xycoords='axes fraction', weight='bold', fontsize=label_size)
ax = plt.subplot(5, 5, nb_patient+3+5)
ax.imshow([[-1.0], [0.0], [1.0]], vmin=-1.0, vmax=1.0, cmap=cmap_red_blue, origin='lower')
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(['significant\ntransition', 'no\nsignificant', 'significant\nno transition'])
ax.set_xticks([])
ax.set_xlim(xmin=0, xmax=0.5)
plt.subplots_adjust(top=1.0, bottom=0.04, left=0.04, right=0.99, wspace=0.23)
plt.savefig('figure/SP_7_transition_significatif.png')


fig = plt.figure(figsize=(10, 8))
ax1 = plt.subplot(4, 5, 1)
im = ax1.imshow(transition_all, vmin=0., vmax=0.6)
ax1.grid()
ax1.set_yticks(np.arange(7)+0.5)
ax1.set_yticklabels([])
ax1.set_xticks(np.arange(7)+0.5)
ax1.set_xticklabels([])
ax1.set_xlabel('# cluster', {"fontsize": label_size}, labelpad=0)
ax1.set_ylabel('# cluster', {"fontsize": label_size}, labelpad=0)
ax1.annotate(letter[0], xy=(-0.15, 0.9), xycoords='axes fraction', weight='bold', fontsize=label_size)
for nb_patient, transition_patient in enumerate(data_patient['transition']):
    ax = plt.subplot(4, 5, nb_patient+2)
    im = ax.imshow(transition_patient, vmin=0., vmax=0.6)
    ax.set_xlabel('# cluster', {"fontsize": label_size}, labelpad=0)
    ax.set_ylabel('# cluster', {"fontsize": label_size}, labelpad=0)
    # for (j, i), label in np.ndenumerate(transition_patient):
    #     ax.text(i, j, np.around(label, 2), ha='center', va='center')
    ax.set_xticks(np.arange(7)+0.5)
    ax.set_xticklabels([])
    ax.set_yticks(np.arange(7)+0.5)
    ax.set_yticklabels([])
    ax.grid()
    ax.annotate(letter[nb_patient+1], xy=(-0.15, 0.9), xycoords='axes fraction', weight='bold', fontsize=label_size)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
colorbar = fig.colorbar(im, cax=cax)
colorbar.ax.yaxis.set_tick_params(pad=0.1, labelsize=tickfont_size)
colorbar.ax.set_ylabel('% of transition', {"fontsize": label_size}, labelpad=2)
plt.subplots_adjust(top=1.0, bottom=0.04, left=0.04, right=0.99, wspace=0.23)
plt.savefig('figure/SP_8_transition_patient.png')

plt.show()