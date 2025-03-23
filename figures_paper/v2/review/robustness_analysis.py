import matplotlib.pyplot as plt
import numpy as np
import os

path = os.path.dirname(os.path.realpath(__file__)) + "/../../../paper/result/default/"
titlefont_size = 12.0
tickfont_size = 12.0
labelfont_size = 12.0
letter_font_size = 12
pvalues_prob_transisiton = np.load(path+'/model_diagonal.npy', allow_pickle=True)

fig = plt.figure(figsize=(6.8, 5.6))
ax = plt.gca()
max_nb_cluster = 15
per_diag = np.array([pvalue['data_per_diagonal_all'] for pvalue in pvalues_prob_transisiton])*100
per_diag_rand = np.array([pvalue['shuffle_per_diagonal_all'] for pvalue in pvalues_prob_transisiton ])*100
ax.plot(range(2, max_nb_cluster), per_diag, 'b', label='data')
ax.plot(range(2, max_nb_cluster), per_diag_rand.mean(axis=1), 'g', label='null model 2')
ax.fill_between(range(2, max_nb_cluster),
                per_diag_rand.mean(axis=1) + per_diag_rand.std(axis=1),
                per_diag_rand.mean(axis=1) - per_diag_rand.std(axis=1), 'g', alpha=0.5)
ax.set_ylabel('% of significant transition', {"fontsize": labelfont_size}, labelpad=-2)
ax.set_ylim(ymin=0.20)
ax.set_xlabel('nb clusters', {"fontsize": labelfont_size}, labelpad=0)
ax.legend(fontsize=tickfont_size, handlelength=0.5, borderpad=0.2, labelspacing=0.1, loc='lower center')
ax.tick_params(which='both', labelsize=tickfont_size, pad=0)
ax.set_title('Significant\ndiagonal', {"fontsize": titlefont_size}, pad=0)
# plt.subplots_adjust(left=0.05, right=0.99, bottom=0.07, top=0.89, hspace=0.6)
plt.savefig('figure/figure_5_25.png', dpi=300)