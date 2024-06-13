import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

path = os.path.dirname(os.path.realpath(__file__)) + "/../../paper/result/default/figure/"
nb_clusters = range(2, 15)
fig, axs = plt.subplots(len(nb_clusters), max(nb_clusters), figsize=(20, 20),
                        gridspec_kw={'left': 0.01, 'bottom': 0.01, 'right': 0.99, 'top': 0.99, 'wspace': 0.01,
                                     'hspace': 0.01})
fig_2d, axs_2d = plt.subplots(len(nb_clusters), max(nb_clusters), figsize=(20, 20),
                        gridspec_kw={'left': 0.01, 'bottom': 0.01, 'right': 0.99, 'top': 0.99, 'wspace': 0.01,
                                     'hspace': 0.01})
for x, nb_cluster in enumerate(nb_clusters):
    for y, k in enumerate(range(nb_cluster)):
        print(x, y)
        path_fig = path + 'k_' + str(nb_cluster) + 'cluster_' + str(k) + '.png'
        axs[x, y].imshow(mpimg.imread(path_fig))
        axs[x, y].set_xticks([])
        axs[x, y].set_yticks([])
        path_fig = path + 'k_' + str(nb_cluster) + 'cluster2D_' + str(k) + '.png'
        axs_2d[x, y].imshow(mpimg.imread(path_fig))
        axs_2d[x, y].set_xticks([])
        axs_2d[x, y].set_yticks([])

for x in range(len(nb_clusters)):
    for y in range(max(nb_clusters)):
        axs[x, y].axis('off')
        axs_2d[x, y].axis('off')
fig.savefig(path+'/all_cluster_vector.png', dpi=600)
fig_2d.savefig(path+'/all_cluster2D_vector.png', dpi=600)
