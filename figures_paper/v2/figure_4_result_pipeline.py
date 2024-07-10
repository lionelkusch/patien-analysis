import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pipeline_phate_clustering.functions_helper.plot_brain import get_brain_mesh
from matplotlib.gridspec import GridSpec
from pipeline_phate_clustering.functions_helper.plot_brain_test import multiview_pyvista, get_region_select
from matplotlib.patches import Rectangle


label_size = 12.0
tickfont_size = 10.0
path = os.path.dirname(os.path.realpath(__file__)) + "/../../paper/result/default/"
kmeans_nb_cluster = 7
kmeans_seed = 123
PHATE_n_pca = 5
avalanches_pattern = np.load(path + "/avalanches.npy", allow_pickle=True)
Y_phate = np.load(path + "/Phate.npy")
cluster_phate = KMeans(n_clusters=kmeans_nb_cluster, random_state=kmeans_seed).fit_predict(Y_phate)
cluster_vector = []
for i in range(kmeans_nb_cluster):
        cluster_vector.append(np.mean(np.concatenate(avalanches_pattern)[np.where(cluster_phate == i)], axis=0)[:78])
cluster_vector = np.array(cluster_vector)
region_selects = get_region_select()
lhfaces, rhfaces, vertices, region_map = get_brain_mesh()
transitions = np.load(path + "/transition_all.npy")
colormap = plt.get_cmap('Accent', 7)

fig = plt.figure(figsize=(6.8, 5.))
# brain image
gs_brain = GridSpec(5, 8, figure=fig, height_ratios=[1., 1., 0.1, 1., 1.])
gs_brain_middle = GridSpec(3, 12, figure=fig, height_ratios=[1., 0.05, 1.])
axs1 = [fig.add_subplot(gs_brain[0, 0]), fig.add_subplot(gs_brain[1, 0]),
        fig.add_subplot(gs_brain[0, 1]), fig.add_subplot(gs_brain[1, 1]), fig.add_subplot(gs_brain_middle[0, 1])]
multiview_pyvista(axs1, vertices, lhfaces, rhfaces, region_map, cluster_vector[0], region_select=region_selects, cmap='plasma')
axs1[0].annotate('A', xy=(-0., 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
fig.patches.extend([Rectangle((0.0615, 0.56), 0.23, 0.42, figure=fig, clip_on=False, facecolor="none", lw=1.5, ec=colormap.colors[0],
                              transform=fig.transFigure)])
axs2 = [fig.add_subplot(gs_brain[0, 2]), fig.add_subplot(gs_brain[1, 2]),
        fig.add_subplot(gs_brain[0, 3]), fig.add_subplot(gs_brain[1, 3]), fig.add_subplot(gs_brain_middle[0, 4])]
multiview_pyvista(axs2, vertices, lhfaces, rhfaces, region_map, cluster_vector[1], region_select=region_selects, cmap='plasma')
axs2[0].annotate('B', xy=(-0., 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
fig.patches.extend([Rectangle((0.2965, 0.56), 0.23, 0.42, figure=fig, clip_on=False, facecolor="none", lw=1.5, ec=colormap.colors[1],
                              transform=fig.transFigure)])
axs3 = [fig.add_subplot(gs_brain[0, 4]), fig.add_subplot(gs_brain[1, 4]),
        fig.add_subplot(gs_brain[0, 5]), fig.add_subplot(gs_brain[1, 5]), fig.add_subplot(gs_brain_middle[0, 7])]
multiview_pyvista(axs3, vertices, lhfaces, rhfaces, region_map, cluster_vector[2], region_select=region_selects, cmap='plasma')
axs3[0].annotate('C', xy=(-0., 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
fig.patches.extend([Rectangle((0.5305, 0.56), 0.23, 0.42, figure=fig, clip_on=False, facecolor="none", lw=1.5, ec=colormap.colors[2],
                              transform=fig.transFigure)])
axs4 = [fig.add_subplot(gs_brain[0, 6]), fig.add_subplot(gs_brain[1, 6]),
        fig.add_subplot(gs_brain[0, 7]), fig.add_subplot(gs_brain[1, 7]), fig.add_subplot(gs_brain_middle[0, 10])]
multiview_pyvista(axs4, vertices, lhfaces, rhfaces, region_map, cluster_vector[3], region_select=region_selects, cmap='plasma')
axs4[0].annotate('D', xy=(-0., 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
fig.patches.extend([Rectangle((0.7645, 0.56), 0.23, 0.42, figure=fig, clip_on=False, facecolor="none", lw=1.5, ec=colormap.colors[3],
                              transform=fig.transFigure)])
axs5 = [fig.add_subplot(gs_brain[3, 0]), fig.add_subplot(gs_brain[4, 0]),
        fig.add_subplot(gs_brain[3, 1]), fig.add_subplot(gs_brain[4, 1]), fig.add_subplot(gs_brain_middle[2, 1])]
multiview_pyvista(axs5, vertices, lhfaces, rhfaces, region_map, cluster_vector[4], region_select=region_selects, cmap='plasma')
axs5[0].annotate('E', xy=(-0., 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
fig.patches.extend([Rectangle((0.0615, 0.04), 0.23, 0.42, figure=fig, clip_on=False, facecolor="none", lw=1.5, ec=colormap.colors[4],
                              transform=fig.transFigure)])
axs6 = [fig.add_subplot(gs_brain[3, 2]), fig.add_subplot(gs_brain[4, 2]),
        fig.add_subplot(gs_brain[3, 3]), fig.add_subplot(gs_brain[4, 3]), fig.add_subplot(gs_brain_middle[2, 4])]
multiview_pyvista(axs6, vertices, lhfaces, rhfaces, region_map, cluster_vector[5], region_select=region_selects, cmap='plasma')
axs6[0].annotate('F', xy=(-0., 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
fig.patches.extend([Rectangle((0.2965, 0.04), 0.23, 0.42, figure=fig, clip_on=False, facecolor="none", lw=1.5, ec=colormap.colors[5],
                              transform=fig.transFigure)])
axs7 = [fig.add_subplot(gs_brain[3, 4]), fig.add_subplot(gs_brain[4, 4]),
        fig.add_subplot(gs_brain[3, 5]), fig.add_subplot(gs_brain[4, 5]), fig.add_subplot(gs_brain_middle[2, 7])]
multiview_pyvista(axs7, vertices, lhfaces, rhfaces, region_map, cluster_vector[6], region_select=region_selects, cmap='plasma')
fig.patches.extend([Rectangle((0.5305, 0.04), 0.23, 0.42, figure=fig, clip_on=False, facecolor="none", lw=1.5, ec=colormap.colors[6],
                              transform=fig.transFigure)])
axs7[0].annotate('G', xy=(-0., 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)



ax = fig.add_axes([0.045, 0.25, 0.01, 0.5])
colbar = fig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=0., vmax=1.), cmap=plt.get_cmap('plasma')), cax=ax, orientation='vertical', )
colbar.set_ticks([-1., 0.0, 1.0])
colbar.set_ticklabels([-1.0, 0.0, 1.0])
colbar.ax.yaxis.set_tick_params(pad=0.1, labelsize=tickfont_size)
colbar.ax.set_ylabel('% activity in a cluster', {"fontsize": label_size}, labelpad=-12)
colbar.ax.yaxis.set_ticks_position('left')
colbar.ax.yaxis.set_label_position('left')
ax = fig.add_axes([0.4, 0.53, 0.2, 0.01])
colbar = fig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=0., vmax=7.), cmap=colormap), cax=ax, orientation='horizontal')
colbar.set_ticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
colbar.set_ticklabels([1, 2, 3, 4, 5, 6, 7])
colbar.ax.xaxis.set_tick_params(pad=0.1, labelsize=tickfont_size)
colbar.ax.set_xlabel('#cluster', {"fontsize": label_size}, labelpad=-2)


gs_transition = GridSpec(3, 5, figure=fig, width_ratios=[1.0, 1.0, 1.0, 1.0, 1.0], height_ratios=[1., 0.8, 0.1])
ax10 = fig.add_subplot(gs_transition[1, 4])
im_transition = ax10.imshow(transitions)
divider = make_axes_locatable(ax10)
cax = divider.append_axes("bottom", size="5%", pad=0.4)
colorbar_transition = fig.colorbar(im_transition, cax=cax, orientation='horizontal')
colorbar_transition.ax.xaxis.set_tick_params(pad=0.1, labelsize=tickfont_size)
colorbar_transition.ax.set_xlabel('% of transition', {"fontsize": label_size}, labelpad=2)
ax10.set_xticks([0.1, 3.1, 6.1])
ax10.set_xticklabels([1, 4, 7])
ax10.set_yticks([0.1, 3.1, 6.1])
ax10.set_yticklabels([1, 4, 7])
ax10.annotate('H', xy=(-0.2, 0.92), xycoords='axes fraction', weight='bold', fontsize=label_size)
ax10.set_title('CTM', {"fontsize": label_size})
ax10.set_ylabel('#cluster', {"fontsize": label_size}, labelpad=0)
ax10.set_xlabel('#cluster', {"fontsize": label_size}, labelpad=0)

plt.subplots_adjust(left=0.065, right=0.99, top=1., bottom=0.0, hspace=0., wspace=0.09)
plt.savefig('figure/figure_4_pre.png')
plt.savefig('figure/figure_4_pre.svg')
plt.show()