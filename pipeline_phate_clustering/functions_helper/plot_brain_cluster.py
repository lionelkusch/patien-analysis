import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pipeline_phate_clustering.functions_helper.plot_brain import get_brain_mesh
from matplotlib.gridspec import GridSpec
from pipeline_phate_clustering.functions_helper.plot_brain_test import multiview_pyvista, get_region_select

region_selects = get_region_select()
lhfaces, rhfaces, vertices, region_map = get_brain_mesh()

def plot_brain_cluster(cluster_vector, path, name):
        fig = plt.figure(figsize=(3, 3))
        # brain image
        gs_brain = GridSpec(2, 2, figure=fig, height_ratios=[1., 1.])
        gs_brain_middle = GridSpec(3, 3, figure=fig, height_ratios=[0.8, 1., 0.8])
        axs1 = [fig.add_subplot(gs_brain[0, 0]), fig.add_subplot(gs_brain[1, 0]),
                fig.add_subplot(gs_brain[0, 1]), fig.add_subplot(gs_brain[1, 1]), fig.add_subplot(gs_brain_middle[1, 1])]
        multiview_pyvista(axs1, vertices, lhfaces, rhfaces, region_map, cluster_vector, region_select=region_selects, cmap='plasma')
        # plt.show()
        plt.savefig(path+name+'.png')
        plt.close('all')

if __name__ == '__main__':

        path = os.path.dirname(os.path.realpath(__file__)) + "/../../paper/result/"
        path_figures = path+'/spectral_cosine/figures/'
        path_sp = path+'/spectral_cosine/'
        avalanches_pattern = np.load(path + "/default/avalanches.npy", allow_pickle=True)

        kmeans_nb_cluster = 7
        kmeans_seed = 123
        # Y_phate = np.load(path + "/default/Phate.npy")
        # cluster_phate = KMeans(n_clusters=kmeans_nb_cluster, random_state=kmeans_seed).fit_predict(Y_phate)
        # for i in range(kmeans_nb_cluster):
        #         cluster_vector = np.mean(np.concatenate(avalanches_pattern)[np.where(cluster_phate == i)], axis=0)[:78]
        #         plot_brain_cluster(cluster_vector, path_figures, 'phate_'+str(i))
        path_all = path + "/all_subject_melbourne/"
        avalanches_pattern_all = np.load(path_all + "avalanches.npy", allow_pickle=True)
        Y_phate_all = np.load(path_all + "/Phate.npy")
        cluster_phate_all = KMeans(n_clusters=kmeans_nb_cluster, random_state=kmeans_seed).fit_predict(Y_phate_all)
        for i in range(kmeans_nb_cluster):
                cluster_vector = np.mean(np.concatenate(avalanches_pattern_all)[np.where(cluster_phate_all == i)], axis=0)[:78]
                plot_brain_cluster(cluster_vector, path_figures, 'phate_all_'+str(i))


        # for i in range(3, 15):
        #         data = np.concatenate(np.load(path_sp+'/spectral_cosine'+str(i)+'/cluster_patient_data.npy', allow_pickle=True))
        #         for j in range(i):
        #                 cluster_vector = np.mean(np.concatenate(avalanches_pattern)[np.where(data == j)], axis=0)[:78]
        #                 plot_brain_cluster(cluster_vector, path_figures, 'sp_'+str(i)+'_cl_'+str(j))