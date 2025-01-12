import matplotlib.pyplot as plt
import numpy as np

def plot_compare_cluster(cluster_1, order_cluster_1, cluster_2, order_cluster_2,
                         title_1, title_2, title_3='difference', vmin=0.0, vmax=1.0, fontsize=2, cmap='viridis', cmap_diff='seismic'):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 10))
        im = ax1.imshow(cluster_1, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.colorbar(im, ax=ax1)
        ax1.set_yticks(np.arange(0, len(order_cluster_1)))
        ax1.set_yticklabels(order_cluster_1)
        ax1.set_title(title_1)
        im = ax2.imshow(cluster_2, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.colorbar(im, ax=ax2)
        ax2.set_yticks(np.arange(0, len(order_cluster_2)))
        ax2.set_yticklabels(order_cluster_2)
        ax2.set_title(title_2)
        if cluster_2.shape[0] == cluster_1.shape[0]:
                im = ax3.imshow(cluster_1-cluster_2, vmin=-vmax, vmax=vmax, cmap=cmap_diff)
        elif cluster_2.shape[0] < cluster_1.shape[0]:
                im = ax3.imshow(cluster_1[-cluster_2.shape[0]:]-cluster_2, vmin=-vmax, vmax=vmax, cmap=cmap_diff)
        elif cluster_2.shape[0] > cluster_1.shape[0]:
                im = ax3.imshow(cluster_1-cluster_2[-cluster_1.shape[0]:], vmin=-vmax, vmax=vmax, cmap=cmap_diff)
        plt.colorbar(im, ax=ax3)
        ax3.set_title(title_3)


if __name__ == '__main__':
        import os
        path_root = os.path.dirname(os.path.realpath(__file__)) + '/../../'
        path_phate = path_root + "/paper/result/default/"
        avalanches_bin = np.load(path_phate + '/avalanches.npy', allow_pickle=True)
        histograms_region_phate = np.load(path_phate + "/histograms_region.npy")
        cluster_vector_phate = histograms_region_phate / histograms_region_phate.max(axis=0).reshape(1, len(avalanches_bin[0][0]))
        order_phate = np.argsort(np.sum(histograms_region_phate, axis=1))
        cluster_vector_phate = cluster_vector_phate[order_phate]

        for i in range(3, 15):
                # spectral
                path_spherical = path_root + "/paper/result/spectral_cosine/"
                histograms_region_spherical = np.load(path_spherical + "/spectral_cosine"+str(i)+"/histograms_region.npy")
                cluster_vector_spherical = histograms_region_spherical / histograms_region_spherical.max(axis=0).reshape(1, len(avalanches_bin[0][0]))
                order_spherical = np.argsort(np.sum(histograms_region_spherical, axis=1))
                cluster_vector_spherical = cluster_vector_spherical[order_spherical]
                print('cluster ', i, 'order', order_spherical)

                plot_compare_cluster(cluster_vector_phate, order_phate, cluster_vector_spherical, order_spherical,
                                     'Phate cluster', 'Spectral cluster', 'cluster Phate - cluster spectral')
                plt.savefig(path_spherical+'/figures/compare_cluster'+str(i)+'_phate.png')

        # # all subject
        # path_all = path_root + "/paper/result/all_subject_melbourne/"
        # avalanches_bin_all = np.load(path_all + '/avalanches.npy', allow_pickle=True)
        # histograms_region_all = np.load(path_all + "/histograms_region.npy")
        # cluster_vector_all = histograms_region_all / histograms_region_all.max(axis=0).reshape(1, len(avalanches_bin_all[0][0]))
        # order_all = np.argsort(np.sum(histograms_region_all, axis=1))
        # cluster_vector_all = cluster_vector_all[order_all]
        #
        # plot_compare_cluster(cluster_vector_phate, order_phate, cluster_vector_all, order_all,
        #                      'Phate cluster', 'all subject', 'cluster Phate - cluster all')
        # plt.savefig(path_all+'/figure_compare/compare_cluster_phate.png')


        # plt.show()

