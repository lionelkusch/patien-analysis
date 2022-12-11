import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
import scipy.io as io


def plot_pvalue(data, data_num, title='patient : '):
    """
    plotting array of the matrix with color and number
    :param data: data for the color
    :param data_num: data for the number to show
    :param title: generic title of each subplot
    :return:
    """
    nb_patient = data.shape[0]
    nb_x = int(np.sqrt(nb_patient)) + 1
    nb_y = int(nb_patient / np.sqrt(nb_patient))

    fig, axs = plt.subplots(nb_x, nb_y, figsize=(20, 20))
    for index_patient in range(nb_patient):
        im = axs[int(index_patient % nb_x), int(index_patient / nb_x)].imshow(data[index_patient], vmin=0.0, vmax=1.0)
        axs[int(index_patient % nb_x), int(index_patient / nb_x)].set_title(title + str(index_patient))
        axs[int(index_patient % nb_x), int(index_patient / nb_x)].autoscale(False)
        fig.colorbar(im, ax=axs[int(index_patient % nb_x), int(index_patient / nb_x)])
        for (j, i), label in np.ndenumerate(data_num[index_patient]):
            axs[int(index_patient % nb_x), int(index_patient / nb_x)].text(i, j, np.around(label, 4), ha='center',
                                                                           va='center')
        axs[int(index_patient % nb_x), int(index_patient / nb_x)].tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)
    for index_no_patient in range(18, nb_x * nb_y):
        axs[int(index_no_patient % nb_x), int(index_no_patient / nb_x)].imshow(
            np.ones_like(data[0]) * np.NAN, vmin=0.0, vmax=1.0)
        axs[int(index_no_patient % nb_x), int(index_no_patient / nb_x)].autoscale(False)
        axs[int(index_no_patient % nb_x), int(index_no_patient / nb_x)].tick_params(
            axis='both', which='both', bottom=False, top=False, labelbottom=False)
    plt.subplots_adjust(left=0.0, right=1.0, wspace=0.0, top=0.975, bottom=0.005, hspace=0.15)


def plot_cluster(data, data_num, title='patient : ', vmin=0.0, vmax=1.0):
    """
    plot matrix
    :param data: data for the color
    :param data_num:  data for the number
    :param title: title of the figure
    :param vmin: minimum values for the color
    :param vmax: maximum value for the color
    :return:
    """
    nb_patient = data.shape[0]
    nb_x = nb_patient

    fig, axs = plt.subplots(nb_x, 1, figsize=(20, 20))
    if nb_patient == 1:
        im = axs.imshow(data[0], vmin=vmin, vmax=vmax)
        axs.set_title(title)
        axs.autoscale(False)
        fig.colorbar(im, ax=axs)
        for (j, i), label in np.ndenumerate(data_num[0]):
            axs.text(i, j, np.around(label, 2), ha='center', va='center', fontsize=10)
        axs.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False)
        plt.subplots_adjust(left=0.005, right=1.0, wspace=0.0, top=0.96, bottom=0.005, hspace=0.15)
    else:
        for index_patient in range(nb_patient):
            im = axs[index_patient].imshow(data[index_patient], vmin=vmin, vmax=vmax)
            axs[index_patient].set_title(title + str(index_patient))
            axs[index_patient].autoscale(False)
            # fig.colorbar(im, ax=axs[index_patient])
            for (j, i), label in np.ndenumerate(data_num[index_patient]):
                axs[index_patient].text(i, j, np.around(label, 2), ha='center', va='center', fontsize=5)
            axs[index_patient].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False)
        plt.subplots_adjust(left=0.005, right=1.0, wspace=0.0, top=1.0, bottom=0.005, hspace=0.15)


def null_model_cluster_regions_res(path_saving, nb_randomize=100, base=None, plot=False,
                                   significatif=0.05, save_mat=False):
    """
    plot result for entropy of the shuffling cluster
    :param path_saving: path where are the result
    :param nb_randomize: number of randomisation to take into account
    :param base: base of the entropy
    :param plot: save the figure or not
    :return:
    """
    # load value
    data_null_model = []
    for nb_rand in range(nb_randomize):
        data_null_model.append(np.load(path_saving + "/histograms_region_" + str(nb_rand) + ".npy"))
    data_patient = np.load(path_saving + "/histograms_region.npy")

    pvalue_cluster_all = []
    for index, cluster_region in enumerate(data_patient):
        pvalue_all = np.sum(
            np.sum(np.array(data_null_model) > cluster_region, axis=0) / nb_randomize, axis=0) / 5
        significatif_high_all = pvalue_all > 1.0 - significatif
        significatif_low_all = pvalue_all < significatif
        significatif_all_all = np.logical_or(significatif_low_all, significatif_high_all)
        pvalue_cluster_all.append(
            [[pvalue_all], [significatif_all_all], [significatif_high_all], [significatif_low_all]])
    pvalue_cluster_all = np.array(pvalue_cluster_all)
    if save_mat:
        io.savemat(path_saving + '/cluster_res_vector_null_trans.mat', {'transmatrix': 2*pvalue_cluster_all[:, 2, :, :]
                                                                        + pvalue_cluster_all[:, 3, :, :]})
    plot_cluster(pvalue_cluster_all[:, 0, :, :], pvalue_cluster_all[:, 0, :, :], title='cluster ')
    if plot:
        plt.savefig(path_saving + '/figure/cluster_res_pvalue.pdf')
        plt.close('all')
    plot_cluster(pvalue_cluster_all[:, 1, :, :], pvalue_cluster_all[:, 0, :, :], title='cluster ')
    if plot:
        plt.savefig(path_saving + '/figure/cluster_res_pvalue_significatif.pdf')
        plt.close('all')
    else:
        plt.show()

    # entropy plotting
    for index, data_cluster in enumerate(data_patient):
        entropy_values = []
        for data in data_null_model:
            entropy_values.append(entropy(data[index], base=base))
        print('cluster ' + str(index) + ' ' + str(entropy(data_patient[index], base=base)) + ' ' + str(
            np.mean(entropy_values)))
        plt.figure()
        plt.suptitle('cluster ' + str(index))
        plt.hist(entropy_values)
        plt.vlines(entropy(data_patient[index], base=base), ymin=0.0, ymax=20.0, color='r')

    entropy_values = []
    for data in data_null_model:
        entropy_values.append(entropy(data.ravel(), base=base))
    print('cluster all ' + str(entropy(data_patient.ravel(), base=base)) + ' ' + str(np.mean(entropy_values)))
    plt.figure()
    plt.suptitle('all_cluster')
    plt.hist(entropy_values)
    plt.vlines(entropy(data_patient.ravel(), base=base), ymin=0.0, ymax=20.0, color='r')
    if plot:
        plt.savefig(path_saving + '/figure/cluster_entropy.pdf')
        plt.close('all')
    else:
        plt.show()


def null_model_transition(path_saving, nb_randomize=100, significatif=0.05, plot=False):
    """
    plot the indicative for transition by patient
    :param path_saving: path where are the result
    :param nb_randomize: number of randomisation to take into account
    :param significatif: acceptation of the percentage to take into account
    :param plot: save the figure or not
    :return:
    """
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

    # compute the variability of indicative transition
    transistion_matrix = np.array(data_patient['transition'])
    transistion_matrix[np.logical_not(significatif_all)] = np.NAN
    std_significant_transition = np.nanstd(transistion_matrix, axis=0)
    mean_significant_transition = np.nanmean(transistion_matrix, axis=0)

    fig = plt.figure(figsize=(20, 20))
    for index, (title, data) in enumerate([('mean', mean_significant_transition),
                                           ('std', std_significant_transition),
                                           ('std/mean', std_significant_transition / mean_significant_transition),
                                           ]):
        ax = plt.subplot(131 + index)
        ax.set_title(title)
        im = plt.imshow(data)
        fig.colorbar(im, ax=ax, fraction=0.05)
        for (j, i), label in np.ndenumerate(data):
            ax.text(i, j, np.around(label, 4), ha='center', va='center')
        plt.subplots_adjust(top=1.0, bottom=0.0, left=0.001, right=0.97 )
    if plot:
        plt.savefig(path_saving + '/figure/transition_patient_std.pdf')
        plt.close('all')

    plot_pvalue(pvalue, pvalue)
    if plot:
        plt.savefig(path_saving + '/figure/transition_patient_pvalue.pdf')
        plt.close('all')
    plot_pvalue(significatif_all, pvalue)
    if plot:
        plt.savefig(path_saving + '/figure/transition_patient_pvalue_significatif.pdf')
        plt.close('all')
    else:
        plt.show()


def null_model_transition_all(path_saving, nb_randomize=100, significatif=0.05, plot=False):
    """
    plot the indicative for transition
    :param path_saving: path where are the result
    :param nb_randomize: number of randomisation to take into account
    :param significatif: acceptation of the percentage to take into account
    :param plot: save the figure or not
    :return:
    """
    # load value
    data_null_model = {'transition': [],
                       'histogram': [],
                       'cluster_patient': []
                       }
    for nb_rand in range(nb_randomize):
        data_null_model['transition'].append(np.load(path_saving + "/transition_all" + str(nb_rand) + ".npy"))

    data_patient = {'transition': np.load(path_saving + "/transition_all.npy"),
                    }
    pvalue = np.sum(np.array(data_null_model['transition']) > data_patient['transition'], axis=0) / nb_randomize
    pvalue = np.expand_dims(pvalue, axis=0)
    significatif_high = pvalue > 1.0 - significatif
    significatif_low = pvalue < significatif
    significatif_all = np.logical_or(significatif_low, significatif_high)

    plot_cluster(np.expand_dims(data_patient['transition'], axis=0),
                 np.expand_dims(data_patient['transition'], axis=0), title='transition of all patient', vmax=0.4)
    if plot:
        plt.savefig(path_saving + '/figure/transition_all.pdf')
        plt.close('all')
    plot_cluster(pvalue, pvalue, title='transition of all patient')
    if plot:
        plt.savefig(path_saving + '/figure/transition_all_pvalue.pdf')
        plt.close('all')
    plot_cluster(significatif_all, pvalue, title='transition of all patient')
    if plot:
        plt.savefig(path_saving + '/figure/transition_all_pvalue_significatif.pdf')
        plt.close('all')
    else:
        plt.show()


def null_model_data(path_saving, path_saving_patient, nb_randomize=100, significatif=0.05, plot=False, save_mat=False):
    """
    plot the indicative for region in each cluster
    :param path_saving: path where are the result
    :param path_saving_patient: path where to get the result from the patient
    :param nb_randomize: number of randomisation to take into account
    :param significatif: acceptation of the percentage to take into account
    :param plot: save the figure or not
    :param save_mat: saving the result in mat format
    :return:
    """
    # load value
    data_null_model = {'Y_phate': [],
                       'transition': [],
                       'histogram': [],
                       'histogram_region': [],
                       'cluster_patient': []
                       }
    for nb_rand in range(nb_randomize):
        data_null_model['Y_phate'].append(np.load(path_saving + "/" + str(nb_rand) + "_Phate.npy"))
        data_null_model['transition'].append(np.load(path_saving + "/" + str(nb_rand) + "_transition.npy"))
        data_null_model['histogram'].append(np.load(path_saving + "/" + str(nb_rand) + "_histograms_patient.npy"))
        data_null_model['histogram_region'].append(np.load(path_saving + "/" + str(nb_rand) + "_histograms_region.npy"))
        data_null_model['cluster_patient'].append(
            np.load(path_saving + "/" + str(nb_rand) + "_cluster_patient_data.npy", allow_pickle=True))

    data_patient = {'Y_phate': np.load(path_saving_patient + "/Phate.npy"),
                    'transition': np.load(path_saving_patient + "/transition.npy"),
                    'histogram': np.load(path_saving_patient + "/histograms_patient.npy"),
                    'histogram_region': np.load(path_saving_patient + "/histograms_region.npy"),
                    'cluster_patient': np.load(path_saving_patient + "/cluster_patient_data.npy", allow_pickle=True)
                    }

    pvalue_cluster_all = []
    for index, cluster_region in enumerate(data_patient['histogram_region']):
        pvalue_all = np.sum(
            np.sum(np.array(data_null_model['histogram_region']) > cluster_region, axis=0) / nb_randomize, axis=0) / 5
        significatif_high_all = pvalue_all > 1.0 - significatif
        significatif_low_all = pvalue_all < significatif
        significatif_all_all = np.logical_or(significatif_low_all, significatif_high_all)
        pvalue_cluster_all.append(
            [[pvalue_all], [significatif_all_all], [significatif_high_all], [significatif_low_all]])
    pvalue_cluster_all = np.array(pvalue_cluster_all)
    if save_mat:
        io.savemat(path_saving + '/vector_null_trans.mat', {'transmatrix': 2*pvalue_cluster_all[:, 2, :, :]
                                                                           + pvalue_cluster_all[:, 3, :, :]})
    plot_cluster(pvalue_cluster_all[:, 0, :, :], pvalue_cluster_all[:, 0, :, :], title='cluster ')
    if plot:
        plt.savefig(path_saving + '/figure/cluster_pvalue.pdf')
        plt.close('all')
    plot_cluster(pvalue_cluster_all[:, 1, :, :], pvalue_cluster_all[:, 0, :, :], title='cluster ')
    if plot:
        plt.savefig(path_saving + '/figure/cluster_pvalue_significatif.pdf')
        plt.close('all')
    else:
        plt.show()


def null_model_data_entropy(path_saving, path_saving_patient, nb_randomize=100, base=None, plot=False):
    """
    plot the indicative for region in each cluster
    :param path_saving: path where are the result
    :param path_saving_patient: path where to get the result from the patient
    :param nb_randomize: number of randomisation to take into account
    :param base: base for the entropy
    :param plot: save the figure or not
    :return:
    """
    # load value
    data_null_model = []
    for nb_rand in range(nb_randomize):
        data_null_model.append(np.load(path_saving + "/"+str(nb_rand)+"_histograms_region.npy"))
    data_patient = np.load(path_saving_patient + "/histograms_region.npy")

    # entropy for each cluster
    for index, data_cluster in enumerate(data_patient):
        entropy_values = []
        for data in data_null_model:
            entropy_values.append(entropy(data[index], base=base))
        print('cluster ' + str(index) + ' ' + str(entropy(data_patient[index], base=base)) + ' ' + str(
            np.mean(entropy_values)))
        plt.figure()
        plt.suptitle('cluster ' + str(index))
        plt.hist(entropy_values)
        plt.vlines(entropy(data_patient[index], base=base), ymin=0.0, ymax=20.0, color='r')
        if plot:
            plt.savefig(path_saving + '/figure/null_nmodel_cluster_entropy_'+str(index)+'.pdf')
            plt.close('all')

    # entropy for all cluster
    entropy_values = []
    for data in data_null_model:
        entropy_values.append(entropy(data.ravel(), base=base))
    print('cluster all ' + str(entropy(data_patient.ravel(), base=base)) + ' ' + str(np.mean(entropy_values)))
    plt.figure()
    plt.suptitle('all_cluster')
    plt.hist(entropy_values)
    plt.vlines(entropy(data_patient.ravel(), base=base), ymin=0.0, ymax=20.0, color='r')
    if plot:
        plt.savefig(path_saving + '/figure/null_nmodel_cluster_entropy.pdf')
        plt.close('all')
    else:
        plt.show()

if __name__ == '__main__':
    null_model_cluster_regions_res(path_saving="/home/kusch/Documents/project/patient_analyse/paper/result/default/",
                                   plot=True, significatif=0.05/(90*7),  # precision/(number of region * number of cluster)
                                   save_mat=True, nb_randomize=10000)
    null_model_transition(path_saving="/home/kusch/Documents/project/patient_analyse/paper/result/default/",
                          plot=True)
    null_model_transition_all(path_saving="/home/kusch/Documents/project/patient_analyse/paper/result/default/",
                              plot=True)
    null_model_data(path_saving="/home/kusch/Documents/project/patient_analyse/paper/result/default/null_model/",
                    path_saving_patient="/home/kusch/Documents/project/patient_analyse/paper/result/default/",
                    significatif=0.05, plot=True, save_mat=True)
    null_model_data_entropy(
        path_saving="/home/kusch/Documents/project/patient_analyse/paper/result/default/null_model/",
        path_saving_patient="/home/kusch/Documents/project/patient_analyse/paper/result/default/",
        plot=True)
