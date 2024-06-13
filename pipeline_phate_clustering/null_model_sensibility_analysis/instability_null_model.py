import os
import numpy as np
import phate
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def get_transition(kmeans_nb_cluster, kmeans_seed, nb_subject, avalanches_bin=None, Y_phate=None,
                   cluster_patient_data=None):
    """
    get the transition from a clustering
    :param kmeans_nb_cluster: number of cluster
    :param kmeans_seed: seed  of the cluster
    :param nb_subject: number of patients
    :param avalanches_bin: avalanches
    :param Y_phate: low dimension
    :param cluster_patient_data: clustering values
    :return: transition probability and cluster for all together and for each patient
    """
    # generate the clustering
    if cluster_patient_data is None:
        if avalanches_bin is None and Y_phate is None:
            raise Exception('missing input values')
        cluster = KMeans(n_clusters=kmeans_nb_cluster, random_state=kmeans_seed).fit_predict(Y_phate)
        cluster_patient_data = []
        begin = 0
        for avalanche in avalanches_bin:
            end = begin + len(avalanche)
            cluster_patient_data.append(cluster[begin:end])
            begin = end

    # get the transition of each patients
    transition = np.empty((nb_subject, kmeans_nb_cluster, kmeans_nb_cluster))
    for index_patient, cluster_k in enumerate(cluster_patient_data):
        next_step = cluster_k[1:]
        step = cluster_k[:-1]
        for i in range(kmeans_nb_cluster):
            data = next_step[np.where(step == i)]
            percentage_trans = np.bincount(data)
            if len(percentage_trans) < kmeans_nb_cluster:
                percentage_trans = np.concatenate(
                    [percentage_trans, np.zeros(kmeans_nb_cluster - percentage_trans.shape[0])])
            transition[index_patient, i, :] = percentage_trans / len(data)
    # get the transition independant of the patient
    transition_all = np.empty((kmeans_nb_cluster, kmeans_nb_cluster))
    cluster_all = np.concatenate(cluster_patient_data)
    next_step = cluster_all[1:]
    step = cluster_all[:-1]
    for i in range(kmeans_nb_cluster):
        data = next_step[np.where(step == i)]
        percentage_trans = np.bincount(data)
        if len(percentage_trans) < kmeans_nb_cluster:
            percentage_trans = np.concatenate(
                [percentage_trans, np.zeros(kmeans_nb_cluster - percentage_trans.shape[0])])
        transition_all[i, :] = percentage_trans / len(data)

    return (transition, cluster_patient_data), (transition_all, cluster_all)


def null_model_diagonal(path_saving, max_nb_cluster=15, kmeans_seed=123, nb_subject=18, nb_randomize=100, seed=123,
                        significant=0.05, plot_save=False):
    """
    null model of transition with instability of cluster
    :param path_saving: path of result of default paper
    :param max_nb_cluster: max number of cluster to evaluate
    :param kmeans_seed: seed for clustering
    :param nb_subject: number of patient
    :param nb_randomize: number of randomization
    :param significant: precision of significant result
    :param seed: seed of the random generator
    :param plot_save: plot or not the result
    :return:
    """
    # get data
    avalanches_bin = np.load(path_saving + '/avalanches.npy', allow_pickle=True)
    Y_phate = np.load(path_saving + "/Phate.npy")

    ## prepare the result
    # function for compute the pvalue
    pvalue_function = lambda randomize, data: np.sum(np.array(randomize > data), axis=0) / nb_randomize
    index_of_cluster_nb = list(range(2, max_nb_cluster))
    # pvalues = []
    # for kmeans_nb_cluster in index_of_cluster_nb:
    #     print(kmeans_nb_cluster)
    #     # get the percentage that the diagonal element has the higher transition  probability
    #     per_diagonal = lambda transition: np.sum(
    #         np.argmax(transition, axis=0) == range(kmeans_nb_cluster)) / kmeans_nb_cluster
    #     # get the transition for the data
    #     (default_transition, default_cluster_patient_data), \
    #     (default_transition_all, default_cluster_all) = \
    #         get_transition(kmeans_nb_cluster, kmeans_seed, nb_subject, avalanches_bin=avalanches_bin, Y_phate=Y_phate, )
    #     result_default_per_diag = [per_diagonal(transition) for transition in default_transition]
    #     result_default_per_diag_all = per_diagonal(default_transition_all)
    #
    #     # generate a null model by shuffling the label of each avalanches
    #     rng = np.random.default_rng(seed=seed)
    #     shuffle_cluster = np.copy(np.array(default_cluster_patient_data, dtype=object))
    #     shuffle_per_diagonal = []
    #     shuffle_transition, shuffle_cluster_patient_data, shuffle_transition_all, shuffle_cluster_all = [], [], [], []
    #     for nb_rand in range(nb_randomize):
    #         for i in range(shuffle_cluster.shape[0]):
    #             rng.shuffle(shuffle_cluster[i])
    #         (transition_patient, cluster_patient_data), \
    #         (transition_all, cluster_all) = \
    #             get_transition(kmeans_nb_cluster, kmeans_seed, nb_subject, cluster_patient_data=shuffle_cluster)
    #         result_per_diag = [per_diagonal(transition) for transition in transition_patient]
    #         result_per_diag.append(per_diagonal(transition_all))
    #         shuffle_per_diagonal.append(result_per_diag)
    #         shuffle_transition.append(transition_patient)
    #         shuffle_cluster_patient_data.append(cluster_patient_data)
    #         shuffle_transition_all.append(transition_all)
    #         shuffle_cluster_all.append(cluster_all)
    #     # put the result in dictionary
    #     shuffle_per_diagonal = np.array(shuffle_per_diagonal)
    #     shuffle_per_diagonal_all = shuffle_per_diagonal[:, -1]
    #     shuffle_per_diagonal = shuffle_per_diagonal[:, :-1]
    #     pvalue_per_diag = pvalue_function(shuffle_per_diagonal, result_default_per_diag)
    #     pvalue_per_diag_all = pvalue_function(shuffle_per_diagonal_all, result_default_per_diag_all)
    #     pvalue_per_transition = pvalue_function(shuffle_transition, default_transition)
    #     pvalue_per_transition_all = pvalue_function(shuffle_transition_all, default_transition_all)
    #     pvalues.append(
    #         {'nb_cluster': kmeans_nb_cluster,
    #          'data_per_diagonal': result_default_per_diag, 'data_per_diagonal_all': result_default_per_diag_all,
    #          'shuffle_per_diagonal': shuffle_per_diagonal, 'shuffle_per_diagonal_all': shuffle_per_diagonal_all,
    #          'per_diag': pvalue_per_diag, 'per_diag_all': pvalue_per_diag_all,
    #          'transition': pvalue_per_transition, 'transition_all': pvalue_per_transition_all,
    #          })

    # np.save(path_saving + 'model_diagonal.npy', pvalues)
    pvalues = np.load(path_saving + 'model_diagonal.npy', allow_pickle=True)
    # plot the evolution of percentage of diagonal
    per_diag = [pvalue['data_per_diagonal_all'] for pvalue in pvalues]
    per_diag_rand = np.array([pvalue['shuffle_per_diagonal_all'] for pvalue in pvalues])
    plt.figure()
    plt.plot(range(2, max_nb_cluster), per_diag, 'b', label='data')
    plt.plot(range(2, max_nb_cluster), per_diag_rand.mean(axis=1), 'g', label='null model')
    plt.fill_between(range(2, max_nb_cluster),
                     per_diag_rand.mean(axis=1) + per_diag_rand.std(axis=1),
                     per_diag_rand.mean(axis=1) - per_diag_rand.std(axis=1), 'g', alpha=0.5)
    plt.title('Probability that the transition on the diagonal is higher probability')
    if plot_save:
        plt.savefig(path_saving + '/diagonal/probability_transition.pdf')
        plt.close('all')

    # plot the significant percentage of transition
    # get element from the diagonal
    trans_values = [pvalue['transition_all'] for pvalue in pvalues]
    per_diags = [np.diag(data) for data in trans_values]
    diag_per_lows, diag_per_nos, diag_per_highs = [], [], []
    for per_diag in per_diags:
        total = len(per_diag)
        diag_per_lows.append(np.sum(per_diag < significant)/total)
        diag_per_highs.append(np.sum(per_diag > 1-significant)/total)
        diag_per_nos.append(1.0-(np.sum(per_diag < significant) + np.sum(per_diag > 1-significant))/total)

    # get the element from non diagonal
    per_no_diag = []
    for data in trans_values:
        # upper triangle. k=1 excludes the diagonal elements.
        xu, yu = np.triu_indices_from(data, k=1)
        # lower triangle
        xl, yl = np.tril_indices_from(data, k=-1)  # Careful, here the offset is -1
        # combine
        x = np.concatenate((xl, xu))
        y = np.concatenate((yl, yu))
        indices = (x, y)
        per_no_diag.append(data[indices])
    # get the percentage of this element by significant high, low and no
    no_diag_per_lows, no_diag_per_nos, no_diag_per_highs = [], [], []
    for percentage in per_no_diag:
        total = len(percentage)
        no_diag_per_lows.append(np.sum(percentage < significant)/total)
        no_diag_per_highs.append(np.sum(percentage > 1-significant)/total)
        no_diag_per_nos.append(1.0-(np.sum(percentage < significant) + np.sum(percentage > 1-significant))/total)

    np.save(path_saving + 'model_diagonalsignificatif.npy',
            np.array([diag_per_lows, diag_per_nos, diag_per_highs,
                                         no_diag_per_lows, no_diag_per_nos, no_diag_per_highs]))
    # plot result
    plt.figure()
    for index, data in enumerate(np.array([diag_per_lows, diag_per_nos, diag_per_highs,
                                         no_diag_per_lows, no_diag_per_nos, no_diag_per_highs]).T):
        plt.bar(index_of_cluster_nb[index], data[0], color=['b'], width=0.4)
        plt.bar(index_of_cluster_nb[index], data[1], color=['w'], width=0.4, bottom=data[0])
        plt.bar(index_of_cluster_nb[index], data[2], color=['r'], width=0.4, bottom=data[0]+data[1])
        plt.bar(index_of_cluster_nb[index] + 0.3, data[3], color='b', width=0.4)
        plt.bar(index_of_cluster_nb[index] + 0.3, data[4], color='w', width=0.4, bottom=data[3])
        plt.bar(index_of_cluster_nb[index] + 0.3, data[5], color='r', width=0.4, bottom=data[3] + data[4])
    plt.title('Percentage of significant transition')
    if plot_save:
        plt.savefig(path_saving + '/diagonal/significant.pdf')
        plt.close('all')



def null_model_diagonal_phate(path_saving, path_data, max_nb_cluster=15, kmeans_seed=123, nb_subject=18, nb_randomize=100, seed=123,
                              significant=0.05, plot_save=False):
    """
    null model of transition with instability of cluster from shuffling avalanches
    :param path_saving: path of result of default paper
    :param path_data: path of low dimension of shuffling avalanches
    :param max_nb_cluster: max number of cluster to evaluate
    :param kmeans_seed: seed for clustering
    :param nb_subject: number of patient
    :param nb_randomize: number of randomization
    :param significant: precision of significant result
    :param seed: seed of the random generator
    :param plot_save: plot or not the result
    :return:
    """
    # get data
    avalanches_bin = np.load(path_saving + '/avalanches.npy', allow_pickle=True)
    Y_phate = np.load(path_saving + "/Phate.npy")
    null_models = []
    for nb_rand in range(nb_randomize):
            null_models.append(np.load(path_data + "/" + str(nb_rand) +'_Phate.npy'))

    ## prepare the result
    # function for compute the pvalue
    pvalue_function = lambda randomize, data: np.sum(np.array(randomize > data), axis=0) / nb_randomize
    index_of_cluster_nb = list(range(2, max_nb_cluster))
    pvalues = []
    for kmeans_nb_cluster in index_of_cluster_nb:
        print(kmeans_nb_cluster)
        # get the percentage that the diagonal element has the higher transition  probability
        per_diagonal = lambda transition: np.sum(
            np.argmax(transition, axis=0) == range(kmeans_nb_cluster)) / kmeans_nb_cluster
        # get the transition for the data
        (default_transition, default_cluster_patient_data), \
        (default_transition_all, default_cluster_all) = \
            get_transition(kmeans_nb_cluster, kmeans_seed, nb_subject, avalanches_bin=avalanches_bin, Y_phate=Y_phate, )
        result_default_per_diag = [per_diagonal(transition) for transition in default_transition]
        result_default_per_diag_all = per_diagonal(default_transition_all)

        # generate a null model by shuffling the label of each avalanches
        shuffle_per_diagonal = []
        shuffle_transition, shuffle_cluster_patient_data, shuffle_transition_all, shuffle_cluster_all = [], [], [], []
        for nb_rand in range(nb_randomize):
            print(nb_rand)
            (transition_patient, cluster_patient_data), \
            (transition_all, cluster_all) = \
                get_transition(kmeans_nb_cluster, kmeans_seed, nb_subject, avalanches_bin=avalanches_bin, Y_phate=null_models[nb_rand])
            result_per_diag = [per_diagonal(transition) for transition in transition_patient]
            result_per_diag.append(per_diagonal(transition_all))
            shuffle_per_diagonal.append(result_per_diag)
            shuffle_transition.append(transition_patient)
            shuffle_cluster_patient_data.append(cluster_patient_data)
            shuffle_transition_all.append(transition_all)
            shuffle_cluster_all.append(cluster_all)
        # put the result in dictionary
        shuffle_per_diagonal = np.array(shuffle_per_diagonal)
        shuffle_per_diagonal_all = shuffle_per_diagonal[:, -1]
        shuffle_per_diagonal = shuffle_per_diagonal[:, :-1]
        pvalue_per_diag = pvalue_function(shuffle_per_diagonal, result_default_per_diag)
        pvalue_per_diag_all = pvalue_function(shuffle_per_diagonal_all, result_default_per_diag_all)
        pvalue_per_transition = pvalue_function(shuffle_transition, default_transition)
        pvalue_per_transition_all = pvalue_function(shuffle_transition_all, default_transition_all)
        pvalues.append(
            {'nb_cluster': kmeans_nb_cluster,
             'data_per_diagonal': result_default_per_diag, 'data_per_diagonal_all': result_default_per_diag_all,
             'shuffle_per_diagonal': shuffle_per_diagonal, 'shuffle_per_diagonal_all': shuffle_per_diagonal_all,
             'per_diag': pvalue_per_diag, 'per_diag_all': pvalue_per_diag_all,
             'transition': pvalue_per_transition, 'transition_all': pvalue_per_transition_all,
             })
    np.save(path_saving + 'model_diagonal_phate.npy', pvalues)

    # plot evolution of percentage of diagonal
    per_diag = [pvalue['data_per_diagonal_all'] for pvalue in pvalues]
    per_diag_rand = np.array([pvalue['shuffle_per_diagonal_all'] for pvalue in pvalues])
    plt.figure()
    plt.plot(range(2, max_nb_cluster), per_diag, 'b', label='data')
    plt.plot(range(2, max_nb_cluster), per_diag_rand.mean(axis=1), 'g', label='null model')
    plt.fill_between(range(2, max_nb_cluster),
                     per_diag_rand.mean(axis=1) + per_diag_rand.std(axis=1),
                     per_diag_rand.mean(axis=1) - per_diag_rand.std(axis=1), 'g', alpha=0.5)
    plt.title('Probability that the transition on the diagonal is higher probability')
    if plot_save:
        plt.savefig(path_saving + '/diagonal/null_phate_probability_transition.pdf')
        plt.close('all')

    # plot the significant percentage of transition
    # get element from the diagonal
    trans_values = [pvalue['transition_all'] for pvalue in pvalues]
    per_diags = [np.diag(data) for data in trans_values]
    diag_per_lows, diag_per_nos, diag_per_highs = [], [], []
    for per_diag in per_diags:
        total = len(per_diag)
        diag_per_lows.append(np.sum(per_diag < significant)/total)
        diag_per_highs.append(np.sum(per_diag > 1-significant)/total)
        diag_per_nos.append(1.0-(np.sum(per_diag < significant) + np.sum(per_diag > 1-significant))/total)

    # get the element from non diagonal
    per_no_diag = []
    for data in trans_values:
        # upper triangle. k=1 excludes the diagonal elements.
        xu, yu = np.triu_indices_from(data, k=1)
        # lower triangle
        xl, yl = np.tril_indices_from(data, k=-1)  # Careful, here the offset is -1
        # combine
        x = np.concatenate((xl, xu))
        y = np.concatenate((yl, yu))
        indices = (x, y)
        per_no_diag.append(data[indices])
    # get the percentage of this element by significant high, low and no
    no_diag_per_lows, no_diag_per_nos, no_diag_per_highs = [], [], []
    for percentage in per_no_diag:
        total = len(percentage)
        no_diag_per_lows.append(np.sum(percentage < significant)/total)
        no_diag_per_highs.append(np.sum(percentage > 1-significant)/total)
        no_diag_per_nos.append(1.0-(np.sum(percentage < significant) + np.sum(percentage > 1-significant))/total)

    np.save(path_saving + 'model_diagonalsignificatif_phate.npy',
            np.array([diag_per_lows, diag_per_nos, diag_per_highs,
                                         no_diag_per_lows, no_diag_per_nos, no_diag_per_highs]))
    # plot result
    plt.figure()
    for index, data in enumerate(np.array([diag_per_lows, diag_per_nos, diag_per_highs,
                                         no_diag_per_lows, no_diag_per_nos, no_diag_per_highs]).T):
        plt.bar(index_of_cluster_nb[index], data[0], color=['b'], width=0.4)
        plt.bar(index_of_cluster_nb[index], data[1], color=['w'], width=0.4, bottom=data[0])
        plt.bar(index_of_cluster_nb[index], data[2], color=['r'], width=0.4, bottom=data[0]+data[1])
        plt.bar(index_of_cluster_nb[index] + 0.3, data[3], color='b', width=0.4)
        plt.bar(index_of_cluster_nb[index] + 0.3, data[4], color='w', width=0.4, bottom=data[3])
        plt.bar(index_of_cluster_nb[index] + 0.3, data[5], color='r', width=0.4, bottom=data[3] + data[4])
    plt.title('Percentage of significant transition')
    if plot_save:
        plt.savefig(path_saving + '/diagonal/null_phate_significant.pdf')
        plt.close('all')
    else:
        plt.show()


if __name__ == '__main__':
    import os
    path_data = os.path.dirname(os.path.realpath(__file__)) + '/../../'
    null_model_diagonal(path_saving=path_data+"/paper/result/default/",
                        plot_save=True)
    null_model_diagonal_phate(path_saving=path_data+"/paper/result/default/",
                              path_data=path_data+"/paper/result/default/null_model/",
                              plot_save=True,
                              significant=0.05  #0.2
                              )
