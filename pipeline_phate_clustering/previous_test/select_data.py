import os
import numpy as np
import h5py
from pipeline_phate_clustering.functions_helper.load_data import go_avalanches


def get_data_all_patient(Nsubs=47, nregions=90, remove_subject=[11, 15, 20]):
    # max_region = 116
    # DATASET EXTRACTION
    # remove suject 11,15,20

    path_data = os.path.dirname(os.path.realpath(__file__)) + '/../data/'
    f = h5py.File(path_data + 'serie_Melbourne.mat', 'r')

    struArray = f['D']
    data = []
    for i in range(Nsubs):
        if not (i in remove_subject):
            data.append(np.swapaxes(f[struArray[i, 0]][:nregions, :], 0, 1))

    avalanches_bin = []
    avalanches_sum = []
    for subject in data:
        Avalanches_human = go_avalanches(subject, thre=3, direc=0, binsize=1)
        out = [[] for i in range(len(Avalanches_human['ranges']))]
        out_sum = [[] for i in range(len(Avalanches_human['ranges']))]
        for kk1 in range(len(Avalanches_human['ranges'])):
            begin = Avalanches_human['ranges'][kk1][0]
            end = Avalanches_human['ranges'][kk1][1]
            sum_kk = np.sum(Avalanches_human['Zbin'][begin:end, :], 0)
            out_sum[kk1] = sum_kk
            out[kk1] = np.zeros(nregions)
            out[kk1][np.where(sum_kk >= 1)] = 1

        avalanches_bin.append(np.concatenate([out], axis=1))
        avalanches_sum.append(np.concatenate([out_sum], axis=1))

    return avalanches_bin, avalanches_sum


# def select_avalanches_rand(avalanches, sensibility=5, count=None, seed=123):
#     concate = np.concatenate(avalanches)
#     sum = np.sum(concate, axis=0)
#     np.random.seed(seed)
#     if count is None:
#         count = int(sum.min())+1  # value to get for all regions
#     select_index = np.arange(concate.shape[0])
#     while any(np.abs(np.unique(concate[select_index].sum(axis=0)) - count) > sensibility):
#         count -=1
#         print(count, np.abs(np.unique(concate[select_index].sum(axis=0)) - count), sensibility)
#         reg_1 = sum.argmin()  # select first region
#         # start selection
#         condition = concate[:, reg_1] == 1
#         index_cond_1 = np.where(condition)[0]
#         nb_reg = np.min([count, int(np.sum(concate[index_cond_1], axis=0)[reg_1])])
#         np.random.shuffle(np.arange(len(index_cond_1)))
#         select_index = index_cond_1[:nb_reg]
#         not_select_index = np.concatenate((np.where(np.logical_not(condition))[0],))
#         remove_index = np.where(condition)[0][nb_reg:]
#         assert(select_index.shape[0] + remove_index.shape[0] + not_select_index.shape[0] == concate.shape[0])
#         # if (select_index.shape[0] + remove_index.shape[0] + not_select_index.shape[0] != concate.shape[0]):
#         #     print(select_index.shape[0] + remove_index.shape[0] + not_select_index.shape[0])
#         #     print(concate.shape[0])
#
#         for i in range(1, concate.shape[1]):
#             print(i)
#             # second region
#             reg_2 = np.sum(concate[not_select_index], axis=0).argsort()[i]
#             nb_reg_2 = int(count - np.sum(concate[select_index], axis=0)[reg_2])
#             cond_2 = np.logical_or(concate[:, reg_2] == 1, condition)
#             select_index_2 = np.where(np.logical_and(concate[:, reg_2] == 1, np.logical_not(condition)))[0]
#             np.random.shuffle(select_index_2)
#             select_index_2 = select_index_2[:nb_reg_2]
#             select_index = np.concatenate((select_index, select_index_2))
#             remove_index = np.concatenate((remove_index, not_select_index[np.where([(not(i in select_index_2) and concate[i, reg_2] == 1 ) for i in not_select_index ])]))
#             not_select_index = np.where(np.logical_not(cond_2))[0]
#             condition = cond_2
#             assert (select_index.shape[0] + remove_index.shape[0] + not_select_index.shape[0] == concate.shape[0])
#             # if (select_index.shape[0] + remove_index.shape[0] + not_select_index.shape[0] != concate.shape[0]):
#             #     print(select_index.shape[0] + remove_index.shape[0] + not_select_index.shape[0])
#             #     print(concate.shape[0])
#             #     break
#     print(count, np.abs(np.unique(concate[select_index].sum(axis=0)) - count), sensibility)
#     return count, select_index
#
# def select_avalanches_random(avalanches, sensibility=10, count=None, seed=123):
#     np.random.seed(seed)
#     concate = np.concatenate(avalanches)
#     sum = np.sum(concate, axis=0)
#     if count is None:
#         count = int(sum.min())+1  # value to get for all regions
#     index = np.arange(concate.shape[0])
#     np.random.shuffle(index)
#     select_index = [index[0]]
#     count_nb_region = np.zeros_like(concate[select_index[-1]])+concate[select_index[-1]]
#     i = 1
#     while np.any(np.abs(count_nb_region - count) > sensibility)\
#             and i < index.shape[0]:
#         if np.all(concate[index[i]]+count_nb_region - count <= sensibility):
#             select_index.append(index[i])
#             count_nb_region += concate[select_index[-1]]
#         i+=1
#     print(np.abs(np.unique(concate[select_index].sum(axis=0)) - count))
#     return count, select_index

def select_avalanches(avalanches, count=None):
    concate = np.concatenate(avalanches)
    sum = np.sum(concate, axis=0)
    if count is None:
        count = int(sum.min())+1  # value to get for all regions
    reg_1 = sum.argmin()  # select first region
    # start selection
    condition = concate[:, reg_1] == 1
    index_cond_1 = np.where(condition)[0]
    nb_reg = np.min([count, int(np.sum(concate[index_cond_1], axis=0)[reg_1])])
    select_index = index_cond_1[:nb_reg]
    not_select_index = np.concatenate((np.where(np.logical_not(condition))[0],))
    remove_index = np.where(condition)[0][nb_reg:]
    assert(select_index.shape[0] + remove_index.shape[0] + not_select_index.shape[0] == concate.shape[0])
    # if (select_index.shape[0] + remove_index.shape[0] + not_select_index.shape[0] != concate.shape[0]):
    #     print(select_index.shape[0] + remove_index.shape[0] + not_select_index.shape[0])
    #     print(concate.shape[0])

    for i in range(1, concate.shape[1]):
        print(i)
        # second region
        reg_2 = np.sum(concate[not_select_index], axis=0).argsort()[i]
        nb_reg_2 = int(count - np.sum(concate[select_index], axis=0)[reg_2])
        cond_2 = np.logical_or(concate[:, reg_2] == 1, condition)
        select_index_2 = np.where(np.logical_and(concate[:, reg_2] == 1, np.logical_not(condition)))[0][:nb_reg_2]
        select_index = np.concatenate((select_index, select_index_2))
        remove_index = np.concatenate((remove_index, not_select_index[np.where([(not(i in select_index_2) and concate[i, reg_2] == 1 ) for i in not_select_index ])]))
        not_select_index = np.where(np.logical_not(cond_2))[0]
        condition = cond_2
        assert (select_index.shape[0] + remove_index.shape[0] + not_select_index.shape[0] == concate.shape[0])
        # if (select_index.shape[0] + remove_index.shape[0] + not_select_index.shape[0] != concate.shape[0]):
        #     print(select_index.shape[0] + remove_index.shape[0] + not_select_index.shape[0])
        #     print(concate.shape[0])
        #     break
    print(count, np.abs(np.unique(concate[select_index].sum(axis=0)) - count))
    return count, select_index

def reshape_avalanches(avalanches, select_index):
    new_avalanches = []
    begin = 0
    for avalanche in avalanches:
        end = begin + len(avalanche)
        index = np.where(np.logical_and(select_index > begin, select_index < end))[0]
        new_avalanches.append(avalanche[select_index[index]-begin])
        begin = end
    return new_avalanches