import matplotlib.pyplot as plt
import os
import numpy as np

titlefont_size = 12.0
tickfont_size = 8.0
labelfont_size = 10.0
letter_font_size = 12

path = os.path.dirname(os.path.realpath(__file__)) + "/../../../paper/result/default/"
data_patient = np.load(path + "/histograms_region.npy")

data_null_1_0 = np.load(path + "/histograms_region_0.npy")
data_null_1_1 = np.load(path + "/histograms_region_1.npy")
data_null_2_0 = np.load(path + "/null_model/0_histograms_region.npy")
data_null_2_1 = np.load(path + "/null_model/1_histograms_region.npy")
for name, data in [ ('cluster_pattern.png', data_patient),
                    ('example_null_1_0.png', data_null_1_0),
                    ('example_null_1_1.png', data_null_1_1),
                    ('example_null_3_0.png', data_null_2_0),
                    ('example_null_3_1.png', data_null_2_1),
                    ]:

    plt.figure()
    plt.imshow(data, origin='lower')
    plt.tick_params('both', pad=1)
    plt.yticks([0, 3, 6], [1, 4, 7])
    plt.ylabel('# cluster')
    plt.xlabel('ROIs', {"fontsize": labelfont_size}, labelpad=0)
    plt.tick_params(which='both', labelsize=tickfont_size)
    plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_label_position("right")
    plt.savefig('figure/'+name)
    # plt.show()
    plt.close('all')