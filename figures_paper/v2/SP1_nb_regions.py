import matplotlib.pyplot as plt
import os
import numpy as np

path = os.path.dirname(os.path.realpath(__file__)) + "/../../paper/result/default/"
avalanches_pattern = np.load(path + "/avalanches.npy", allow_pickle=True)

label_size = 12.0
tickfont_size = 10.0
label_col_size = 8.0
linewidth = 0.5

plt.figure()
plt.bar(np.arange(0, 90, 1), np.sum(np.concatenate(avalanches_pattern), axis=0))
plt.xlim(xmin=-1, xmax=90)
plt.ylabel('cumulative activity in avalanches pattern', {"fontsize": label_size})
plt.xlabel('brain region', {"fontsize": label_size})
plt.subplots_adjust(left=0.11, top=0.98, right=0.98, )
plt.savefig('figure/SP_1_sum_regions.png')
plt.show()