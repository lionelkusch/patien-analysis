import matplotlib.pyplot as plt
import os
import numpy as np

path = os.path.dirname(os.path.realpath(__file__)) + "/../../paper/result/default/"
avalanches_pattern = np.load(path + "/avalanches.npy", allow_pickle=True)

plt.figure()
plt.bar(np.arange(0, 90, 1), np.sum(np.concatenate(avalanches_pattern), axis=0))
plt.xlim(xmin=-1, xmax=90)
plt.ylabel('number of time region\nis part of avalanches')
plt.xlabel('brain region')
plt.savefig('figure/SP_1_sum_regions.png')
plt.show()