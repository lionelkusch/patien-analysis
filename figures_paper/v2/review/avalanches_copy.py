import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm, zscore, gaussian_kde


# Preparation data for the pipeline
path_data = os.path.dirname(os.path.realpath(__file__)) + '/../../../data/'
f = h5py.File(path_data + 'serie_Melbourne.mat', 'r')
struArray = f['D']
data = {}
Nsubs = 44
nregions = 90
for i in range(Nsubs):
    data['%d' % i] = np.swapaxes(f[struArray[i, 0]][:nregions, :], 0, 1)
selected_subjects = ['43', '39', '38', '35', '34', '29', '26', '21', '20', '19', '18', '17', '15', '13', '9', '8', '6', '5']
all_data = np.concatenate([zscore(data[i], axis=1) for i in selected_subjects])
x = np.linspace(-6, 6, 1000)

plt.figure(figsize=(10, 6))
kk = []
# Compute the KDE for each region and store the values
for region_index in range(all_data.shape[1]):
    print(region_index)
    kernel = gaussian_kde(all_data[:, region_index])
    kde_values = kernel(x)
    plt.plot(x, kde_values, alpha=0.5, color='c')
    kk.append(kde_values)

# Compute the mean KDE values across regions
mean_kde_values = np.mean(kk, axis=0)

# Plot the mean KDE curve
plt.plot(x, mean_kde_values, color='blue', lw=2, label='Mean KDE')

# Generate values for the Gaussian curve
gaussian = norm.pdf(x, 0, 1)

# Plot the Gaussian curve
plt.plot(x, gaussian, color='red', lw=2, label='Gaussian (mean=0, var=1)')
plt.legend()
# you can also use the log scale for y axis
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.xlabel('Signal Value')
plt.ylabel('Frequency')
plt.title(f'Distribution of Data')
plt.xlim(xmin=-6, xmax=6)
plt.ylim(ymin=1e-7, ymax=5)
plt.savefig(f'./figure/evaluation/all.png')
plt.close('all')


