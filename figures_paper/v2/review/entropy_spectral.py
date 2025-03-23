import matplotlib.pyplot as plt
import numpy as np

entropy = {
'Phate' :  5.625,
'3' :  5.218,
'4' :  5.337,
'5' :  5.441,
'6' :  5.520,
'7' :  5.541,
'8' :  5.558,
'9' :  5.762,
'10' :  5.757,
'11' :  5.757,
'12' :  5.757,
'13' :  5.757,
}
values = np.array(list(entropy.values()))

plt.figure()
plt.plot(values[0], 'xr'),
plt.plot(np.arange(1, len(values)), values[1:], 'xb'),
plt.xticks(np.arange(0, values.shape[0]), entropy.keys(), rotation=90)
plt.ylabel('entropy')
plt.xlabel('    spectral clustering')
plt.show()