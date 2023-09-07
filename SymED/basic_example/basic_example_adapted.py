import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import symbolic_algorithms
import numpy as np
import matplotlib.pyplot as plt
from tslearn.metrics import dtw as dtw
from pathlib import Path
plt.rcParams.update({'font.size': 15})

"""
Adapted Version of the basic example from [1] for SymED.

References
------
[1] S. Elsworth and S. Güttel. ABBA: Aggregate Brownian bridge-based
approximation of time series, MIMS Eprint 2019.11
(http://eprints.maths.manchester.ac.uk/2712/), Manchester
Institute for Mathematical Sciences, The University of Manchester, UK, 2019.
"""


output_path = "../../output/basic_example_adapted/"
Path(output_path).mkdir(parents=True, exist_ok=True)

tol = 0.4
alpha = 0.02
c_method = "kmeans"
scl = 0
min_k = 3

np.random.seed(0)
# Construct time series
t1 = np.arange(0, 10, 0.5)              # up
t2 = 9.5*np.ones(50)                    # flat
t3 = 10.5*np.ones(50)                    # flat
t4 = np.arange(10, 20, 0.5)             # up
t5 = np.arange(20, 0, -1)               # down
t6 = np.arange(0, 10, 0.5)              # up
t7 = 10*np.ones(50)                     # flat
ts = np.hstack([t1, t2, t3, t4, t5, t6, t7])
ts = ts + 0.5*np.random.randn(len(ts))

mean = np.mean(ts)
std = np.std(ts)
ts_norm = np.divide(ts - mean, std)



print('###### ABBA ######')
abba_result, abba, centers, _ = symbolic_algorithms.abba(ts, tol, c_method=c_method, scl=scl, min_k=min_k)
reconstruction_normed = np.divide(abba_result.reconstruction - mean, std)
pieces = abba.inverse_digitize(abba_result.conversion, centers)
pieces = abba.quantize(pieces)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_ylim([min(ts)-1, max(ts)+3])
ax1.plot(range(len(ts)), ts, c="black")
ax1.plot(range(len(ts)), abba_result.reconstruction, linestyle='--', color="red")
ax1.set_xlabel('time point')
ax1.set_ylabel('value')
ax1.legend(['Original', 'ABBA'])

x2_ticks = []
x2_labels = []
for i in range(len(pieces)):
    x = round(np.sum(pieces[0:i+1, 0]))
    x2_ticks.append(x)
    x2_labels.append(abba_result.conversion[i])

ax2 = ax1.twiny()
ax2.set_xbound(ax1.get_xbound())
ax2.set_xticks(x2_ticks)
ax2.set_xticklabels(x2_labels)
ax2.set_xlabel("symbols")

fig.savefig(f"{output_path}/basic_example_adapted_abba.pdf")
fig.savefig(f"{output_path}/basic_example_adapted_abba.png")
plt.show()
plt.clf()
plt.close()

print(abba_result.conversion)
print('Compression rate (length):', round(abba_result.compression_rate_length, 3))
print('Compression rate (bytes):', round(abba_result.compression_rate_bytes, 3))
print('dtw (normed):', round(dtw(ts_norm, reconstruction_normed), 3))
print()



print('###### SymED ######')
symed_online_result, symed_offline_result, symed = symbolic_algorithms.symed(ts, tol, c_method=c_method, scl=scl, alpha=alpha, min_k=min_k)
reconstruction_online_normed = np.divide(symed_online_result.reconstruction - mean, std)
reconstruction_historical_normed = np.divide(symed_offline_result.reconstruction - mean, std)
pieces = symed.pieces

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_ylim([min(ts)-1, max(ts)+3])
ax1.plot(range(len(ts)), ts, c="black")
ax1.plot(range(len(ts)), symed_online_result.reconstruction, linestyle='--', color="red")
ax1.plot(range(len(ts)), symed_offline_result.reconstruction, linestyle='-.', color="orange")
ax1.set_xlabel('time point')
ax1.set_ylabel('value')
ax1.legend(['Original', 'SymED - pieces', 'SymED - symbols'])

x2_ticks = []
x2_labels = []
for i in range(len(pieces)):
    x = round(np.sum(pieces[0:i+1, 0]))
    x2_ticks.append(x)
    if i >= 4:
        x2_labels.append(symed_online_result.conversion[i])
    elif i == 0:
        x2_labels.append(str(symed_online_result.conversion[0:4]))
    else:
        x2_labels.append('')

ax2 = ax1.twiny()
ax2.set_xbound(ax1.get_xbound())
ax2.set_xticks(x2_ticks)
ax2.set_xticklabels(x2_labels)
ax2.set_xlabel("symbols")

fig.savefig(f"{output_path}/basic_example_adapted_symed.pdf", dpi=300)
fig.savefig(f"{output_path}/basic_example_adapted_symed.png", dpi=300)
plt.show()
plt.clf()
plt.close()

print(symed_online_result.conversion)
print('Compression rate (length):', round(symed_online_result.compression_rate_length, 3))
print('Compression rate (bytes):', round(symed_online_result.compression_rate_bytes, 3))
print('dtw pieces:', round(dtw(ts_norm, reconstruction_online_normed), 3))
print('dtw symbols:',  round(dtw(ts_norm,reconstruction_historical_normed), 3))
