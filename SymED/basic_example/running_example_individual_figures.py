import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import symbolic_algorithms
import numpy as np
import matplotlib.pyplot as plt
from tslearn.metrics import dtw as dtw
import queue

import matplotlib.patheffects as pe
from pathlib import Path
plt.rcParams.update({'font.size': 12})

"""
Adapted Version of the basic example from [1] to generate individual figures for the paper.

References
------
[1] S. Elsworth and S. Güttel. ABBA: Aggregate Brownian bridge-based
approximation of time series, MIMS Eprint 2019.11
(http://eprints.maths.manchester.ac.uk/2712/), Manchester
Institute for Mathematical Sciences, The University of Manchester, UK, 2019.
"""



output_path = "../../output/running_example_individual_figures/"
Path(output_path).mkdir(parents=True, exist_ok=True)

tol = 0.4
alpha = 0.02
c_method = "kmeans"
scl = 0.0
min_k = 3


c1 = (0.968,0.867,0.631)
c2 = (0.961,0.737,0.639)
c3 = (0.2,0.678,1)
colors = [c1, c2, c3]


np.random.seed(0)

t1 = np.arange(0, 10, 0.5)              # up
t2 = 9.5*np.ones(50)                    # flat
t3 = 10.5*np.ones(50)                    # flat
t4 = np.arange(10, 20, 0.5)             # up
t5 = np.arange(20, 0, -1)               # down
t6 = np.arange(0, 10, 0.5)              # up
t7 = 10*np.ones(50)                     # flat
t_all = [t1, t2, t3, t4, t5, t6, t7]
ts = np.hstack(t_all)
t_rand = 0.5 * np.random.randn(len(ts))
ts = ts + t_rand
mean = np.mean(ts)
std = np.std(ts)
ts_norm = np.divide(ts - mean, std)

print(f'###### SymED running example ######')
online_result_queue = queue.Queue()
abba_online_result, abba_historical_result, abbao = symbolic_algorithms.symed(ts, tol, c_method=c_method, scl=scl, alpha=alpha, min_k=min_k, online_result_queue=online_result_queue)
reconstruction_online_normed = np.divide(abba_online_result.reconstruction - mean, std)
reconstruction_historical_normed = np.divide(abba_historical_result.reconstruction - mean, std)
pieces_all = abbao.pieces

rows = 2
cols = 5

row = 0
col = 0

figsize = (3,3)

while True:
    res = online_result_queue.get()

    if isinstance(res, type(None)):
        break

    if len(res.pieces) in [1, 2, 3, 4, 5, 6]:
        continue

    ts_reconstructed = abbao.inverse_compress(ts[0], res.pieces)

    len_cum_sum = np.cumsum(res.pieces[:, 0])
    inc_cum_sum = np.cumsum(res.pieces[:, 1]) + ts[0]

    fig = plt.figure(figsize=figsize)
    ### sender figure
    plt.xlim([-11.45, 240.45])
    plt.ylim([min(ts) - 1, max(ts) + 3])
    plt.plot(range(len(ts))[:res.i], ts[:res.i], c="black", label='Original', linewidth=1.0)
    plt.plot(range(len(ts))[:res.i], ts_reconstructed[:res.i], linestyle='--', color="red", label='SymED - sender', linewidth=1.0)
    plt.scatter(len_cum_sum, inc_cum_sum, color='red', s=15)

    for j in range(len(res.pieces)):
        if j == 0:
            x = np.round(len_cum_sum[j]) + 4
            y = inc_cum_sum[j] - 1.8
        elif j == 1:
            x = np.round(len_cum_sum[j]) + 4
            y = inc_cum_sum[j] - 0.6
        elif j == 2:
            x = np.round(len_cum_sum[j]) + 4
            y = inc_cum_sum[j] - 1.5
        elif j == 3:
            x = np.round(len_cum_sum[j]) + 4
            y = inc_cum_sum[j] + 0.7
        elif j == 4 or j == 5:
            x = np.round(len_cum_sum[j]) + 2
            y = inc_cum_sum[j] + 1.0
        elif j == 6:
            x = np.round(len_cum_sum[j]) + 2
            y = inc_cum_sum[j] + 0.0
        elif j == 8:
            x = np.round(len_cum_sum[j]) + 5
            y = inc_cum_sum[j] + 0.2
        elif j == 10:
            x = np.round(len_cum_sum[j]) - 25
            y = inc_cum_sum[j] - 2.5
        else:
            x = np.round(len_cum_sum[j]) + 2
            y = inc_cum_sum[j] + 0.5
        plt.annotate(f"t{j}", (x, y), fontsize=15)

    xtick_labels = plt.gca().get_xticklabels()
    plt.setp(xtick_labels, visible=False)
    if col == 0:
        plt.ylabel("value")
        plt.legend()
    else:
        ytick_labels = plt.gca().get_yticklabels()
        plt.setp(ytick_labels, visible=False)

    fig.savefig(f"{output_path}/running_example_sender_{chr(col + 97)}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{output_path}/running_example_sender_{chr(col + 97)}.pdf", dpi=300, bbox_inches="tight")
    plt.clf()
    plt.close()

    ### receiver figure
    fig = plt.figure(figsize=figsize)
    plt.xlim([-11.45, 240.45])
    plt.ylim([min(ts) - 1, max(ts) + 3])
    plt.plot(len_cum_sum, inc_cum_sum, linestyle='-.', color="black", label='SymED - receiver', linewidth=1.0)

    x2_ticks = []
    x2_labels = []
    len_cum_sum_with_start = np.hstack([[0], len_cum_sum[:]])
    inc_cum_sum_with_start = np.hstack([[0], inc_cum_sum[:]])

    for j in range(len(res.pieces)):
        c = colors[ord(res.string[j]) - ord('a')]
        if j == 0:
            x = np.round(len_cum_sum[j] - res.pieces[j, 0] / 2) + 4.0
            y = inc_cum_sum[j] - res.pieces[j, 1] / 2 - 1.8
        elif j == 1:
            x = np.round(len_cum_sum[j] - res.pieces[j, 0] / 2) + 4.0
            y = inc_cum_sum[j] - res.pieces[j, 1] / 2 - 1.5
        elif j == 2:
            x = np.round(len_cum_sum[j] - res.pieces[j, 0] / 2) + 4.0
            y = inc_cum_sum[j] - res.pieces[j, 1] / 2 - 0.6
        elif j == 3:
            x = np.round(len_cum_sum[j] - res.pieces[j, 0] / 2) + 4.0
            y = inc_cum_sum[j] - res.pieces[j, 1] / 2 + 0.7
        elif j == 4:
            x = np.round(len_cum_sum[j] - res.pieces[j, 0] / 2) + 4.0
            y = inc_cum_sum[j] - res.pieces[j, 1] / 2 - 0.5
        elif j == 5:
            x = np.round(len_cum_sum[j] - res.pieces[j, 0] / 2) - 7.0
            y = inc_cum_sum[j] - res.pieces[j, 1] / 2 + 0.5
        elif j == 6:
            x = np.round(len_cum_sum[j] - res.pieces[j, 0] / 2) - 7.0
            y = inc_cum_sum[j] - res.pieces[j, 1] / 2 + 0.5
        elif j == 7:
            x = np.round(len_cum_sum[j] - res.pieces[j, 0] / 2) - 20.0
            y = inc_cum_sum[j] - res.pieces[j, 1] / 2 - 0.5
        elif j == 8:
            x = np.round(len_cum_sum[j] - res.pieces[j, 0] / 2) + 2.0
            y = inc_cum_sum[j] - res.pieces[j, 1] / 2 - 0.5
        elif j == 9:
            x = np.round(len_cum_sum[j] - res.pieces[j, 0] / 2) + 4.0
            y = inc_cum_sum[j] - res.pieces[j, 1] / 2 - 0.5
        elif j == 10:
            x = np.round(len_cum_sum[j] - res.pieces[j, 0] / 2) - 7.0
            y = inc_cum_sum[j] - res.pieces[j, 1] / 2 + 0.5
        else:
            x = np.round(len_cum_sum[j] - res.pieces[j, 0] / 2) - 7.0
            y = inc_cum_sum[j] - res.pieces[j, 1] / 2 - 0.5
        plt.annotate(res.string[j], (x, y), color=c, fontweight="bold", fontsize=15, path_effects=[pe.withStroke(linewidth=1, foreground="black")])
        plt.scatter(len_cum_sum[j], inc_cum_sum[j], color='black', s=15)

    plt.xlabel("time point")
    if col == 0:
        plt.ylabel("value")
        plt.legend()
    else:
        ytick_labels = plt.gca().get_yticklabels()
        plt.setp(ytick_labels, visible=False)

    fig.savefig(f"{output_path}/running_example_receiver_{chr(col + cols + 97)}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{output_path}/running_example_receiver_{chr(col + cols + 97)}.pdf", dpi=300, bbox_inches="tight")
    plt.clf()
    plt.close()
    col += 1

print(abba_online_result.conversion)
print('Compression rate (length):', round(abba_online_result.compression_rate_length, 3))
print('Compression rate (bytes):', round(abba_online_result.compression_rate_bytes, 3))
print('dtw pieces:', round(dtw(ts_norm, reconstruction_online_normed), 3))
print('dtw symbols:',  round(dtw(ts_norm,reconstruction_historical_normed), 3))
