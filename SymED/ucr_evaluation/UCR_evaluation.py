import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import symbolic_algorithms
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from tslearn.metrics import dtw as dtw
import os
import csv
from pathlib import Path
import multiprocessing
from timeit import default_timer as timer
import pandas as pd
import signal
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # kmeans warning
plt.rcParams.update({'font.size': 15})

def comparison_figure(tols, scores, colors, markers, y_lims, y_label, legend, output_path, figure_name):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.grid(True)

    ax1.set_xticks(tols)
    if not isinstance(y_lims, type(None)):
        ax1.set_ylim(y_lims)
    for score, color, marker in zip(scores, colors, markers):
        ax1.plot(tols, score, c=color, marker=marker)
    ax1.set_xlabel("tol")
    ax1.set_ylabel(y_label)
    ax1.legend(legend)

    plt.setp(ax1.get_xticklabels()[::2], visible=False)
    plt.gca().set_ylim(bottom=0)
    fig.savefig(f"{output_path}/{figure_name}.pdf")
    fig.savefig(f"{output_path}/{figure_name}.png")
    plt.clf()
    plt.close()

def run_simulation_for_tol(ind, tols, abba_scores_mean_shared, abba_times_mean_shared,
                           abba_times_mean_per_symbol_shared, alpha, c_method, file_names, scl,
                           symed_offline_scores_mean_shared, symed_offline_times_mean_per_symbol_shared,
                           symed_online_scores_mean_shared,
                           symed_online_times_mean_shared, tol, datasets, output_path, min_k, run_in_parallel):
    if run_in_parallel:
        abba_scores_mean = np.ctypeslib.as_array(abba_scores_mean_shared).reshape((len(tols), 12))
        symed_online_scores_mean = np.ctypeslib.as_array(symed_online_scores_mean_shared).reshape((len(tols), 12))
        symed_offline_scores_mean = np.ctypeslib.as_array(symed_offline_scores_mean_shared).reshape((len(tols), 12))
        abba_times_mean = np.ctypeslib.as_array(abba_times_mean_shared).reshape((len(tols), 9))
        abba_times_mean_per_symbol = np.ctypeslib.as_array(abba_times_mean_per_symbol_shared).reshape((len(tols), 9))
        symed_online_times_mean = np.ctypeslib.as_array(symed_online_times_mean_shared).reshape((len(tols), 9))
        symed_offline_times_mean_per_symbol = np.ctypeslib.as_array(symed_offline_times_mean_per_symbol_shared).reshape(
            (len(tols), 3))
    else:
        abba_scores_mean = abba_scores_mean_shared
        symed_online_scores_mean = symed_online_scores_mean_shared
        symed_offline_scores_mean = symed_offline_scores_mean_shared
        abba_times_mean = abba_times_mean_shared
        abba_times_mean_per_symbol = abba_times_mean_per_symbol_shared
        symed_online_times_mean = symed_online_times_mean_shared
        symed_offline_times_mean_per_symbol = symed_offline_times_mean_per_symbol_shared

    print(f"--- ABBA --- tol={round(tol, 2)} c_method={c_method} scl={scl} alpha={alpha}")
    abba_scores = np.empty([0, 12])
    times_sender = []
    times_receiver = []
    times_combined = []
    times_sender_per_symbol = []
    times_receiver_per_symbol = []
    times_combined_per_symbol = []
    for i, d in enumerate(datasets):
        print(file_names[i])
        abba_scores_per_dataset = np.empty([0, 12])
        times_sender_per_dataset = []
        times_receiver_per_dataset = []
        times_combined_per_dataset = []
        times_sender_per_symbol_per_dataset = []
        times_receiver_per_symbol_per_dataset = []
        times_combined_per_symbol_per_dataset = []
        for j, ts in enumerate(d):
            mean = np.mean(ts)
            std = np.std(ts)
            ts_norm = np.divide(ts - mean, std)

            # Sometimes ABBA will fail if min_k is too large, so we need to find the largest min_k that works
            selected_k = min_k
            while min_k > 0:
                try:
                    # symbolic_algorithms standardize/de-standardize internally
                    abba_result, abba, _, pieces = symbolic_algorithms.abba(ts, tol, c_method=c_method, scl=scl, min_k=selected_k)
                    break
                except ValueError as e:
                    selected_k -= 1

            compressed_ts = np.divide(abba.inverse_compress(ts[0], pieces) - mean, std)
            string = abba_result.conversion
            reconstruction = np.divide(abba_result.reconstruction - mean, std)

            if j == 0:
                # names of the entries in the list
                L = ['index', 'N', 'n', 'euclid(T,hat{T})', 'k', 'dtw(hat{T},tilde{T})', 'tol', 'sqrt{N}tol', 'MSE',
                     'comp_length', 'comp_bytes', 'dtw(T, tilde{T})']

            L = [j, len(ts), len(string), np.linalg.norm(ts_norm - compressed_ts), len(set(string)),
                 dtw(compressed_ts, reconstruction),
                 abba.tol, np.sqrt(len(ts)) * abba.tol, np.linalg.norm(ts_norm - reconstruction),
                 abba_result.compression_rate_length,
                 abba_result.compression_rate_bytes, dtw(ts_norm, reconstruction)]
            abba_scores_per_dataset = np.vstack([abba_scores_per_dataset, L])
            times_sender_per_dataset = times_sender_per_dataset + abba_result.times_sender
            times_receiver_per_dataset = times_receiver_per_dataset + abba_result.times_receiver
            times_combined_per_dataset = times_combined_per_dataset + abba_result.times_combined
            times_sender_per_symbol_per_dataset = times_sender_per_symbol_per_dataset + [t / len(string) for t in
                                                                                         abba_result.times_sender]
            times_receiver_per_symbol_per_dataset = times_receiver_per_symbol_per_dataset + [t / len(string) for t in
                                                                                             abba_result.times_receiver]
            times_combined_per_symbol_per_dataset = times_combined_per_symbol_per_dataset + [t / len(string) for t in
                                                                                             abba_result.times_combined]
        abba_scores = np.vstack([abba_scores, np.mean(abba_scores_per_dataset, axis=0)])
        times_sender.append(sum(times_sender_per_dataset) / len(times_sender_per_dataset))
        times_receiver.append(sum(times_receiver_per_dataset) / len(times_receiver_per_dataset))
        times_combined.append(sum(times_combined_per_dataset) / len(times_combined_per_dataset))
        times_sender_per_symbol.append(
            sum(times_sender_per_symbol_per_dataset) / len(times_sender_per_symbol_per_dataset))
        times_receiver_per_symbol.append(
            sum(times_receiver_per_symbol_per_dataset) / len(times_receiver_per_symbol_per_dataset))
        times_combined_per_symbol.append(
            sum(times_combined_per_symbol_per_dataset) / len(times_combined_per_symbol_per_dataset))

    abba_times_mean[ind] = [np.mean(times_sender),
                            np.percentile(times_sender, 1),
                            np.percentile(times_sender, 99),
                            np.mean(times_receiver),
                            np.percentile(times_receiver, 1),
                            np.percentile(times_receiver, 99),
                            np.mean(times_combined),
                            np.percentile(times_combined, 1),
                            np.percentile(times_combined, 99)]
    abba_scores_mean[ind] = np.mean(abba_scores, axis=0)

    abba_times_mean_per_symbol[ind] = [t / abba_scores_mean[ind, 2] for t in abba_times_mean[ind]]

    print(f"--- SymED --- tol={round(tol, 2)} c_method={c_method} scl={scl} alpha={alpha}")
    symed_online_scores = np.empty([0, 12])
    symed_offline_scores = np.empty([0, 12])
    times_sender = []
    times_receiver = []
    times_combined = []
    times_receiver_offline_per_symbol = []

    for i, d in enumerate(datasets):
        print(file_names[i])
        symed_online_scores_per_dataset = np.empty([0, 12])
        symed_offline_scores_per_dataset = np.empty([0, 12])
        times_sender_per_dataset = []
        times_receiver_per_dataset = []
        times_combined_per_dataset = []
        times_receiver_offline_per_symbol_per_dataset = []
        for j, ts in enumerate(d):
            mean = np.mean(ts)
            std = np.std(ts)
            ts_norm = np.divide(ts - mean, std)
            symed_online_result, symed_offline_result, symed = symbolic_algorithms.symed(ts, tol, verbose=0, alpha=alpha,
                                                                     c_method=c_method, scl=scl, min_k=min_k)
            compressed_ts = np.divide(symed.inverse_compress(ts[0], symed.pieces) - mean, std)
            string = symed_online_result.conversion
            symbolic_ts = np.divide(symed_online_result.reconstruction - mean, std)
            reconstruction_online = np.divide(symed_online_result.reconstruction - mean, std)
            reconstruction_offline = np.divide(symed_offline_result.reconstruction - mean, std)

            if j == 0:
                # names of the entries in the list
                L_online = ['index', 'N', 'n', 'euclid(T,hat{T})', 'k', 'dtw(hat{T},tilde{T})', 'tol', 'sqrt{N}tol',
                            'MSE', 'comp_length', 'comp_bytes', 'dtw(T, tilde{T})']
                L_offline = ['index', 'N', 'n', 'euclid(T,hat{T})', 'k', 'dtw(hat{T},tilde{T})', 'tol', 'sqrt{N}tol',
                          'MSE', 'comp_length', 'comp_bytes', 'dtw(T, tilde{T})']

            L_online = [j, len(ts_norm), len(string), np.linalg.norm(ts_norm - compressed_ts), len(set(string)),
                        dtw(compressed_ts, symbolic_ts),
                        symed.tol, np.sqrt(len(ts_norm)) * symed.tol, np.linalg.norm(ts_norm - reconstruction_online),
                        symed_online_result.compression_rate_length,
                        symed_online_result.compression_rate_bytes, dtw(ts_norm, reconstruction_online)]
            symed_online_scores_per_dataset = np.vstack([symed_online_scores_per_dataset, L_online])

            L_offline = [j, len(ts_norm), len(string), np.linalg.norm(ts_norm - compressed_ts), len(set(string)),
                      dtw(compressed_ts, symbolic_ts),
                      symed.tol, np.sqrt(len(ts_norm)) * symed.tol, np.linalg.norm(ts_norm - reconstruction_offline),
                      symed_offline_result.compression_rate_length,
                      symed_offline_result.compression_rate_bytes, dtw(ts_norm, reconstruction_offline)]
            symed_offline_scores_per_dataset = np.vstack([symed_offline_scores_per_dataset, L_offline])

            times_sender_per_dataset = times_sender_per_dataset + symed_online_result.times_sender
            times_receiver_per_dataset = times_receiver_per_dataset + symed_online_result.times_receiver
            times_combined_per_dataset = times_combined_per_dataset + symed_online_result.times_combined
            times_receiver_offline_per_symbol_per_dataset = times_receiver_offline_per_symbol_per_dataset + [
                t / len(string) for t in
                symed_offline_result.times_receiver]

        symed_online_scores = np.vstack([symed_online_scores, np.mean(symed_online_scores_per_dataset, axis=0)])
        symed_offline_scores = np.vstack([symed_offline_scores, np.mean(symed_offline_scores_per_dataset, axis=0)])
        times_sender.append(sum(times_sender_per_dataset) / len(times_sender_per_dataset))
        times_receiver.append(sum(times_receiver_per_dataset) / len(times_receiver_per_dataset))
        times_combined.append(sum(times_combined_per_dataset) / len(times_combined_per_dataset))
        times_receiver_offline_per_symbol.append(sum(times_receiver_offline_per_symbol_per_dataset) / len(
            times_receiver_offline_per_symbol_per_dataset))

    symed_online_times_mean[ind] = [np.mean(times_sender),
                                    np.percentile(times_sender, 1),
                                    np.percentile(times_sender, 99),
                                    np.mean(times_receiver),
                                    np.percentile(times_receiver, 1),
                                    np.percentile(times_receiver, 99),
                                    np.mean(times_combined),
                                    np.percentile(times_combined, 1),
                                    np.percentile(times_combined, 99)]
    symed_offline_times_mean_per_symbol[ind] = [np.mean(times_receiver_offline_per_symbol),
                                                np.percentile(times_receiver_offline_per_symbol, 1),
                                                np.percentile(times_receiver_offline_per_symbol, 99)]

    symed_online_scores_mean[ind] = np.mean(symed_online_scores, axis=0)
    symed_offline_scores_mean[ind] = np.mean(symed_offline_scores, axis=0)

def run_simulation_for_tols(c_method, scl, alpha, datasets, tols, file_names, output_path, run_in_parallel, min_k):
    if run_in_parallel:
        abba_scores_mean_shared = multiprocessing.RawArray('d', len(tols) * 12)
        symed_online_scores_mean_shared = multiprocessing.RawArray('d', len(tols) * 12)
        symed_offline_scores_mean_shared = multiprocessing.RawArray('d', len(tols) * 12)
        abba_times_mean_shared = multiprocessing.RawArray('d', len(tols) * 9)
        abba_times_mean_per_symbol_shared = multiprocessing.RawArray('d', len(tols) * 9)
        symed_online_times_mean_shared = multiprocessing.RawArray('d', len(tols) * 9)
        symed_offline_times_mean_per_symbol_shared = multiprocessing.RawArray('d', len(tols) * 3)

        jobs = []
        signal.signal(signal.SIGINT, lambda signum, frame: sigint_handler(signum, frame, jobs))
        for i, tol in enumerate(tols):
            p = multiprocessing.Process(target=run_simulation_for_tol, args=(
            i, tols, abba_scores_mean_shared, abba_times_mean_shared, abba_times_mean_per_symbol_shared, alpha,
            c_method, file_names, scl,
            symed_offline_scores_mean_shared, symed_offline_times_mean_per_symbol_shared,
            symed_online_scores_mean_shared,
            symed_online_times_mean_shared, tol, datasets, output_path, min_k, run_in_parallel))
            jobs.append(p)
            p.start()
            # Manually mange processes, as pools don't seem to work in nested functions
            if len(jobs) >= multiprocessing.cpu_count() / 2:
                for j in jobs:
                    j.join()
                jobs.clear()
        for j in jobs:
            j.join()

        abba_scores_mean = np.ctypeslib.as_array(abba_scores_mean_shared).reshape((len(tols), 12))
        symed_online_scores_mean = np.ctypeslib.as_array(symed_online_scores_mean_shared).reshape((len(tols), 12))
        symed_offline_scores_mean = np.ctypeslib.as_array(symed_offline_scores_mean_shared).reshape((len(tols), 12))
        abba_times_mean = np.ctypeslib.as_array(abba_times_mean_shared).reshape((len(tols), 9))
        abba_times_mean_per_symbol = np.ctypeslib.as_array(abba_times_mean_per_symbol_shared).reshape((len(tols), 9))
        symed_online_times_mean = np.ctypeslib.as_array(symed_online_times_mean_shared).reshape((len(tols), 9))
        symed_offline_times_mean_per_symbol = np.ctypeslib.as_array(symed_offline_times_mean_per_symbol_shared).reshape(
            (len(tols), 3))
    else:
        abba_scores_mean = np.full((len(tols), 12), np.inf)
        symed_online_scores_mean = np.full((len(tols), 12), np.inf)
        symed_offline_scores_mean = np.full((len(tols), 12), np.inf)
        abba_times_mean = np.full((len(tols), 9), np.inf)
        abba_times_mean_per_symbol = np.full((len(tols), 9), np.inf)
        symed_online_times_mean = np.full((len(tols), 9), np.inf)
        symed_offline_times_mean_per_symbol = np.full((len(tols), 3), np.inf)

        for i, tol in enumerate(tols):
            run_simulation_for_tol(
                i, tols, abba_scores_mean, abba_times_mean, abba_times_mean_per_symbol, alpha, c_method, file_names,
                scl,
                symed_offline_scores_mean, symed_offline_times_mean_per_symbol, symed_online_scores_mean,
                symed_online_times_mean,
                tol, datasets, output_path, min_k, run_in_parallel)

    figure_name_last_part = f"cm-{c_method}_scl-{scl}_a-{alpha}"
    figure_name_last_part = figure_name_last_part.replace('.', '-')

    with open(f'{output_path}/mean_values.txt', 'w') as f:

        # Reconstruction Error
        f.write(f"ABBA Reconstruction Error mean {np.mean(abba_scores_mean[:, 11])}\n")
        f.write(f"SymED online Reconstruction Error mean {np.mean(symed_online_scores_mean[:, 11])}\n")
        f.write(f"SymED offline Reconstruction Error mean {np.mean(symed_offline_scores_mean[:, 11])}\n")
        comparison_figure(tols, np.vstack(
            [abba_scores_mean[:, 11], symed_online_scores_mean[:, 11], symed_offline_scores_mean[:, 11]]),
                          ["black", "red", "orange"], ["s", "o", "v"], None, "DTW",
                          ["ABBA", "SymED - pieces", "SymED - symbols"], output_path, f"RE_{figure_name_last_part}")

        # Compression Rate Length
        f.write(f"ABBA Dimension Reduction Rate mean {np.mean(abba_scores_mean[:, 9])}\n")
        f.write(f"SymED Dimension Reduction Rate mean {np.mean(symed_online_scores_mean[:, 9])}\n")
        comparison_figure(tols, np.vstack([abba_scores_mean[:, 9], symed_online_scores_mean[:, 9]]),
                          ["black", "red"], ["s", "o"], None, "dimension reduction rate",
                          ["ABBA", "SymED"], output_path, f"DRR_{figure_name_last_part}")

        # Compression Rate Bytes
        f.write(f"ABBA Compression Rate mean {np.mean(abba_scores_mean[:, 10])}\n")
        f.write(f"SymED Compression Rate mean {np.mean(symed_online_scores_mean[:, 10])}\n")
        comparison_figure(tols, np.vstack([abba_scores_mean[:, 10], symed_online_scores_mean[:, 10]]),
                          ["black", "red"], ["s", "o"], None, "compression rate",
                          ["ABBA", "SymED"], output_path, f"CR_{figure_name_last_part}")

        # Total Latency ABBA and SymED
        f.write(f"ABBA Offline Latency mean {np.mean(abba_times_mean[:, 6])}\n")
        f.write(f"SymED Offline Latency mean {np.mean(symed_online_times_mean[:, 6] * symed_online_scores_mean[:, 2])}\n")
        comparison_figure(tols, np.vstack(
            [abba_times_mean[:, 6], symed_online_times_mean[:, 6] * symed_online_scores_mean[:, 2]]),
                          ["black", "red"], ["s", "o"], None, "computational latency [s]",
                          ["ABBA", "SymED"], output_path, f"CLT_{figure_name_last_part}"
                          )

        # Latencies for SymED
        f.write(f"SymED Online Latency Sender mean {np.mean(symed_online_times_mean[:, 0])}\n")
        f.write(f"SymED Online Latency Receiver mean {np.mean(symed_online_times_mean[:, 3])}\n")
        comparison_figure(tols, np.vstack([symed_online_times_mean[:, 0], symed_online_times_mean[:, 3]]) * 1000,
                          ["red", "orange"], ["o", "v"], None, "computational latency [ms]",
                          ["SymED - Sender", "SymED - Receiver"], output_path, f"CL_SymED_{figure_name_last_part}"
                          )

    with open(f'{output_path}/mean_values.txt', 'r') as f:
        file_contents = f.read()
        print(file_contents)

def sigint_handler(signum, frame, jobs):
    print("Caught SIGINT, terminating child processes...")
    for p in jobs:
        p.terminate()
    exit(1)

if __name__ == '__main__':
    """
    Adapted Version of the UCR Suite Evaluation from [1] to generate individual figures for the paper.

    References
    ------
    [1] S. Elsworth and S. Güttel. ABBA: Aggregate Brownian bridge-based
    approximation of time series, MIMS Eprint 2019.11
    (http://eprints.maths.manchester.ac.uk/2712/), Manchester
    Institute for Mathematical Sciences, The University of Manchester, UK, 2019.
    """

    output_path = '../../output/UCR_evaluation'
    datadir = './../../datasets/UCRArchive_2018/'
    max_number_of_datasets = sys.maxsize
    min_len_of_time_series = 1000  # per time series within the dataset
    # parallelize evaluation, not recommended for the computational latency metric, but doesn't affect other metrics
    run_outer_loops_in_parallel = False  # run simulations for different ABBA, SymED parameters (other than tol) in parallel
    run_inner_loops_in_parallel = False  # run simulations for all tol values in parallel

    tols = np.arange(0.1, 2.1, 0.1)     # tolerance values for ABBA and SymED
    alphas = [10 ** x for x in [-2]]    # alpha values for SymED
    min_k = 3

    # Select some ts from UCR Suite
    # ---------------------------------------------------------------------------- #
    dataset_info = []
    datasets = []
    folder_list = next(os.walk(datadir))[1]
    Path(output_path).mkdir(parents=True, exist_ok=True)
    i = 0
    for folder in folder_list:
        for root, dirs, files in os.walk(os.path.join(datadir, folder)):
            for file in files:
                if file.endswith('_TEST.tsv'):
                    dataset = []
                    classes = []
                    with open(os.path.join(root, file)) as tsvfile:
                        tsvfile = csv.reader(tsvfile, delimiter='\t')
                        for j, row in enumerate(tsvfile):
                            if len(row) >= min_len_of_time_series:
                                if j == 0:
                                    print(i, "Suitable file", file)
                                    i += 1

                                c = row[0]
                                if c in classes:
                                    continue

                                classes.append(c)
                                ts = [float(i) for i in row]  # convert to list of floats
                                ts = np.array(ts[1:])  # remove class information
                                # remove NaN from time series
                                ts = ts[~np.isnan(ts)]
                                dataset.append(ts)
                            else:
                                break
                    if len(dataset) > 0 and len(datasets) < max_number_of_datasets:
                        avg_len = sum(len(ts) for ts in dataset) / len(dataset)
                        if avg_len >= min_len_of_time_series:
                            flat_dataset = [d for ts in dataset for d in ts]

                            dataset_info.append({
                                'dataset': file[:-len('_TEST.tsv')],
                                'size': round(len(dataset)),
                                'length': round(avg_len),
                            })
                            datasets.append(dataset)

    df = pd.DataFrame(dataset_info)
    df = df.sort_values('dataset')
    file_names = df['dataset']
    df.to_csv(f'{output_path}/datasets.csv', index=False)

    print(
        f"Selected {len(df)} datasets with a total of {np.sum(df['size'])} time series with an average time series length of {np.sum(df['size'] * df['length']) / np.sum(df['size'])}")

    # perform test
    # ---------------------------------------------------------------------------- #

    # grid evaluation of parameters (used for preliminary evaluation with multiple parameters)
    time_start = timer()
    if run_outer_loops_in_parallel:
        jobs = []
        for c_method, scl in [("kmeans", 1.0)]:
            for alpha in alphas:
                p = multiprocessing.Process(target=run_simulation_for_tols, args=(
                c_method, scl, alpha, datasets, tols, file_names, output_path, run_inner_loops_in_parallel, min_k))
                jobs.append(p)
                p.start()
        for j in jobs:
            j.join()
    else:
        for c_method, scl in [("kmeans", 1.0)]:
            for alpha in alphas:
                run_simulation_for_tols(c_method, scl, alpha, datasets, tols, file_names, output_path,
                                        run_inner_loops_in_parallel, min_k)

    print(f"Finished in {round(timer() - time_start, 2)} seconds")
