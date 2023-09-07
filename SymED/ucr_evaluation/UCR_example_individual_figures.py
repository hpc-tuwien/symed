import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import symbolic_algorithms
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
import os
import csv
from pathlib import Path
import pandas as pd

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

    output_path = '../../output/UCR_example_individual_figures'
    datadir = '../../datasets/UCRArchive_2018/'
    alpha = 10**(-2)
    tol = 0.4
    scl = 1.0
    min_k = 3
    c_method = 'kmeans'

    # Select some ts from UCR Suite
    # ---------------------------------------------------------------------------- #
    dataset_info = []
    datasets = []
    folder_list = ['CinCECGTorso', 'HouseTwenty', 'StarLightCurves']
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
                        row = next(tsvfile)

                        ts = [float(i) for i in row]  # convert to list of floats
                        ts = np.array(ts[1:])  # remove class information
                        # remove NaN from time series
                        ts = ts[~np.isnan(ts)]
                        dataset.append(ts)

                    avg_len = sum(len(ts) for ts in dataset) / len(dataset)
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

    # perform test
    # ---------------------------------------------------------------------------- #

    rows = len(datasets)
    cols = 1


    for row, d in enumerate(datasets):

        fig = plt.figure(figsize=(10,1))

        ts = d[0]
        symed_online_result, symed_offline_result, symed = symbolic_algorithms.symed(ts, tol, verbose=0, alpha=alpha,
                                                                 c_method=c_method, scl=scl, min_k=min_k)
        plt.plot(range(len(ts)), ts, color='black', label='Original')
        plt.plot(range(len(symed_online_result.reconstruction)), symed_online_result.reconstruction, linestyle='--', color='red')
        plt.plot(range(len(symed_offline_result.reconstruction)), symed_offline_result.reconstruction, linestyle='-.', color='orange')

        if row == 2 or row == 1:
            plt.legend(['Original', 'SymED - pieces', 'SymED - symbols'], loc='upper right', prop={'size': 8})
        else:
            plt.legend(['Original', 'SymED - pieces', 'SymED - symbols'], loc='lower right', prop={'size': 8})
        if row == 0:
            labelpad = 11.8
        elif row == 1:
            labelpad = 1
        else:
            labelpad = 20
        plt.ylabel('value', labelpad=labelpad)

        if row == len(datasets) - 1:
            plt.xlabel('time point')

        figure_name = f'UCR_examples_SymED_{chr(row + 97)}'
        fig.savefig(f"{output_path}/{figure_name}.pdf", bbox_inches='tight', dpi=300)
        fig.savefig(f"{output_path}/{figure_name}.png", bbox_inches='tight', dpi=300)
        #plt.show()
        plt.clf()
        plt.close()
