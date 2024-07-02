from aux import Aux
from ml_methods import Method, get_methods

import os
import pandas as pd
import numpy as np

# List of ML Methods we'll use
methods = get_methods()

# Open the 'results' csv correctly
csv_file_path = 'results.csv'
headers = ['Method', 'Dataset', 'F1-score', 'Precision', 'Recall', 'Accuracy', 'AUC', 'Normal', 'Anomaly']

if not os.path.exists(csv_file_path):
    df = pd.DataFrame(columns=headers)
    df.to_csv(csv_file_path, index=False)
else:
    df = pd.read_csv(csv_file_path)

# Get the list of methods already in the CSV
methods_in_csv = df['Method'].tolist()
methods_not_in_csv = [m for m in methods if m.name not in methods_in_csv]

# Run the metrics for every method not in the csv
for method in methods_not_in_csv:
    #try:
        print(f"Running {method.name}:")

        ds_names = ['og', 'pca', 'pearson', 'lda']
        datasets = Aux.get_datasets()
        rows_to_append = []

        for dataset, ds_name in zip(datasets, ds_names):
            print(f'\tRunning on {ds_name} dataset...')
            results = Aux.get_results_for_model(dataset, method.f, method.params)

            new_df_row = {
                'Method': method.name,
                'Dataset': ds_name,
                **results
            }
            rows_to_append.append(pd.DataFrame([new_df_row]))

        # Add new rows to df and write new csv
        if rows_to_append:
            df = pd.concat([df] + rows_to_append, ignore_index=True)
        df.to_csv(csv_file_path, index=False)

        print("Done!")
    #except:
    #    print(f'{method.name} failed.')
    #    continue

