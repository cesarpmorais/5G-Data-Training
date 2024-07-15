from aux import Aux
from ml_methods import Method, get_methods

import os
import pandas as pd
import numpy as np

# List of ML Methods we'll use
methods = get_methods()

# Open the 'results' csv correctly
csv_file_path = 'results.csv'
headers = ['Method', 'Dataset', 'ari', 'nmi', 'F1-score', 'Precision', 'Recall', 'Accuracy']

if not os.path.exists(csv_file_path):
    df = pd.DataFrame(columns=headers)
    df.to_csv(csv_file_path, index=False)
else:
    df = pd.read_csv(csv_file_path)

# Get the list of methods already in the CSV
methods_in_csv = df['Method'].tolist()
methods_not_in_csv = [m for m in methods if m.name not in methods_in_csv]

# Get datasets
ds_names = ['og', 'pca', 'pearson', 'lda', 'pearson-pca', 'pearson-lda']
datasets = Aux.get_datasets()

# Run the metrics: if method not in csv, run everything. If so, check if all datasets have been tested
for method in methods:
    if method.name not in methods_in_csv:
        print(f"Running {method.name}:")
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


    else:
        print(f"{method.name}'s been calculated. Checking for unused datasets...")

        for dataset, ds_name in zip(datasets, ds_names):
            if not ((df['Method'] == method.name) & (df['Dataset'] == ds_name)).any():
                try:
                    print(f'\tRunning on {ds_name} dataset...')
                    results = Aux.get_results_for_model(dataset, method.f, method.params)

                    new_df_row = {
                        'Method': method.name,
                        'Dataset': ds_name,
                        **results
                    }

                    # Find the index of the last occurrence of the method name
                    method_indices = df.index[df['Method'] == method.name].tolist()
                    if method_indices:
                        last_method_index = method_indices[-1]
                    else:
                        last_method_index = len(df) - 1  # If method is not found, append at the end

                    # Insert the new row underneath the last occurrence of method.name
                    df = pd.concat([df.iloc[:last_method_index + 1],
                                    pd.DataFrame([new_df_row]),
                                    df.iloc[last_method_index + 1:]]).reset_index(drop=True)
                    df.to_csv(csv_file_path, index=False)
                except:
                    print(f"\t{ds_name} failed!")
                    break

    print("Done!")
        
