from aux import Aux

import os
import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

class Method:
    def __init__(self, name, function, params):
        self.name = name
        self.f = function
        self.params = params

# List of ML Methods we'll use
methods = [
    Method('k-neighbors', KNeighborsClassifier, 
        { 
            'n_neighbors': np.arange(1,10),
            'weights': ['uniform', 'distance'],
            'leaf_size': [20, 30, 50, 100],
        }
    ),

    Method('supervised_neural_network', MLPClassifier, 
        {
            'hidden_layer_sizes': [(50,), (100,), (150,), (100, 50), (100, 100, 50)],
            'learning_rate_init': [0.0001, 0.001, 0.01, 0.1],
            'alpha': [0.0001, 0.001, 0.01, 0.1]
        }
    ),
    
    #Method('random_forest', RandomForestClassifier, {
    #    'n_estimators': [100, 200, 300, 400, 500],
    #    'max_depth': [None, 10, 20, 30, 40, 50],
    #    'max_features': ['sqrt', 'log2', None]
    #}),
    #Method('gradient_boosting', HistGradientBoostingClassifier, {
    #    'learning_rate': [0.01, 0.1, 0.2],
    #    'max_iter': [100, 300],
    #    'max_leaf_nodes': [10, 31, 50]
    #})
]

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

