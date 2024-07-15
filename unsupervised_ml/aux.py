import pandas as pd
import numpy as np

from sklearn.model_selection import ParameterGrid, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, adjusted_rand_score, normalized_mutual_info_score, confusion_matrix, precision_score, recall_score, f1_score

class Aux:
    @classmethod
    def __analyse_ml_results(cls, y, y_pred, average='macro'):
        ari = adjusted_rand_score(y, y_pred)
        nmi = normalized_mutual_info_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)

        accuracy = np.sum(np.amax(cm, axis=0)) / np.sum(cm)
        precision = precision_score(y, y_pred, average=average, zero_division=0)
        recall = recall_score(y, y_pred, average=average, zero_division=0)
        f1 = f1_score(y, y_pred, average=average, zero_division=0)

        return {
            'ari': ari,
            'nmi': nmi,
            'F1-Score': f1, 
            'Precision': precision,
            'Recall': recall,
            'Accuracy': accuracy
        }

    @classmethod
    def get_datasets(cls):
        og_df = pd.read_csv("../files/5G_attack_detection_ds.csv")
        pca_df = pd.read_csv("../files/pca_reduced_ds.csv")
        pearson_df = pd.read_csv("../files/pearson_reduced_ds.csv")
        lda_df = pd.read_csv("../files/lda_reduced_ds.csv")
        pearson_pca_df = pd.read_csv("../files/pearson-pca_reduced_ds.csv")
        pearson_lda_df = pd.read_csv("../files/pearson-lda_reduced_ds.csv")

        return [og_df, pca_df, pearson_df, lda_df, pearson_pca_df, pearson_lda_df]
    
    @classmethod
    def get_results_for_model(cls, ds, model, params: dict) -> dict:
        # Preparing data for training
        X = ds.iloc[:, :-1]

        # Normalizing data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        y = ds.iloc[:, -1]

        best_params = None
        best_score = -np.inf

        # Cross-validation with 5 groups
        n_splits = 3
        kf = KFold(n_splits=n_splits)
            
        for param_combination in ParameterGrid(params):
            print(f'\t\tRunning on parameters {param_combination}')
            cv_scores = []

            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                parametrized_model = model(**param_combination)
                
                # Fit the model on the training data and predict on the test data
                labels = parametrized_model.fit_predict(X_train)
                
                # Only calculate ARI if there are at least two clusters
                if len(set(labels)) > 1:
                    score = adjusted_rand_score(y_train, labels)
                else:
                    score = -1  # Assign a low score if only one cluster is formed
                
                cv_scores.append(score)
            
            # Calculate the average cross-validation score for the current parameter combination
            avg_score = np.mean(cv_scores)

            # Update the best score and parameters if the current score is better
            if avg_score > best_score:
                best_score = avg_score
                best_params = param_combination

        parametrized_model = model(**best_params)
        labels = parametrized_model.fit_predict(X)

        return cls.__analyse_ml_results(y, labels)


    