import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split


class Aux:
    @classmethod
    def __prepare_dataset(cls, df):
        X = df.iloc[:, :-1]

        # Normalize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        y = df.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

        return X_train, X_test, y_train, y_test

    @classmethod
    def __analyse_ml_results(cls, y_test, y_pred_proba):
        # Convert probabilities to class labels for confusion matrix and classification report
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        cm = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)

        f1_score = class_report['weighted avg']['f1-score']
        precision = class_report['weighted avg']['precision']
        recall = class_report['weighted avg']['recall']
        accuracy = np.trace(cm) / np.sum(cm)  # Calculate accuracy correctly

        # Calculate AUC for multiclass
        auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovo', average='weighted')

        # Calculate normal and anomaly based on the confusion matrix
        normal = cm[0, 0] + cm[1, 0]
        anomaly = cm[1, 1] + cm[0, 1]
        
        results = {
            'F1-score': f1_score,
            'Precision': precision,
            'Recall': recall,
            'Accuracy': accuracy,
            'AUC': auc,
            'Normal': normal,
            'Anomaly': anomaly
        }
        
        return results

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
    def get_multiclass_datasets(cls):
        og_df = pd.read_csv("../files_multiclass/mixed_mult_class.csv")
        pca_df = pd.read_csv("../files_multiclass/pca_reduced_ds.csv")
        pearson_df = pd.read_csv("../files_multiclass/pearson_reduced_ds.csv")
        lda_df = pd.read_csv("../files_multiclass/lda_reduced_ds.csv")
        pearson_pca_df = pd.read_csv("../files_multiclass/pearson-pca_reduced_ds.csv")
        pearson_lda_df = pd.read_csv("../files_multiclass/pearson-lda_reduced_ds.csv")

        return [og_df, pca_df, pearson_df, lda_df, pearson_pca_df, pearson_lda_df]
    
    @classmethod
    def get_results_for_model(cls, ds: pd.DataFrame, model, params: dict) -> dict:
        # Get best parameters through Random Search
        X_train, X_test, y_train, y_test = cls.__prepare_dataset(ds)

        rand_search = RandomizedSearchCV(model(), params, random_state=42, n_jobs=-1)
        rand_search.fit(X_train, y_train)

        best_params = rand_search.best_params_

        # Get results
        best_model = model(**best_params)
        best_model.fit(X_train, y_train)
        y_pred_proba = best_model.predict_proba(X_test)

        return cls.__analyse_ml_results(y_test, y_pred_proba)
