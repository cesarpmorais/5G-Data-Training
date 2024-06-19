import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score

def get_datasets():
    og_df = pd.read_csv("../files/5G_attack_detection_ds.csv")
    pca_df = pd.read_csv("../files/pca_reduced_ds.csv")
    pearson_df = pd.read_csv("../files/pearson_reduced_ds.csv")

    return og_df, pca_df, pearson_df


def prepare_dataset(df):
    X = df.iloc[:, :-1]

    # Normalizar dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


# Para analisar, usaremos o classification_report e também a Area Under the Curve (https://www.geeksforgeeks.org/auc-roc-curve/)
def analyse_ml_results(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    f1_score = class_report['weighted avg']['f1-score']
    auc = np.round(roc_auc_score(y_test, y_pred), 3)

    print(f'Verdadeiros Ataques: {cm[0][0]}, Falsos Não-Ataques: {cm[0][1]}')
    print(f'Falsos Ataques: {cm[1][0]}, Verdadeiros Não-Ataques: {cm[1][1]}')
    print(f'AUC: {auc}')

    print(classification_report(y_test, y_pred))

    return auc, f1_score


def plot_auc_and_f1(auc_list, f1_score_list):
    model_names = ['Original DS', 'Pearson', 'PCA']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

    # Plot AUC
    ax1.bar(range(len(auc_list)), auc_list)
    ax1.set_ylabel('AUC Values')
    ax1.set_title('AUC')

    # Adding x-axis labels with specific model names
    ax1.set_xticks(range(len(auc_list)))  # Setting the tick positions
    ax1.set_xticklabels(model_names) 

    # Plot F1-Score
    ax2.bar(range(len(f1_score_list)), f1_score_list)
    ax2.set_ylabel('F1-Score Values')
    ax2.set_title('F1-Score')

    # Adding x-axis labels with specific model names
    ax2.set_xticks(range(len(f1_score_list)))  # Setting the tick positions
    ax2.set_xticklabels(model_names) 

    # Display the plot
    plt.show()