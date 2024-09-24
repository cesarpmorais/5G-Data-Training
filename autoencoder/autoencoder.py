import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session
from tensorflow.keras.optimizers import Adam

class AutoencoderAux:
    @classmethod
    def separate_datasets(cls, df):
        # Get non-anomalous labels and features
        non_anomalous_data = df[df.iloc[:, -1] == 1]
        non_anomalous_labels = non_anomalous_data.iloc[:, -1]
        non_anomalous_features = non_anomalous_data.iloc[:, :-1]

        # Get anomalous labels and features
        anomalous_data = df[df.iloc[:, -1] == -1]
        anomalous_labels = anomalous_data.iloc[:, -1]
        anomalous_features = anomalous_data.iloc[:, :-1]

        # Scale data
        scaler = StandardScaler()
        non_anomalous_features_scaled = scaler.fit_transform(non_anomalous_features)
        anomalous_features_scaled = scaler.transform(anomalous_features)

        return non_anomalous_features_scaled, non_anomalous_labels, anomalous_features_scaled, anomalous_labels

    @classmethod
    def add_instances_to_testing(cls, X_train, y_train, X_test, y_test, percentage=0.2):
        # Getting 20% of non-anomalies into training set
        num_instances_to_move = int(percentage * X_train.shape[0])
        random_indices = np.random.choice(X_train.shape[0], num_instances_to_move, replace=False)

        ## Select 20% of the instances and their corresponding labels
        X_train_subset = X_train[random_indices]
        y_train_subset = y_train.iloc[random_indices]

        ## Add the selected instances to the test set
        X2_test = np.concatenate([X_test, X_train_subset], axis=0)
        y2_test = np.concatenate([y_test, y_train_subset], axis=0)

        ## Remove the selected instances from the training set
        X2_train = np.delete(X_train, random_indices, axis=0)
        y2_train = y_train.drop(y_train.index[random_indices])

        return X2_train, y2_train, X2_test, y2_test 
    
    @classmethod
    def compare_results(cls, anomalies_found, anomaly_label):
        anomaly_labels = anomaly_label == -1

        # Compare the arrays
        correctly_identified = (anomalies_found == anomaly_labels)

        # Calculate the number of correctly identified anomalies
        correct_count = correctly_identified.sum()

        # Calculate the total number of anomalies and normal instances
        total_anomalies = anomaly_labels.sum()
        total_normal = (~anomaly_labels).sum()

        num_anomalies_detected = anomalies_found.sum()
        print(f"Number of anomalies detected: {num_anomalies_detected}")

        # Calculate accuracy, precision, recall, etc.
        accuracy = correct_count / len(anomalies_found)

        true_positives = (anomalies_found & anomaly_labels).sum()
        false_positives = (anomalies_found & ~anomaly_labels).sum()
        true_negatives = (~anomalies_found & ~anomaly_labels).sum()
        false_negatives = (~anomalies_found & anomaly_labels).sum()

        print(f"True Positives: {true_positives}")
        print(f"False Positives: {false_positives}")
        print(f"True Negatives: {true_negatives}")
        print(f"False Negatives: {false_negatives}")

        if anomalies_found.sum() > 0:
            precision = true_positives / anomalies_found.sum()
        else:
            precision = 0.0  # To handle division by zero

        if total_anomalies > 0:
            recall = true_positives / total_anomalies
        else:
            recall = 0.0  # To handle division by zero

        print()

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")


class Autoencoder:
    def __init__(self, input_dimension, normal_training=True,activation='relu'):
        input_layer = Input(shape=(input_dimension,))
        
        if(normal_training):
            # Encoder
            encoding_layer_1 = Dense(512, activation=activation)(input_layer)
            encoding_layer_2 = Dense(256, activation=activation)(encoding_layer_1)
            encoding_layer_3 = Dense(128, activation=activation)(encoding_layer_2)
            encoding_layer_4 = Dense(64, activation=activation)(encoding_layer_3)
            encoding_layer_5 = Dense(32, activation=activation)(encoding_layer_4)
            # Decoder
            decoding_layer_0 = Dense(32, activation=activation)(encoding_layer_5)
            decoding_layer_1 = Dense(64, activation=activation)(decoding_layer_0)
            decoding_layer_2 = Dense(128, activation=activation)(decoding_layer_1)
            decoding_layer_3 = Dense(256, activation=activation)(decoding_layer_2)
            decoding_layer_4 = Dense(512, activation=activation)(decoding_layer_3)
            output_layer = Dense(input_dimension)(decoding_layer_4)

            # Create the autoencoder model
            self.autoencoder = Model(input_layer, output_layer)

            custom_optimizer = Adam(learning_rate=0.001)
            self.autoencoder.compile(optimizer=custom_optimizer, loss='mean_absolute_error')

            self.initial_weights = self.autoencoder.get_weights()
        else:
            # Encoder
            encoding_layer_1 = Dense(512, activation=activation)(input_layer)
            encoding_layer_2 = Dense(256, activation=activation)(encoding_layer_1)
            encoding_layer_3 = Dense(128, activation=activation)(encoding_layer_2)
            encoding_layer_4 = Dense(64, activation=activation)(encoding_layer_3)
            encoding_layer_5 = Dense(32, activation=activation)(encoding_layer_4)
            # Decoder
            decoding_layer_0 = Dense(32, activation=activation)(encoding_layer_5)
            decoding_layer_1 = Dense(64, activation=activation)(decoding_layer_0)
            decoding_layer_2 = Dense(128, activation=activation)(decoding_layer_1)
            decoding_layer_3 = Dense(256, activation=activation)(decoding_layer_2)
            decoding_layer_4 = Dense(512, activation=activation)(decoding_layer_3)
            output_layer = Dense(input_dimension)(decoding_layer_4)

            # Create the autoencoder model
            self.autoencoder = Model(input_layer, output_layer)

            custom_optimizer = Adam(learning_rate=0.001)
            self.autoencoder.compile(optimizer=custom_optimizer, loss='mean_squared_error')

            self.initial_weights = self.autoencoder.get_weights()


    def train_model(self, X_train, epochs=200, batch_size=128):
        # Reset weights
        self.autoencoder.set_weights(self.initial_weights)
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping])

    def predict_and_get_reconstruction_errors(self, X_test):
        reconstructions = self.autoencoder.predict(X_test)
        reconstruction_errors = np.mean(np.square(X_test - reconstructions), axis=1)

        return reconstruction_errors