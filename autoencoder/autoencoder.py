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

class AutoencoderAux:
    @classmethod
    def plot_reconstruction_error_distribution(cls, reconstruction_errors:list, labels:list, threshold=None):
        for error, label in zip(reconstruction_errors, labels):
            color = [random.random() for _ in range(3)]
            plt.hist(error, bins=50, color=color, label=label)

        if threshold:
            plt.axvline(threshold, color='r', linestyle='dashed', linewidth=2)

        plt.title('Reconstruction Error Histogram')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Frequency')

        plt.legend()
        plt.show()


class Autoencoder:
    def __init__(self, input_dimension):
        input_layer = Input(shape=(input_dimension,))

        # Encoder
        encoding_layer_1 = Dense(64, activation='sigmoid')(input_layer)
        encoding_layer_2 = Dense(32, activation='sigmoid')(encoding_layer_1)
        encoding_layer_3 = Dense(16, activation='sigmoid')(encoding_layer_2)
        encoding_layer_4 = Dense(8, activation='sigmoid')(encoding_layer_3)

        # Decoder
        decoding_layer_1 = Dense(8, activation='sigmoid')(encoding_layer_4)
        decoding_layer_2 = Dense(16, activation='sigmoid')(decoding_layer_1)
        decoding_layer_3 = Dense(32, activation='sigmoid')(decoding_layer_2)
        decoding_layer_4 = Dense(64, activation='sigmoid')(decoding_layer_3)
        output_layer = Dense(input_dimension, activation='sigmoid')(decoding_layer_4)

        # Create the autoencoder model
        self.autoencoder = Model(input_layer, output_layer)
        self.autoencoder.compile(optimizer='adam', loss='mean_squared_error')

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