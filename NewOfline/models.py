
"""Inference models"""

import numpy as np
import tensorflow as tf
# Build model
import keras
from keras.models import Model
from keras.layers import  Input, Dense,Dropout, Conv1D, MaxPooling1D, Flatten,BatchNormalization
from keras.optimizers import Adam
from keras import backend as K                                                          


def simpleModel(frequency_6Hz_Left, frequency_10Hz_Right, f_6Hz_threshold=3.0, f_10Hz_threshold=8.0) -> list:
    """
    This function classify the frequency of the 6Hz(Left) and 10Hz(Right) signals by giving threshold
    
    parameters
    ----------
    frequency_6Hz_Left <class 'list' or 'numpy.ndarray'>
    frequency_10Hz_Right <class 'list' or 'numpy.ndarray'>

    return
    ------
    outputs <class 'list'> ex. [5, 2, 2, 5, 2, ...]
    """
    if not isinstance(frequency_6Hz_Left, list):
        frequency_6Hz_Left, frequency_10Hz_Right = [frequency_6Hz_Left], [frequency_10Hz_Right]
    
    outputs = []

    for f6, f10 in zip(frequency_6Hz_Left, frequency_10Hz_Right):

        if f10 >= f_10Hz_threshold: outputs.append(5)
        elif f6 >= f_6Hz_threshold: outputs.append(2)
    
    return outputs

class CNNModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Input layer shape
        self.inputshape = (5, 1250)

        # Parameters
        self.bath_size = 4
        self.train_epochs = 5
        self.upper_threshold = 0.8
        self.lower_threshold = 0.2

        # Define layers
        self.conv1_layer1 = Conv1D(filters=128, kernel_size=3, activation='relu')
        self.bath_norm_layer1 =BatchNormalization(name='batch_norm_layer1')
        self.maxpool_layer1 = MaxPooling1D(2, name='maxpool_layer1')
        self.dropout_layer1 = Dropout(0.2, name='dropout_layer1')
        self.dense_layer1 = Dense(64, activation='relu', name='dense_layer1')
        self.fatten_layer1 = Flatten(name='fatten_layer1')
        self.dropout_layer2 = Dropout(0.2, name='dropout_layer2')
        self.dense_layer2 = Dense(32, activation='relu', name='dense_layer2')
        self.output_layer = Dense(3, activation='softmax', name='output_layer')

        # Get model
        self.model = self.get_model()

    def get_model(self):

        # Define input
        input_layer = Input(shape=self.inputshape, name='input_layer')

        # Model layers
        x = self.conv1_layer1(input_layer)
        x = self.bath_norm_layer1(x)
        x = self.maxpool_layer1(x)
        x = self.dropout_layer1(x)
        x = self.dense_layer1(x)
        x = self.fatten_layer1(x)
        x = self.dropout_layer2(x)
        x = self.dense_layer2(x)
        output_layer = self.output_layer(x)

        # Model
        model = Model(inputs=input_layer, outputs=output_layer, name='CNNModel')
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def model_train(self, X_train, Y_train):
        self.model.fit(X_train, Y_train, epochs=self.train_epochs, batch_size=self.bath_size, verbose=1)

    def model_predict(self, X_test):
        return self.model.predict(X_test)
    
    def model_predict_classes(self, X_test):
        pred = self.model.predict(X_test)
        pred = np.argmax(pred, axis=1)
        return pred
    
    def load_weights(self, path):
        self.model.load_weights(path)

