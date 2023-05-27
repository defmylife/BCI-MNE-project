
"""Inference models"""

import numpy as np
import tensorflow as tf
# Build model
import keras
from keras.models import Model
from keras.layers import  Input, Dense,Dropout, Conv1D, MaxPooling1D, Flatten,BatchNormalization
from keras.optimizers import Adam
from keras import backend as K                                                          
from mne.decoding import Scaler
import scipy
import matplotlib.pyplot as plt


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
    def __init__(self, input_channels = 5, input_size = 1250, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Model 
        self._name = 'CNNModel'
        self.weights_path = '../weights/'

        # Input layer shape
        self.inputshape = (input_channels, input_size)

        # Parameters
        self.bath_size = 4
        self.train_epochs = 5
        self.upper_threshold = 0.8
        self.lower_threshold = 0.2

        # Define layers
        self.conv1_layer1 = Conv1D(filters=128, kernel_size=3, activation='relu')
        self.bath_norm_layer1 = BatchNormalization(name='batch_norm_layer1')
        if self.inputshape[0] > 3:
            self.maxpool_layer1 = MaxPooling1D(2, name='maxpool_layer1')
        else: 
            self.maxpool_layer1 = MaxPooling1D(1, name='maxpool_layer1')
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
        
        try: 
            model.load_weights(self.weights_path + self._name + '.h5')  
            print('Loaded pretrained'+ self._name + '.h5')
        except:
            print('Warning: no pretrained model')
        
        return model
    
    def model_train(self, X_train, Y_train, X_val=None, Y_val=None):

        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        Y_train = tf.convert_to_tensor(Y_train, dtype=tf.float32)
        
        # Callbacks API
        checkpoint_filepath = self.weights_path + f'weight-{self._name}.h5'
        self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss' if X_val is not None else 'loss',
            # monitor='val_accuracy' if X_val is not None else 'accuracy',
            mode='min',
            # mode='max',
            save_best_only=True)
        
        if X_val is None:
            self.model.fit(X_train, Y_train, 
                epochs=self.train_epochs, 
                batch_size=self.bath_size, 
                callbacks=[self.model_checkpoint_callback],
                verbose=1)
        else:
            F_val = tf.convert_to_tensor(F_val, dtype=tf.float32)
            Y_val = tf.convert_to_tensor(Y_val, dtype=tf.float32)
            self.model.fit(X_train, Y_train, 
                epochs=self.train_epochs, 
                batch_size=self.bath_size, 
                callbacks=[self.model_checkpoint_callback],
                validation_data=(F_val, Y_val),
                verbose=1)
        
        # The model weights (that are considered the best) are loaded into the
        # model.
        self.model.load_weights(checkpoint_filepath)
        self.model.save( self.weights_path + f'{self._name}.h5')

    def model_predict(self, X_test):
        return self.model.predict(X_test)
    
    def model_predict_classes(self, X_test):
        pred = self.model.predict(X_test)
        classout = np.argmax(pred, axis=1)
        return pred,classout
    
    def load_weights(self, path):
        self.model.load_weights(path)

class FFT_CNNModel(Model):
    """
    Fast Fourier tranform + CNNModel

    setup
    ------------------------------------------------------------
    FFT_CNN = FFT_CNNModel()
    FFT_CNN.model.summary()

    FFT_CNN.load_weights('FFT_CNNModel.h5')


    training
    ------------------------------------------------------------
    X shape should be : (None, 5, 1250)
    Y shape should be : (None, 2)
    
    FFT_CNN.model_train(X, Y)
    (or)
    FFT_CNN.model_train(X, Y, X_val, Y_val)


    eval/inference
    ------------------------------------------------------------
    X shape should be : (None, 5, 1250)
    
    predictions          = FFT_CNN.model_predict(X_test)
    (or)
    predictions, classes = FFT_CNN.model_predict_classes(X_test)


    """
    def __init__(self, input_channels = 5, input_size = 626, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Model 
        self._name = 'FFT_CNNModel'
        self.weights_path = '../weights/'

        # Input layer shape
        self.inputshape = (input_channels, input_size)

        # FFT Parameters
        self.acq_f = 250
        self.scaler = Scaler(scalings='mean')

        # Parameters
        self.bath_size = 4
        self.train_epochs = 10
        self.upper_threshold = 0.8
        self.lower_threshold = 0.2

        # Define layers
        self.conv1_layer1 = Conv1D(filters=128, kernel_size=3, activation='relu')
        self.bath_norm_layer1 =BatchNormalization(name='batch_norm_layer1')
        if self.inputshape[0] > 3:
            self.maxpool_layer1 = MaxPooling1D(2, name='maxpool_layer1')
        else: 
            self.maxpool_layer1 = MaxPooling1D(1, name='maxpool_layer1')
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
        model = Model(inputs=input_layer, outputs=output_layer, name=self._name)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        try: 
            model.load_weights(self.weights_path + self._name + '.h5')  
            print('Loaded pretrained FFT_CNNModel.h5')
        except:
            print('Warning: no pretrained model')

        return model
    
    def fft_transform(self, X):
        X = self.scaler.fit_transform(X)
        # Estimate power spectral density using Welch’s method
        f, F = scipy.signal.welch(X[:,:,:], self.acq_f, nperseg=X.shape[-1]) #, average='median'
        return f, F
    
    def fft_plot(self,X):
        f, F = self.fft_transform(X)
        # Extract each channel
        num_channels = self.inputshape[0]
        start_freq = 2
        end_freq = 15
        step_freq = 5 # 250/1250 = 0.2 Hz then inverse 1/0.2 = 5
        # plot
        fig, axs = plt.subplots(num_channels, 1, figsize=(10,10), gridspec_kw={'hspace': 0.5})
        for i in range(num_channels):
            axs[i].plot(f[start_freq*step_freq:end_freq*step_freq], F[:,i,start_freq*step_freq:end_freq*step_freq].T)
            axs[i].set_title(f'Channel {i}')
            axs[i].set_xlabel('Frequency [Hz]')
            axs[i].set_ylabel('Magnitude')
        plt.show()
    
    def load_weights(self, path):
        self.model.load_weights(path)
    
    def model_train(self, X_train, Y_train, X_val=None, Y_val=None):

        # FFT transform
        _ , F_train = self.fft_transform(X_train)

        F_train = tf.convert_to_tensor(F_train, dtype=tf.float32)
        Y_train = tf.convert_to_tensor(Y_train, dtype=tf.float32)
        
        # Callbacks API
        checkpoint_filepath = self.weights_path + f'weight-{self._name}.h5'
        self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss' if X_val is not None else 'loss',
            # monitor='val_accuracy' if X_val is not None else 'accuracy',
            mode='min',
            # mode='max',
            save_best_only=True)
        
        if X_val is None:
            self.model.fit(F_train, Y_train, 
                epochs=self.train_epochs, 
                batch_size=self.bath_size, 
                callbacks=[self.model_checkpoint_callback],
                verbose=1)
        else:
            X_val = self.scaler.fit_transform(X_val)
            _, F_val = scipy.signal.welch(X_val[:,:,:], 250, nperseg=X_val.shape[-1]) #, average='median'

            F_val = tf.convert_to_tensor(F_val, dtype=tf.float32)
            Y_val = tf.convert_to_tensor(Y_val, dtype=tf.float32)

            self.model.fit(F_train, Y_train, 
                epochs=self.train_epochs, 
                batch_size=self.bath_size, 
                callbacks=[self.model_checkpoint_callback],
                validation_data=(F_val, Y_val),
                verbose=1)
        
        # The model weights (that are considered the best) are loaded into the
        # model.
        self.model.load_weights(checkpoint_filepath)
        self.model.save( self.weights_path + f'{self._name}.h5')

    def model_predict(self, X_test):
        # FFT transform
        f, F_test = self.fft_transform(X_test)
        F_test = tf.convert_to_tensor(F_test, dtype=tf.float32)
        return self.model.predict(F_test)
    
    def model_predict_classes(self, X_test):
        # Predict
        pred = self.model_predict(X_test)
        classout = np.argmax(pred, axis=1)
        return pred, classout

        