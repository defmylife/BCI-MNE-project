"""Inference models"""

import numpy as np
import tensorflow as tf
# Build model
import keras
from keras.models import Model, Sequential
from keras.layers import  Input, Dense,Dropout, Conv1D, MaxPooling1D, Flatten,BatchNormalization
from keras.optimizers import Adam
from keras import backend as K                                                          
from mne.decoding import Scaler

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
    def __init__(self, inputshape=None, preprocess_scalar=True, preprocess_fft=True, preprocess_fft4to12=False, preprocess_reduceCh=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._name = ('n' if preprocess_scalar else '') + ('FFT_' if preprocess_fft else '') + ('4to12_' if preprocess_fft4to12 else '') + ('3ch_' if preprocess_reduceCh else '') + 'CNNModel'

        # Input layer shape ----------------------------------------------------
        if not inputshape:
            self.inputshape = (
                5 if not preprocess_reduceCh else 3, # 3: O1, O2, Oz
                1250 if not preprocess_fft else (626 if not preprocess_fft4to12 else 39))
        else:
            self.inputshape = inputshape

        # Setup proprocessing method -------------------------------------------
        self.preprocess_scalar = preprocess_scalar
        self.preprocess_fft = preprocess_fft
        self.preprocess_fft4to12 = preprocess_fft4to12
        self.preprocess_reduceCh = preprocess_reduceCh

        # ----------------------------------------------------------------------

        self.scaler = Scaler(scalings='mean')

        # Parameters
        self.bath_size = 4
        self.train_epochs = 10
        self.upper_threshold = 0.8
        self.lower_threshold = 0.2

        # Define layers
        self.conv1_layer1 = Conv1D(filters=128, kernel_size=3, activation='relu')
        self.bath_norm_layer1 =BatchNormalization(name='batch_norm_layer1')
        self.maxpool_layer1 = MaxPooling1D((2 if not preprocess_reduceCh else 1), name='maxpool_layer1')
        self.dropout_layer1 = Dropout(0.2, name='dropout_layer1')
        self.dense_layer1 = Dense(64, activation='relu', name='dense_layer1')
        self.fatten_layer1 = Flatten(name='fatten_layer1')
        self.dropout_layer2 = Dropout(0.2, name='dropout_layer2')
        self.dense_layer2 = Dense(32, activation='relu', name='dense_layer2')
        self.output_layer = Dense(2, activation='softmax', name='output_layer')

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
            model.load_weights(self._name+'.h5')
            print(f'Loaded pretrained {self._name}.h5')
        except:
            print('Warning: no pretrained model')

        return model
    
    def load_weights(self, path):
        self.model.load_weights(path)
    
    def model_train(self, X_train, Y_train, X_val=None, Y_val=None):
        if self.preprocess_reduceCh:
            X_train = X_train[:,:3,:]
            
        if self.preprocess_scalar:
            X_train = self.scaler.fit_transform(X_train)

        # Estimate power spectral density using Welch’s method
        if self.preprocess_fft:
            f, X_train = scipy.signal.welch(X_train[:,:,:], 250, nperseg=X_train.shape[-1]) #, average='median'
            # Select f-range 4.0-12.0Hz
            if self.preprocess_fft4to12:
                f, X_train = f[f<12.0],     X_train[:,:,f<12.0]
                f, X_train = f[4.0<f],      X_train[:,:,4.0<f]

        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        Y_train = tf.convert_to_tensor(Y_train, dtype=tf.float32)

        # Callbacks API
        checkpoint_filepath = f'weights/weight-{self._name}.h5'
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
            if self.preprocess_reduceCh:
                X_val = X_val[:,:3,:]
                
            if self.preprocess_scalar:
                X_val = self.scaler.fit_transform(X_val)
            
            if self.preprocess_fft:
                f, X_val = scipy.signal.welch(X_val[:,:,:], 250, nperseg=X_val.shape[-1]) #, average='median'
                # Select f-range 4.0-12.0Hz
                if self.preprocess_fft4to12:
                    f, X_val = f[f<12.0],   X_val[:,:,f<12.0]
                    f, X_val = f[4.0<f],    X_val[:,:,4.0<f]

            X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
            Y_val = tf.convert_to_tensor(Y_val, dtype=tf.float32)

            self.model.fit(X_train, Y_train, 
                epochs=self.train_epochs, 
                batch_size=self.bath_size, 
                callbacks=[self.model_checkpoint_callback],
                validation_data=(X_val, Y_val),
                verbose=1)
        
        # The model weights (that are considered the best) are loaded into the
        # model.
        self.model.load_weights(checkpoint_filepath)
        self.model.save(f'weights/{self._name}.h5')

    def model_predict(self, X_test):
        if self.preprocess_reduceCh:
            X_test = X_test[:,:3,:]
            
        if self.preprocess_scalar:
            X_test = self.scaler.fit_transform(X_test)

        # Estimate power spectral density using Welch’s method
        if self.preprocess_fft:
            f, X_test = scipy.signal.welch(X_test[:,:,:], 250, nperseg=X_test.shape[-1]) #, average='median'
            # Select f-range 4.0-12.0Hz
            if self.preprocess_fft4to12:
                f, X_test = f[f<12.0],      X_test[:,:,f<12.0]
                f, X_test = f[4.0<f],       X_test[:,:,4.0<f]
        X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)

        return self.model.predict(X_test)
    
    def model_predict_classes(self, X_test):
        if self.preprocess_reduceCh:
            X_test = X_test[:,:3,:]
            
        if self.preprocess_scalar:
            X_test = self.scaler.fit_transform(X_test)
        
        # Estimate power spectral density using Welch’s method
        if self.preprocess_fft:
            f, X_test = scipy.signal.welch(X_test[:,:,:], 250, nperseg=X_test.shape[-1]) #, average='median'
            # Select f-range 4.0-12.0Hz
            if self.preprocess_fft4to12:
                f, X_test = f[f<12.0],     X_test[:,:,f<12.0]
                f, X_test = f[4.0<f],      X_test[:,:,4.0<f]
        X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
        
        pred = self.model.predict(X_test)
        classout = np.argmax(pred, axis=1)
        return pred, classout

class DenseModel(Model):
    """
    1-Dense layer Model (Logistic regression)

    setup
    ------------------------------------------------------------
    # simple usage
    DENSE = DenseModel()

    # advance usage
    DENSE = DenseModel(         # Preprocessing options
        preprocess_scalar=True,     # normalize
        preprocess_fft=True,        # fast-fourier
        preprocess_fft4to12=False,  # fast-fourier + filter just 4-12Hz
        preprocess_reduceCh=False,  # channel reduction from 5 to 3 channels (O1,Oz,O2)
    )

    DENSE.model.summary()

    DENSE.load_weights('nFFT_CNNModel.h5')


    training
    ------------------------------------------------------------
    X shape should be : (None, 5, 1250)
    Y shape should be : (None, 2)
    
    DENSE.model_train(X, Y)
    (or)
    DENSE.model_train(X, Y, X_val, Y_val)


    eval/inference
    ------------------------------------------------------------
    X shape should be : (None, 5, 1250)
    
    predictions          = DENSE.model_predict(X_test)
    (or)
    predictions, classes = DENSE.model_predict_classes(X_test)


    """
    def __init__(self, inputshape=None, preprocess_scalar=True, preprocess_fft=True, preprocess_fft4to12=False, preprocess_reduceCh=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._name = ('n' if preprocess_scalar else '') + ('FFT_' if preprocess_fft else '') + ('4to12_' if preprocess_fft4to12 else '') + ('3ch_' if preprocess_reduceCh else '') + 'DenseModel'

        # Input layer shape ----------------------------------------------------
        if not inputshape:
            self.inputshape = (
                5 if not preprocess_reduceCh else 3, # 3: O1, O2, Oz
                1250 if not preprocess_fft else (626 if not preprocess_fft4to12 else 39))
        else:
            self.inputshape = inputshape

        # Setup proprocessing method -------------------------------------------
        self.preprocess_scalar = preprocess_scalar
        self.preprocess_fft = preprocess_fft
        self.preprocess_fft4to12 = preprocess_fft4to12
        self.preprocess_reduceCh = preprocess_reduceCh

        # ----------------------------------------------------------------------

        self.scaler = Scaler(scalings='mean')

        # Parameters
        self.bath_size = 4
        self.train_epochs = 10
        self.upper_threshold = 0.8
        self.lower_threshold = 0.2

        # self.conv1_layer1 = Conv1D(filters=128, kernel_size=3, activation='relu')
        # self.bath_norm_layer1 =BatchNormalization(name='batch_norm_layer1')
        # self.maxpool_layer1 = MaxPooling1D((2 if not preprocess_reduceCh else 1), name='maxpool_layer1')
        # self.dropout_layer1 = Dropout(0.2, name='dropout_layer1')
        # self.dense_layer1 = Dense(64, activation='relu', name='dense_layer1')
        # self.fatten_layer1 = Flatten(name='fatten_layer1')
        # self.dropout_layer2 = Dropout(0.2, name='dropout_layer2')
        # self.dense_layer2 = Dense(32, activation='relu', name='dense_layer2')
        # self.output_layer = Dense(2, activation='softmax', name='output_layer')

        # Define layers
        self.flatten = Flatten(input_shape=self.inputshape)
        self.dense_layer1 = Dense(units=2, activation='softmax')

        # Get model
        self.model = self.get_model()

    def get_model(self):

        # Define input
        # input_layer = Input(shape=self.inputshape, name='input_layer')

        # Model layers
        # x = self.conv1_layer1(input_layer)
        # x = self.bath_norm_layer1(x)
        # x = self.maxpool_layer1(x)
        # x = self.dropout_layer1(x)
        # x = self.dense_layer1(x)
        # x = self.fatten_layer1(x)
        # x = self.dropout_layer2(x)
        # x = self.dense_layer2(x)
        # output_layer = self.output_layer(x)

        # Define the model architecture
        model = Sequential()
        # model.add(Dense(units=64, activation='relu', input_shape=(input_dim,)))
        model.add(self.flatten)
        model.add(self.dense_layer1)

        # Model
        # model = Model(inputs=input_layer, outputs=output_layer, name=self._name)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        try: 
            model.load_weights(self._name+'.h5')
            print(f'Loaded pretrained {self._name}.h5')
        except:
            print('Warning: no pretrained model')

        return model
    
    def load_weights(self, path):
        self.model.load_weights(path)
    
    def model_train(self, X_train, Y_train, X_val=None, Y_val=None):
        if self.preprocess_reduceCh:
            X_train = X_train[:,:3,:]
            
        if self.preprocess_scalar:
            X_train = self.scaler.fit_transform(X_train)

        # Estimate power spectral density using Welch’s method
        if self.preprocess_fft:
            f, X_train = scipy.signal.welch(X_train[:,:,:], 250, nperseg=X_train.shape[-1]) #, average='median'
            # Select f-range 4.0-12.0Hz
            if self.preprocess_fft4to12:
                f, X_train = f[f<12.0],     X_train[:,:,f<12.0]
                f, X_train = f[4.0<f],      X_train[:,:,4.0<f]

        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        Y_train = tf.convert_to_tensor(Y_train, dtype=tf.float32)

        # Callbacks API
        checkpoint_filepath = f'weights/weight-{self._name}.h5'
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
            if self.preprocess_reduceCh:
                X_val = X_val[:,:3,:]
                
            if self.preprocess_scalar:
                X_val = self.scaler.fit_transform(X_val)
            
            if self.preprocess_fft:
                f, X_val = scipy.signal.welch(X_val[:,:,:], 250, nperseg=X_val.shape[-1]) #, average='median'
                # Select f-range 4.0-12.0Hz
                if self.preprocess_fft4to12:
                    f, X_val = f[f<12.0],   X_val[:,:,f<12.0]
                    f, X_val = f[4.0<f],    X_val[:,:,4.0<f]

            X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
            Y_val = tf.convert_to_tensor(Y_val, dtype=tf.float32)

            self.model.fit(X_train, Y_train, 
                epochs=self.train_epochs, 
                batch_size=self.bath_size, 
                callbacks=[self.model_checkpoint_callback],
                validation_data=(X_val, Y_val),
                verbose=1)
        
        # The model weights (that are considered the best) are loaded into the
        # model.
        self.model.load_weights(checkpoint_filepath)
        self.model.save(f'weights/{self._name}.h5')

    def model_predict(self, X_test):
        if self.preprocess_reduceCh:
            X_test = X_test[:,:3,:]
            
        if self.preprocess_scalar:
            X_test = self.scaler.fit_transform(X_test)

        # Estimate power spectral density using Welch’s method
        if self.preprocess_fft:
            f, X_test = scipy.signal.welch(X_test[:,:,:], 250, nperseg=X_test.shape[-1]) #, average='median'
            # Select f-range 4.0-12.0Hz
            if self.preprocess_fft4to12:
                f, X_test = f[f<12.0],      X_test[:,:,f<12.0]
                f, X_test = f[4.0<f],       X_test[:,:,4.0<f]
        X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)

        return self.model.predict(X_test)
    
    def model_predict_classes(self, X_test):
        if self.preprocess_reduceCh:
            X_test = X_test[:,:3,:]
            
        if self.preprocess_scalar:
            X_test = self.scaler.fit_transform(X_test)
        
        # Estimate power spectral density using Welch’s method
        if self.preprocess_fft:
            f, X_test = scipy.signal.welch(X_test[:,:,:], 250, nperseg=X_test.shape[-1]) #, average='median'
            # Select f-range 4.0-12.0Hz
            if self.preprocess_fft4to12:
                f, X_test = f[f<12.0],     X_test[:,:,f<12.0]
                f, X_test = f[4.0<f],      X_test[:,:,4.0<f]
        X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
        
        pred = self.model.predict(X_test)
        classout = np.argmax(pred, axis=1)
        return pred, classout
