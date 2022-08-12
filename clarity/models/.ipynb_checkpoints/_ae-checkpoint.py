from math import ceil, sqrt
import tensorflow as tf
from tensorflow.keras import layers,optimizers

from .utils import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import warnings
warnings.filterwarnings("ignore")

class AutoEncoder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.save_path = kwargs['save_path']
        if kwargs['mode'] != 'inf':
            self.create_model(kwargs['features_name'], kwargs['cat_features'], kwargs['padding_max_len'], kwargs['cat_unique_values'],
                              kwargs['hidden_nLayers'], kwargs['hidden_dims'], kwargs['latent_dims'], kwargs['drop_out'],
                              kwargs['kernel_regularizer'], kwargs['activity_regularizer'], kwargs['leatning_rate'])

    def __get_emb_layer__(self, uValue, name, window, type_):
        if type_ == 'cat':
            n_unique = len(uValue[name])
            n_dim = int(sqrt(n_unique)) * 3
            input_layer = layers.Input(shape=(window, ), name = name)

            output_layer = layers.Embedding(input_dim=n_unique, 
                                output_dim=n_dim, name = name + '_emb')(input_layer)
            output_layer = layers.Conv1D(20, kernel_size=1, activation='relu', name = name+'_conv')(output_layer)
            # output_layer = layers.AveragePooling1D(strides=1, padding='same', name = name+'_avpool')(output_layer)
            output_layer = layers.LayerNormalization(epsilon=1e-6)(output_layer)
            
        else:
            input_layer = layers.Input(shape=(window, 1), name = name)

            output_layer = layers.Conv1D(20, kernel_size=1, activation='relu', name = name+'_conv')(input_layer)
            # output_layer = layers.AveragePooling1D(strides=1, padding='same', name = name+'_avpool')(output_layer)
            output_layer = layers.LayerNormalization(epsilon=1e-6)(output_layer)
            
        return input_layer, output_layer    

    
    def create_model(self, features_name, cat_features, padding_max_len,cat_unique_values, hidden_nLayers, hidden_dims, latent_dims,
                     drop_out=None, kernel_regularizer=None, activity_regularizer=None, leatning_rate=0.001):

        in_cat = {}
        out_cat = {}
        for col in features_name:
            if col in cat_features:
                in_cat[col], out_cat[col] = self.__get_emb_layer__(cat_unique_values, col, padding_max_len, 'cat')
            else:
                in_cat[col], out_cat[col] = self.__get_emb_layer__(None, col, padding_max_len, 'num')

        enc_input = [item for item in in_cat.values()]
        enc_output = [item for item in out_cat.values()]
        self.encoder = layers.Concatenate()(enc_output)
        self.encoder = layers.BatchNormalization()(self.encoder)

        for i in range(hidden_nLayers):
            self.encoder = layers.Conv1D(hidden_dims[i], kernel_size=1, activation='relu',
                                   kernel_regularizer=kernel_regularizer,
                                   activity_regularizer = activity_regularizer)(self.encoder)
            self.encoder = layers.AveragePooling1D(strides=1, padding='same')(self.encoder)
            self.encoder = layers.BatchNormalization()(self.encoder)
            self.encoder = layers.ActivityRegularization()(self.encoder)
            self.encoder = layers.Dropout(drop_out)(self.encoder)
        
        self.encoder = layers.Dense(latent_dims, activation='relu')(self.encoder)
        self.encoder = layers.BatchNormalization()(self.encoder)
        self.encoder = layers.Dropout(0.2)(self.encoder)
        self.decoder = layers.TimeDistributed(layers.Dense(len(features_name), activation='softmax'))(self.encoder)
        self.autoencoder = tf.keras.models.Model(enc_input, self.decoder)
        
        # optimizer = 'rmsprop'
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss = 'categorical_crossentropy'
        # loss = 'mse'
        self.autoencoder.compile(optimizer=optimizer, loss=loss)      
        
    
    def train(self, xTrain, xVal):
        trainDict = arr2dict(xTrain, self.kwargs['features_name'])
        testDict = arr2dict(xVal, self.kwargs['features_name'])
        
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.kwargs['log_dir'])
        earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
        mcp_save = tf.keras.callbacks.ModelCheckpoint(self.save_path + 'ae_best_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
        reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='min')

        self.history = self.autoencoder.fit(trainDict, xTrain,
                      batch_size=self.kwargs['batch_size'],
                      epochs=self.kwargs['epochs'],
                      shuffle=True,
                      validation_data=(testDict, xVal),
                      verbose=2,
                      callbacks=[tensorboard_callback, earlyStopping, mcp_save, reduce_lr_loss])

    
    def get_encoder(self):
        encoder = tf.keras.models.Model(inputs=self.autoencoder.inputs,
                                outputs=self.autoencoder.layers[-3].output)
        encoder.trainable = False
        return encoder
        
        
    
    def save_model(self):
        self.autoencoder.trainable = False
        self.autoencoder.save(self.save_path + 'ae_model.h5')
    
    def load_model(self):
        self.autoencoder = tf.keras.models.load_model(self.save_path + 'ae_model.h5')
        self.autoencoder.load_weights(self.save_path + 'ae_best_wts.hdf5')