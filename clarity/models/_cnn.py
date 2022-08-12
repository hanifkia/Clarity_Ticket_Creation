import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers,optimizers

import sys
sys.path.append('../../../../ml-sequence-detector')
from clarity.models import utils

import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class CnnClassifier():
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        if kwargs['mode'] != 'inf':
            self.create_model(pre_train = kwargs['pre_train'],
                              input_shape=kwargs['input_shape'],
                              nClasses=kwargs['nClasses'],
                              hidden_nLayers=kwargs['hidden_nLayers'],
                              filter_dims=kwargs['filter_dims'],
                              kernel_size=kwargs['kernel_size'],
                              activation=kwargs['activation'],
                              conv_padding=kwargs['conv_padding'],
                              maxPooling=kwargs['maxPooling'],
                              pool_size=kwargs['pool_size'],
                              pooling_strides=kwargs['pooling_strides'],
                              pooling_padding =kwargs['pooling_padding'],
                              drop_out=kwargs['drop_out']
                             )
        
    
    def create_model(self, pre_train=None, input_shape=None, nClasses=1,
                     hidden_nLayers=2, filter_dims=[100,100], kernel_size=2, activation='relu',
                     conv_padding='valid', maxPooling=True, pool_size=1, pooling_strides=1,
                     pooling_padding = 'valid', drop_out=0.0 
                    ):
        
        if pre_train == None:
            model_input = layers.Input(shape=input_shape)
        else:
            model_input = pre_train.output

        for i in range(hidden_nLayers):
            self.model = model_input if i== 0 else self.model
            self.model = layers.Conv1D(filter_dims[i],
                                kernel_size,
                                activation = activation,
                                padding = conv_padding
                               )(self.model)
            self.model = layers.MaxPool1D(pool_size = pool_size,
                                   strides = pooling_strides,
                                   padding = pooling_padding
                                  )(self.model) if maxPooling == True else self.model
            self.model = layers.BatchNormalization()(self.model)
            self.model = layers.ActivityRegularization()(self.model)
            self.model = layers.Dropout(drop_out)(self.model)

        self.model = layers.Flatten()(self.model)
        self.model = layers.BatchNormalization()(self.model)
        self.model = layers.ActivityRegularization()(self.model)
        self.model = layers.Dropout(drop_out)(self.model)
        self.model = layers.Dense(nClasses, activation='softmax')(self.model)
        in_ = pre_train.input if pre_train != None else self.model_input
        self.model = tf.keras.models.Model(inputs = in_, outputs=self.model)
        
        optimizer=tf.keras.optimizers.Adam(learning_rate=self.kwargs['leatning_rate'])
        loss = 'categorical_crossentropy'
        
        self.model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    
    def train(self, x_train, y_train,
              x_val=None, y_val=None,
              validation_split = 0.0,
              validation_data = None
             ):
        if self.kwargs['pre_train'] != None:
            x_train = getattr(utils, self.kwargs['inStyle_func'])(x_train, self.kwargs['features_name'])
            x_val = getattr(utils, self.kwargs['inStyle_func'])(x_val, self.kwargs['features_name'])

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.kwargs['log_dir'])
        earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
        mcp_save = tf.keras.callbacks.ModelCheckpoint(self.kwargs['save_path'] + 'cnnClf_best_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
        reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='min')
        
        if x_val != None:
            validation_data = (x_val, pd.get_dummies(y_val))
            
        
        self.history = self.model.fit(x_train, pd.get_dummies(y_train),
                                      batch_size=self.kwargs['batch_size'],
                                      epochs = self.kwargs['epochs'],
                                      shuffle = True,
                                      validation_data = validation_data,
                                      validation_split = validation_split,
                                      verbose=2,
                                      callbacks=[tensorboard_callback, earlyStopping, mcp_save, reduce_lr_loss]
                                     )
        
    def predict(self, x):
        return self.model.predict(x)
        
    
    def save_model(self):
        self.model.trainable = False
        self.model.save(self.kwargs['save_path'] + 'cnnClf_model.h5')
    
    def load_model(self):
        self.model = tf.keras.models.load_model(self.kwargs['save_path'] + 'cnnClf_model.h5')
        self.model.load_weights(self.kwargs['save_path'] + 'cnnClf_best_wts.hdf5')