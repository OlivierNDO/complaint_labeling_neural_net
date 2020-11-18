### Configuration
########################################################################################################
# System Path Insert
import sys
sys.path.insert(1, 'D:/complaint_labeling_neural_net/src')

# Python Module Imports
from collections import Counter
import itertools
import numpy as np
import pandas as pd
import pickle
import sklearn
import time

# Tensorflow / Keras Modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization, SpatialDropout1D
from tensorflow.keras.layers import Add, ZeroPadding2D, AveragePooling2D, GaussianNoise, SeparableConv2D, Embedding, Bidirectional, LSTM
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.python.client import device_lib
from tensorflow.keras.utils import multi_gpu_model

# Project Modules
import configuration as config
import text_processing as tp
import misc_functions as mf



### Load Data
########################################################################################################
# Load Product & Text Data Using TextProcessingPipeline Class
pipeline = tp.TextProcessingPipeline(string_list = None, test_string_list = None)
train_sequences, train_y, test_sequences, test_y = pipeline.load_transformed_train_test_data()


### Model Architecture
###############################################################################
def bidirectional_lstm_classifier(input_dim, output_dim, input_length, n_classes, word_index, dropout_rate = 0.2):
    
    x = Sequential()
    x.add(Embedding(input_dim = len(word_index) + 1, output_dim = output_dim, input_length = input_length, trainable = True))
    x.add(SpatialDropout1D(dropout_rate))
    x.add(Bidirectional(LSTM(output_dim, return_sequences = True)))
    x.add(Bidirectional(LSTM(output_dim, return_sequences = False)))
    x.add(Dense(output_dim, activation = 'relu'))
    x.add(Dropout(dropout_rate))
    x.add(Dense(output_dim, activation = 'relu'))
    x.add(Dense(n_classes, activation = 'softmax'))
    return x


### Model Fitting
###############################################################################

#tf.compat.v2.enable_v2_behavior()

# Clear Session (this removes any trained model from your PC's memory)
train_start_time = time.time()
keras.backend.clear_session()

# Define Model Object and Scale Across GPUs
model = bidirectional_lstm_classifier(input_dim = train_sequences.shape[1],
                                      output_dim = 300,
                                      input_length = train_sequences.shape[1],
                                      n_classes = train_y.shape[1],
                                      word_index = pipeline.get_tokenizer_word_index())

# Compile Model
model.compile(loss='categorical_crossentropy', optimizer = Adam(0.0001), metrics = ['categorical_accuracy'])


# Keras Model Checkpoints (used for early stopping & logging epoch accuracy)
check_point = keras.callbacks.ModelCheckpoint(config.config_product_model_save_name, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min')
early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min',  patience = 15)

# Model Fit
model.fit(train_sequences,
          train_y,
          epochs = 5,
          validation_split = 0.2,
          batch_size = 5,
          callbacks = [check_point, early_stop],
          class_weight = mf.make_class_weight_dict(train_y, return_dict = True))

train_end_time = time.time()
mf.sec_to_time_elapsed(train_end_time, train_start_time)











# Define Model with Mirrored Strategy (allowing multiple gpu training)
model = bidirectional_lstm_classifier(input_dim = train_sequences.shape[1],
                                      output_dim = 300,
                                      input_length = train_sequences.shape[1],
                                      n_classes = train_y.shape[1],
                                      word_index = pipeline.get_tokenizer_word_index(),
                                      dropout_rate = 0.5,
                                      model_name = 'bidirectional_lstm_classifier')



# Keras Model Checkpoints (used for early stopping & logging epoch accuracy)
check_point = keras.callbacks.ModelCheckpoint(config.config_product_model_save_name, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min')
early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min',  patience = 15)


# Define Model Compilation
model.compile(loss='categorical_crossentropy',
              optimizer = Adam(0.0001),
              metrics = ['categorical_accuracy'])

model.fit(train_sequences,
          train_y,
          epochs = 5,
          validation_split = 0.2,
          batch_size = 10,
          callbacks = [check_point, early_stop],
          class_weight = make_class_weight_dict(train_y, return_dict = True))

train_end_time = time.time()
sec_to_time_elapsed(train_end_time, train_start_time)


