### Configuration
########################################################################################################
# System Path Insert
import sys
sys.path.insert(1, 'D:/complaint_labeling_neural_net/src')

# Python Module Imports
from collections import Counter
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
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
from tensorflow.keras.layers import Add, ZeroPadding2D, AveragePooling2D, GaussianNoise, SeparableConv2D, Embedding, Bidirectional, LSTM, GRU
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
train_sequences, train_y, test_sequences, test_y = pipeline.load_transformed_train_test_product_data()


# Separate Validation Set
train_sequences, valid_sequences, train_y, valid_y = sklearn.model_selection.train_test_split(train_sequences, train_y, test_size = 0.2, random_state = 11182020)


### Model Architecture
###############################################################################
def bidirectional_lstm_classifier(input_dim, output_dim, input_length, n_classes, word_index, dropout_rate = 0.2):
    
    x = Sequential()
    x.add(Embedding(input_dim = len(word_index) + 1, output_dim = output_dim, input_length = input_length, trainable = True))
    #x.add(SpatialDropout1D(dropout_rate))
    x.add(GRU(32))
    #x.add(Bidirectional(LSTM(output_dim, return_sequences = True)))
    #x.add(Bidirectional(LSTM(output_dim, return_sequences = False)))
    x.add(Dense(output_dim, activation = 'relu'))
    x.add(Dropout(dropout_rate))
    x.add(Dense(output_dim, activation = 'relu'))
    x.add(Dense(n_classes, activation = 'softmax'))
    return x


# Generator to Augment Images During Training
def batch_generator(x_arr, y_arr, batch_size = 20):
    """
    Create Keras generator objects for minibatch training that flips images vertically and horizontally
    Args:
        x_arr: array of predictors
        y_arr: array of targets
        batch_size: size of minibatches
    """
    indices = np.arange(len(x_arr)) 
    batch_list = []
    while True:
            np.random.shuffle(indices) 
            for i in indices:
                batch_list.append(i)
                if len(batch_list)==batch_size:
                    yield x_arr[batch_list], y_arr[batch_list]
                    batch_list=[]
                    
                    
class CyclicalRateSchedule:
    """
    Return a list of cyclical learning rates with the first <warmup_epochs> using minimum value
    
    Args:
        min_lr: minimum learning rate in cycle and learning rate during the first <warmup_epochs> epochs
        max_lr: maximum learning rate in cycle
        warmup_epochs: the number of initial epochs for which to run the minimum learning rate
        cooldown_epochs: the number of epochs after each cycle to run the minimum learning rate
        cycle_length: the number of epochs between min and max learning rates
        n_epochs: number of epochs for which to generate a learning rate
        logarithmic: if true, increase rates logarithmically (rather than linearly) during cycle
        decrease_factor: reduction factor for rates in consecutive cycles; 
                         e.g. 0.9 will reduce each cycle by 10% of the prior cycle
    """
    def __init__(self, min_lr = 0.00001, max_lr = 0.0006, warmup_epochs = 5, cooldown_epochs = 5,
                 cycle_length = 10, n_epochs = 100, logarithmic = False, decrease_factor = None):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_epochs = warmup_epochs
        self.cooldown_epochs = warmup_epochs
        self.cycle_length = cycle_length
        self.n_epochs = n_epochs
        self.logarithmic = logarithmic
        self.decrease_factor = decrease_factor
    
    def get_rates(self):
        warmup_rates = [self.min_lr] * self.warmup_epochs
        cycle_delta = (self.max_lr - self.min_lr) / self.cycle_length
        n_cycles = int(np.ceil(self.n_epochs / self.cycle_length))
        if self.logarithmic:
            single_cycle = np.array(list(np.logspace(np.log10(self.min_lr), np.log10(self.max_lr), num = self.cycle_length)))
            single_cycle = np.concatenate([single_cycle, np.array([self.min_lr] * self.cooldown_epochs)])
        else:
            single_cycle = np.array([self.min_lr + (i * cycle_delta) for i in range(self.cycle_length)])
            single_cycle = np.concatenate([single_cycle, np.array([self.min_lr] * self.cooldown_epochs)])
        if self.decrease_factor is not None:
            cycle_placeholder = []
            for i in range(n_cycles):
                if i == 0:
                    last_cycle = single_cycle
                    for c in last_cycle:
                        cycle_placeholder.append(c)
                else:
                    new_cycle = [c * self.decrease_factor for c in last_cycle]
                    for c in new_cycle:
                        cycle_placeholder.append(c)
                    last_cycle = new_cycle
            cycle_rates = [max(c, self.min_lr) for c in cycle_placeholder]
        else:
            cycle_rates = list(single_cycle) * n_cycles
        all_rates = warmup_rates + cycle_rates
        return all_rates[:self.n_epochs]
    
    def plot_cycle(self, first_n = None):
        rates = self.get_rates()
        if first_n is not None:
            show_n = rates[0:first_n]
        else:
            show_n = rates
        plt.plot(show_n)
        plt.show()
        
    def lr_scheduler(self):
        rates = self.get_rates()
        def schedule(epoch):
            return rates[epoch]
        return LearningRateScheduler(schedule, verbose = 1)




### Model Fitting
###############################################################################
                    
# Create Generators
use_batch_size = 20
train_gen = batch_generator(train_sequences, train_y, batch_size = use_batch_size)
valid_gen = batch_generator(valid_sequences, valid_y, batch_size = use_batch_size)
test_gen = batch_generator(test_sequences, test_y, batch_size = use_batch_size)

# Calculate Training Steps
tsteps = int(train_sequences.shape[0]) // use_batch_size
vsteps = int(valid_sequences.shape[0]) // use_batch_size

# Create Learning Rate Schedule
lr_schedule = CyclicalRateSchedule(min_lr = 0.000015,
                                   max_lr = 0.00025,
                                   n_epochs = 200,
                                   warmup_epochs = 5,
                                   cooldown_epochs = 1,
                                   cycle_length = 10,
                                   logarithmic = True,
                                   decrease_factor = 0.9)

lr_schedule.plot_cycle()


#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#config_tf = tf.compat.v1.ConfigProto()
#config_tf.gpu_options.allow_growth=True
#session = tf.compat.v1.Session(config=config_tf)

# Clear Session (this removes any trained model from your PC's memory)
train_start_time = time.time()
keras.backend.clear_session()

# Define Model Object and Scale Across GPUs
model = bidirectional_lstm_classifier(input_dim = train_sequences.shape[1],
                                      output_dim = 100,
                                      input_length = train_sequences.shape[1],
                                      n_classes = train_y.shape[1],
                                      word_index = pipeline.get_tokenizer_word_index())

# Compile Model
model.compile(loss='categorical_crossentropy', optimizer = Adam(0.0001), metrics = ['categorical_accuracy'])


# Keras Model Checkpoints (used for early stopping & logging epoch accuracy)
check_point = keras.callbacks.ModelCheckpoint(config.config_product_model_save_name, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min')
early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min',  patience = 15)

# Model Fit
model.fit(train_gen,
          epochs = 200,
          validation_data = valid_gen,
          steps_per_epoch = tsteps,
          validation_steps = vsteps,
          callbacks = [check_point, early_stop, lr_schedule.lr_scheduler()],
          class_weight = mf.make_class_weight_dict(train_y, return_dict = True))


train_end_time = time.time()
mf.sec_to_time_elapsed(train_end_time, train_start_time)

