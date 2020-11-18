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



### Define Functions
########################################################################################################
def seconds_to_time(sec):
    """
    Convert seconds (integer or float) to time in 'hh:mm:ss' format
    Args:
        sec (int | float): number of seconds
    Returns:
        string
    """
    if (sec // 3600) == 0:
        HH = '00'
    elif (sec // 3600) < 10:
        HH = '0' + str(int(sec // 3600))
    else:
        HH = str(int(sec // 3600))
    min_raw = (np.float64(sec) - (np.float64(sec // 3600) * 3600)) // 60
    if min_raw < 10:
        MM = '0' + str(int(min_raw))
    else:
        MM = str(int(min_raw))
    sec_raw = (sec - (np.float64(sec // 60) * 60))
    if sec_raw < 10:
        SS = '0' + str(int(sec_raw))
    else:
        SS = str(int(sec_raw))
    return HH + ':' + MM + ':' + SS + ' (hh:mm:ss)'

def sec_to_time_elapsed(end_tm, start_tm, return_time = False):
    """
    Apply seconds_to_time function to start and end times
    
    Args:
        end_tm (float): unix timestamp representing end time
        start_tm (float): unix timestamp representing start time
    """
    sec_elapsed = (np.float64(end_tm) - np.float64(start_tm))
    if return_time:
        return seconds_to_time(sec_elapsed)
    else:
        print('Execution Time: ' + seconds_to_time(sec_elapsed))

def one_hot_label_dict(labels):
    """
    One-hot encode labels, returning dictionary of labels and one-hot encoded lists
    Args:
        labels (list): list of strings
    Returns:
        dictionary
    """
    unique_labels = sorted(np.unique(labels))
    one_hot_nested_list = []
    for r in range(len(unique_labels)):
        zeroes = [0] * len(unique_labels)
        zeroes[r] = 1
        one_hot_nested_list.append(zeroes)
        
    one_hot_dict = dict(zip(unique_labels, one_hot_nested_list))
    return one_hot_dict


def get_unique_counts(input_list):
    """
    Get counts of unique elements in a list
    Args:
        input_list (list): list from which to count unique elements
    Returns:
        dictionary
    """
    return dict(zip(Counter(input_list).keys(), Counter(input_list).values()))


def unnest_list_of_lists(LOL):
    """
    Unnest a list of lists
    Args:
        LOL (list): list of other lists
    Returns:
        list
    """
    return list(itertools.chain.from_iterable(LOL))

def make_class_weight_dict(train_y_labels, return_dict = False):
    """
    Return dictionary of inverse class weights for imbalanced response
    
    Args:
        train_y_labels: training set response variable (list or numpy array)
        return_dict: if True, return dictionary of classes & weights..else return list of classes and list of weights
    """
    
    if str(type(train_y_labels)) == "<class 'numpy.ndarray'>":
        labs = list(range(train_y_labels.shape[1]))
        freq = list(np.sum(train_y_labels, axis = 0))
        train_class_counts = dict(zip(labs, freq))
    else:
        train_class_counts = dict((x,train_y_labels.count(x)) for x in set(train_y_labels))
    max_class = max(train_class_counts.values())
    class_weights = [max_class / x for x in train_class_counts.values()]
    class_weight_dict = dict(zip([i for i in train_class_counts.keys()], class_weights))
    if return_dict:
        return class_weight_dict
    else:
        return list(class_weight_dict.keys()), list(class_weight_dict.values())
 

def get_number_gpu():
    """Return number of available GPUs on local system"""
    n_gpu = len([x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU'])
    return n_gpu
