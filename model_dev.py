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

# Tensorflow / Keras Modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization, GlobalAvgPool2D
from tensorflow.keras.layers import Add, ZeroPadding2D, AveragePooling2D, GaussianNoise, SeparableConv2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

import configuration as config
import text_processing as tp


### File Reading
########################################################################################################
# Read Csv File from CFPB
complaint_df = pd.read_csv(config.config_complaints_file, encoding = config.config_complaints_file_encoding)
complaint_df = complaint_df[complaint_df[config.config_complaints_narrative_column].notna()].\
drop_duplicates(subset = config.config_complaints_narrative_column, keep = 'first')

# Create Product Category Field
complaint_df['Product_Category'] = [config.config_product_dict.get(x) for x in complaint_df['Product']]

# Split into Train & Test
train_df, test_df = sklearn.model_selection.train_test_split(complaint_df, test_size = 0.2, random_state = 11172020)


### Define Functions
########################################################################################################


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




### Execute Functions
########################################################################################################

# Create Y Array
one_hot_dict = one_hot_label_dict(complaint_df['Product_Category'])
train_y = np.array([one_hot_dict.get(x) for x in train_df['Product_Category']])
test_y = np.array([one_hot_dict.get(x) for x in test_df['Product_Category']])



# Create X Arrays
pipeline = tp.TextProcessingPipeline(string_list = train_df[config.config_complaints_narrative_column],
                                     test_string_list = test_df[config.config_complaints_narrative_column])


#pipeline.tokenizer_fit_and_save()
train_sequences = pipeline.tokenizer_load_and_transform_train()
test_sequences = pipeline.tokenizer_load_and_transform_test()




























