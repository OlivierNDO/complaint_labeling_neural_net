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
from tensorflow.keras.layers import Add, ZeroPadding2D, AveragePooling2D, GaussianNoise, SeparableConv2D, Embedding, Bidirectional, LSTM, GRU, Attention
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
import modeling as m



### Card Complaint Categorization Model
########################################################################################################
# Load Transformed Data
card_pipeline = tp.TextProcessingPipeline(string_list = None, test_string_list = None,
                                          save_token_name = config.config_tokenizer_save_name_card,
                                          train_x_save_name = config.config_train_x_save_name_card,
                                          test_x_save_name = config.config_test_x_save_name_card,
                                          train_y_save_name = config.config_train_y_save_name_card,
                                          test_y_save_name = config.config_test_y_save_name_card)

train_x, train_y, valid_x, valid_y, test_x, test_y = card_pipeline.load_transformed_train_test_valid_data()

# Fit Model
card_model = m.cudnn_lstm_classifier(input_dim = train_x.shape[1],
                                         output_dim = 200,
                                         input_length = train_x.shape[1],
                                         n_classes = train_y.shape[1],
                                         word_index = card_pipeline.get_tokenizer_word_index())

card_classifier = m.RNNClassificationTrainer(model = card_model,
                                             train_x = train_x,
                                             train_y = train_y,
                                             valid_x = valid_x,
                                             valid_y = valid_y,
                                             word_index = card_pipeline.get_tokenizer_word_index(),
                                             model_save_name = config.config_card_model_save_name)

card_classifier.fit()

