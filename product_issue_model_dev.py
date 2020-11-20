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



### Load Data
########################################################################################################
# Load Product & Text Data
pipeline = tp.TextProcessingPipeline(string_list = None, test_string_list = None)
train_x_product, train_y_product, test_x_product, test_y_product = pipeline.load_transformed_train_test_product_data()

# Load Issue & Text Data
train_x_issue, train_y_issue, test_x_issue, test_y_issue = pipeline.load_transformed_train_test_issue_data()

# Separate Validation Sets
train_x_product, valid_x_product, train_y_product, valid_y_product = sklearn.model_selection.train_test_split(train_x_product, train_y_product, test_size = 0.2, random_state = 11182020)
train_x_issue, valid_x_issue, train_y_issue, valid_y_issue = sklearn.model_selection.train_test_split(train_x_issue, train_y_issue, test_size = 0.2, random_state = 11182020)



### Model Fitting - Issues (Complaints)
###############################################################################
complaint_model = m.cudnn_lstm_classifier(input_dim = train_x_issue.shape[1],
                                          output_dim = 100,
                                          input_length = train_x_issue.shape[1],
                                          n_classes = train_y_issue.shape[1],
                                          word_index = pipeline.get_tokenizer_word_index())


complaint_classifier = m.RNNClassificationTrainer(model = complaint_model,
                                                  train_x = train_x_issue,
                                                  train_y = train_y_issue,
                                                  valid_x = valid_x_issue,
                                                  valid_y = valid_y_issue,
                                                  word_index = pipeline.get_tokenizer_word_index(),
                                                  model_save_name = config.config_issue_model_save_name)

complaint_classifier.fit()


final_complaint_model = complaint_classifier.load()
pred_test_complaint = final_complaint_model.predict(test_x_issue)


### Model Fitting - Product
###############################################################################
product_model = m.cudnn_lstm_classifier(input_dim = train_x_product.shape[1],
                                        output_dim = 100,
                                        input_length = train_x_product.shape[1],
                                        n_classes = train_y_product.shape[1],
                                        word_index = pipeline.get_tokenizer_word_index())


product_classifier = m.RNNClassificationTrainer(model = product_model,
                                                train_x = train_x_product,
                                                train_y = train_y_product,
                                                valid_x = valid_x_product,
                                                valid_y = valid_y_product,
                                                word_index = pipeline.get_tokenizer_word_index(),
                                                model_save_name = config.config_product_model_save_name)

product_classifier.fit()


