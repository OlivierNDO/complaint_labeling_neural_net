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


### File Reading
########################################################################################################
# Read Csv File from CFPB
complaint_df = pd.read_csv(config.config_complaints_file, encoding = config.config_complaints_file_encoding)
complaint_df = complaint_df[complaint_df[config.config_complaints_narrative_column].notna()]
complaint_df = complaint_df[complaint_df['Issue'].notna()].\
drop_duplicates(subset = config.config_complaints_narrative_column, keep = 'first')

# Create Product Category Field
complaint_df['Product_Category'] = [config.config_product_dict.get(x) for x in complaint_df['Product']]
complaint_df['Product_Issue_Subissue'] = complaint_df['Product_Category'] + ' | ' + complaint_df['Issue'] + ' | ' + complaint_df['Sub-issue']
complaint_df['Complaint_Category'] = [config.config_issue_dict.get(x) for x in complaint_df['Product_Issue_Subissue']]
complaint_df = complaint_df[complaint_df['Complaint_Category'].notnull()]

# Split into Train & Test
train_df, test_df = sklearn.model_selection.train_test_split(complaint_df, test_size = 0.2, random_state = 11172020)


### Execute Functions
########################################################################################################
# Create Y Array
one_hot_dict = mf.one_hot_label_dict(complaint_df['Complaint_Category'])
train_y = np.array([one_hot_dict.get(x) for x in train_df['Complaint_Category']])
test_y = np.array([one_hot_dict.get(x) for x in test_df['Complaint_Category']])



# Transform Data Using TextProcessingPipeline Class
pipeline = tp.TextProcessingPipeline(string_list = train_df[config.config_complaints_narrative_column],
                                     test_string_list = test_df[config.config_complaints_narrative_column])

#pipeline.tokenizer_fit_and_save()
train_sequences = pipeline.tokenizer_load_and_transform_train()
test_sequences = pipeline.tokenizer_load_and_transform_test()
np.save(config.config_train_x_save_name_issue, train_sequences)
np.save(config.config_test_x_save_name_issue, test_sequences)
np.save(config.config_train_y_save_name_issue, train_y)
np.save(config.config_test_y_save_name_issue, test_y)












