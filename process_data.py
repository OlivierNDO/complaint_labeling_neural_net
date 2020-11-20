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
complaint_df['Product_Issue'] = complaint_df['Product'] + ' | ' + complaint_df['Issue']
complaint_df = complaint_df[complaint_df.Product_Issue.isin(config.config_product_issue_list)].\
drop_duplicates(subset = config.config_complaints_narrative_column, keep = 'first')

# Split into Train & Test
train_df, test_df = sklearn.model_selection.train_test_split(complaint_df, test_size = 0.2, random_state = 11172020)


### Process & Write Files: Product
########################################################################################################
# Y Variables
one_hot_dict_product = mf.one_hot_label_dict(complaint_df['Product'])
train_y_product = np.array([one_hot_dict_product.get(x) for x in train_df['Product']])
test_y_product = np.array([one_hot_dict_product.get(x) for x in test_df['Product']])

# X Variables
pipeline = tp.TextProcessingPipeline(string_list = train_df[config.config_complaints_narrative_column],
                                     test_string_list = test_df[config.config_complaints_narrative_column],
                                     save_token_name = config.config_tokenizer_save_name_product)

pipeline.tokenizer_fit_and_save()
train_sequences_product = pipeline.tokenizer_load_and_transform_train()
test_sequences_product = pipeline.tokenizer_load_and_transform_test()
np.save(config.config_train_x_save_name_product, train_sequences_product)
np.save(config.config_test_x_save_name_product, test_sequences_product)
np.save(config.config_train_y_save_name_product, train_y_product)
np.save(config.config_test_y_save_name_product, test_y_product)



### Process & Write Files: Checking
########################################################################################################
# Y Variables
train_df_checking = train_df[train_df.Product == 'Checking or savings account']
test_df_checking = test_df[test_df.Product == 'Checking or savings account']

one_hot_dict_checking = mf.one_hot_label_dict(train_df_checking['Issue'])
train_y_checking= np.array([one_hot_dict_checking.get(x) for x in train_df_checking['Issue']])
test_y_checking = np.array([one_hot_dict_checking.get(x) for x in test_df_checking['Issue']])

# X Variables
pipeline = tp.TextProcessingPipeline(string_list = train_df[config.config_complaints_narrative_column],
                                     test_string_list = test_df[config.config_complaints_narrative_column],
                                     save_token_name = config.config_tokenizer_save_name_checking)

pipeline.tokenizer_fit_and_save()
train_sequences_checking = pipeline.tokenizer_load_and_transform_train()
test_sequences_checking = pipeline.tokenizer_load_and_transform_test()
np.save(config.config_train_x_save_name_checking, train_sequences_checking)
np.save(config.config_test_x_save_name_checking, test_sequences_checking)
np.save(config.config_train_y_save_name_checking, train_y_checking)
np.save(config.config_test_y_save_name_checking, test_y_checking)


### Process & Write Files: Card
########################################################################################################
# Y Variables
train_df_card = train_df[train_df.Product == 'Credit card or prepaid card']
test_df_card = test_df[test_df.Product == 'Credit card or prepaid card']

one_hot_dict_card = mf.one_hot_label_dict(train_df_card['Issue'])
train_y_card= np.array([one_hot_dict_card.get(x) for x in train_df_card['Issue']])
test_y_card = np.array([one_hot_dict_card.get(x) for x in test_df_card['Issue']])

# X Variables
pipeline = tp.TextProcessingPipeline(string_list = train_df[config.config_complaints_narrative_column],
                                     test_string_list = test_df[config.config_complaints_narrative_column],
                                     save_token_name = config.config_tokenizer_save_name_card)

pipeline.tokenizer_fit_and_save()
train_sequences_card = pipeline.tokenizer_load_and_transform_train()
test_sequences_card = pipeline.tokenizer_load_and_transform_test()
np.save(config.config_train_x_save_name_card, train_sequences_card)
np.save(config.config_test_x_save_name_card, test_sequences_card)
np.save(config.config_train_y_save_name_card, train_y_card)
np.save(config.config_test_y_save_name_card, test_y_card)


### Process & Write Files: Credit Reporting
########################################################################################################
# Y Variables
train_df_cr = train_df[train_df.Product == 'Credit reporting, credit repair services, or other personal consumer reports']
test_df_cr = test_df[test_df.Product == 'Credit reporting, credit repair services, or other personal consumer reports']

one_hot_dict_cr = mf.one_hot_label_dict(train_df_cr['Issue'])
train_y_cr= np.array([one_hot_dict_cr.get(x) for x in train_df_cr['Issue']])
test_y_cr = np.array([one_hot_dict_cr.get(x) for x in test_df_cr['Issue']])

# X Variables
pipeline = tp.TextProcessingPipeline(string_list = train_df[config.config_complaints_narrative_column],
                                     test_string_list = test_df[config.config_complaints_narrative_column],
                                     save_token_name = config.config_tokenizer_save_name_cr)

pipeline.tokenizer_fit_and_save()
train_sequences_cr = pipeline.tokenizer_load_and_transform_train()
test_sequences_cr = pipeline.tokenizer_load_and_transform_test()
np.save(config.config_train_x_save_name_cr, train_sequences_cr)
np.save(config.config_test_x_save_name_cr, test_sequences_cr)
np.save(config.config_train_y_save_name_cr, train_y_cr)
np.save(config.config_test_y_save_name_cr, test_y_cr)


### Process & Write Files: Debt Collection
########################################################################################################
# Y Variables
train_df_dc = train_df[train_df.Product == 'Debt collection']
test_df_dc = test_df[test_df.Product == 'Debt collection']

one_hot_dict_dc = mf.one_hot_label_dict(train_df_dc['Issue'])
train_y_dc= np.array([one_hot_dict_dc.get(x) for x in train_df_dc['Issue']])
test_y_dc = np.array([one_hot_dict_dc.get(x) for x in test_df_dc['Issue']])

# X Variables
pipeline = tp.TextProcessingPipeline(string_list = train_df[config.config_complaints_narrative_column],
                                     test_string_list = test_df[config.config_complaints_narrative_column],
                                     save_token_name = config.config_tokenizer_save_name_dc)

pipeline.tokenizer_fit_and_save()
train_sequences_dc = pipeline.tokenizer_load_and_transform_train()
test_sequences_dc = pipeline.tokenizer_load_and_transform_test()
np.save(config.config_train_x_save_name_dc, train_sequences_dc)
np.save(config.config_test_x_save_name_dc, test_sequences_dc)
np.save(config.config_train_y_save_name_dc, train_y_dc)
np.save(config.config_test_y_save_name_dc, test_y_dc)


### Process & Write Files: Student Loan
########################################################################################################
# Y Variables
train_df_sl = train_df[train_df.Product == 'Student loan']
test_df_sl = test_df[test_df.Product == 'Student loan']

one_hot_dict_sl = mf.one_hot_label_dict(train_df_sl['Issue'])
train_y_sl= np.array([one_hot_dict_sl.get(x) for x in train_df_sl['Issue']])
test_y_sl = np.array([one_hot_dict_sl.get(x) for x in test_df_sl['Issue']])

# X Variables
pipeline = tp.TextProcessingPipeline(string_list = train_df[config.config_complaints_narrative_column],
                                     test_string_list = test_df[config.config_complaints_narrative_column],
                                     save_token_name = config.config_tokenizer_save_name_sl)

pipeline.tokenizer_fit_and_save()
train_sequences_sl = pipeline.tokenizer_load_and_transform_train()
test_sequences_sl = pipeline.tokenizer_load_and_transform_test()
np.save(config.config_train_x_save_name_sl, train_sequences_sl)
np.save(config.config_test_x_save_name_sl, test_sequences_sl)
np.save(config.config_train_y_save_name_sl, train_y_sl)
np.save(config.config_test_y_save_name_sl, test_y_sl)



