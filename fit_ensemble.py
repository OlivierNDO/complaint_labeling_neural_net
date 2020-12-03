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
import xgboost as xgb


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



### Load Transformed Data
######################################################################################################## 
product_issue_pipeline = tp.TextProcessingPipeline(string_list = None, test_string_list = None,
                                             save_token_name = config.config_tokenizer_save_name_product_issue,
                                             train_x_save_name = config.config_train_x_save_name_product_issue,
                                             test_x_save_name = config.config_test_x_save_name_product_issue,
                                             train_y_save_name = config.config_train_y_save_name_product_issue,
                                             test_y_save_name = config.config_test_y_save_name_product_issue)

train_x, train_y, valid_x, valid_y, test_x, test_y = product_issue_pipeline.load_transformed_train_test_valid_data()



### Load Fitted Models
######################################################################################################## 
product_issue_model = keras.models.load_model(config.config_product_issue_model_save_name)
product_model = keras.models.load_model(config.config_product_model_save_name)
checking_model = keras.models.load_model(config.config_checking_model_save_name)
card_model = keras.models.load_model(config.config_card_model_save_name)
cr_model = keras.models.load_model(config.config_cr_model_save_name)
dc_model = keras.models.load_model(config.config_dc_model_save_name)
sl_model = keras.models.load_model(config.config_sl_model_save_name)



### Y-Variables by Model
######################################################################################################## 
prod_issue_df = pd.DataFrame({'Product' : [s.split(' | ')[0] for s in sorted(config.config_product_issue_list)],
                              'Issue' : [s.split(' | ')[1] for s in sorted(config.config_product_issue_list)]})


product_labels = sorted(list(set(prod_issue_df['Product'])))
checking_labels = sorted(list(set(prod_issue_df[prod_issue_df['Product'] == 'Checking or savings account']['Issue'])))
card_labels = sorted(list(set(prod_issue_df[prod_issue_df['Product'] == 'Credit card or prepaid card']['Issue'])))
cr_labels = sorted(list(set(prod_issue_df[prod_issue_df['Product'] == 'Credit reporting, credit repair services, or other personal consumer reports']['Issue'])))
dc_labels = sorted(list(set(prod_issue_df[prod_issue_df['Product'] == 'Debt collection']['Issue'])))
sl_labels = sorted(list(set(prod_issue_df[prod_issue_df['Product'] == 'Student loan']['Issue'])))



### Predict on Train, Test, Validation Sets with Each Model
########################################################################################################

train_x_pred = pd.DataFrame(np.concatenate((product_issue_model.predict(train_x),
                                            product_model.predict(train_x),
                                            checking_model.predict(train_x),
                                            card_model.predict(train_x),
                                            cr_model.predict(train_x),
                                            dc_model.predict(train_x),
                                            sl_model.predict(train_x)), axis = 1))

test_x_pred = pd.DataFrame(np.concatenate((product_issue_model.predict(test_x),
                                           product_model.predict(test_x),
                                           checking_model.predict(test_x),
                                           card_model.predict(test_x),
                                           cr_model.predict(test_x),
                                           dc_model.predict(test_x),
                                           sl_model.predict(test_x)), axis = 1))

valid_x_pred = pd.DataFrame(np.concatenate((product_issue_model.predict(valid_x),
                                            product_model.predict(valid_x),
                                            checking_model.predict(valid_x),
                                            card_model.predict(valid_x),
                                            cr_model.predict(valid_x),
                                            dc_model.predict(valid_x),
                                            sl_model.predict(valid_x)), axis = 1))


### Create Y Variables for Train, Test, Validation Sets
########################################################################################################
train_y_pred = pd.DataFrame(np.argmax(train_y, axis = 1), columns = ['Product_Issue'])
test_y_pred = pd.DataFrame(np.argmax(test_y, axis = 1), columns = ['Product_Issue'])
valid_y_pred = pd.DataFrame(np.argmax(valid_y, axis = 1), columns = ['Product_Issue'])


### Fit XGBoost Model
########################################################################################################
# Class Weights Based on Training Set
class_weights = mf.make_class_weight_dict(list(train_y_pred['Product_Issue']),  return_dict = True)

# Parameters and Number of Classes
n_class = test_y.shape[1]
param_dict = {'objective' : 'multi:softprob',
              'eta' : 0.015,
              'min_child_weight' : 8,
              'subsample' : 0.7,
              'colsample_bytree' : 0.7,
              'max_depth' : 6,
              'early_stopping_rounds' : 20,
              'stopping_metric' : 'mlogloss',
              'num_class' : n_class}

# Early Stopping Watchlist
dat_train = xgb.DMatrix(train_x_pred, label = train_y_pred, weight = [class_weights.get(x) for x in train_y_pred['Product_Issue']])
dat_valid = xgb.DMatrix(valid_x_pred, label = valid_y_pred, weight = [class_weights.get(x) for x in valid_y_pred['Product_Issue']])
watchlist = [(dat_train, 'train'), (dat_valid, 'valid')]

# Training Call
xgb_trn = xgb.train(params = param_dict,
                    dtrain = dat_train,
                    num_boost_round = 5000,
                    evals = watchlist,
                    early_stopping_rounds = 12,
                    maximize = False,
                    verbose_eval = True)










































product_issue_pred = product_issue_model.predict(test_x)
product_pred = product_model.predict(test_x)
checking_pred = checking_model.predict(test_x)
card_pred = card_model.predict(test_x)
cr_pred = cr_model.predict(test_x)
dc_pred = dc_model.predict(test_x)
sl_pred = sl_model.predict(test_x)



temp_x = pd.DataFrame(np.concatenate((product_issue_pred, product_pred, checking_pred, card_pred, cr_pred, dc_pred, sl_pred), axis = 1))
temp_y = pd.DataFrame(np.argmax(test_y, axis = 1), columns = ['Product_Issue'])
n_class = test_y.shape[1]

#temp = np.argmax(test_y, axis = 1)


param_dict = {'objective' : 'multi:softprob',
              'eta' : 0.03,
              'min_child_weight' : 8,
              'subsample' : 0.7,
              'colsample_bytree' : 0.6,
              'max_depth' : 6,
              'early_stopping_rounds' : 15,
              'stopping_metric' : 'mlogloss',
              'num_class' : n_class}




dat_train = xgb.DMatrix(temp_x, label = temp_y)
dat_valid = xgb.DMatrix(temp_x, label = temp_y)
watchlist = [(dat_train, 'train'), (dat_valid, 'valid')]

xgb_trn = xgb.train(params = param_dict,
                    dtrain = dat_train,
                    num_boost_round = 50,
                    evals = watchlist,
                    early_stopping_rounds = 12,
                    maximize = False,
                    verbose_eval = True)




























