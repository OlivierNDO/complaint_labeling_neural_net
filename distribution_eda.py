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



### Load Transformed Data
######################################################################################################## 
product_issue_pipeline = tp.TextProcessingPipeline(string_list = None, test_string_list = None,
                                             save_token_name = config.config_tokenizer_save_name_product_issue,
                                             train_x_save_name = config.config_train_x_save_name_product_issue,
                                             test_x_save_name = config.config_test_x_save_name_product_issue,
                                             train_y_save_name = config.config_train_y_save_name_product_issue,
                                             test_y_save_name = config.config_test_y_save_name_product_issue)

train_x, train_y, valid_x, valid_y, test_x, test_y = product_issue_pipeline.load_transformed_train_test_valid_data()



### Load Fitted Model
######################################################################################################## 

product_issue_model = keras.models.load_model(config.config_product_issue_model_save_name)
pred_test_y = product_issue_model.predict(test_x)


# Compare Distributions
actual_distribution = [x / len(test_y) for x in list(test_y.sum(axis = 0))]
pred_distribution = [x / len(test_y) for x in pred_test_y.sum(axis = 0)]

distribution_df = pd.DataFrame({'Product' : [s.split(' | ')[0] for s in sorted(config.config_product_issue_list)],
                                'Issue' : [s.split(' | ')[1] for s in sorted(config.config_product_issue_list)],
                                'Actual Distribution' : actual_distribution,
                                'Predicted Distribution' : pred_distribution})














[i for i, x in range()]



def actual_vs_predicted(test_y_array, pred_y_array,  topic_labels = sorted(config.config_product_issue_list)):
    
    output_list = []
    for i, x in enumerate(test_y_array):
        actual_index = list(x).index(1)
        actual_topic = topic_labels[actual_index]
        max_pred = max(pred_y_array[i])
        pred_topic = topic_labels[list(pred_y_array[i]).index(max_pred)]
        if actual_topic == pred_topic:
            correct = 1
        else:
            correct = 0
        output_dict = {'Actual Topic' : actual_topic,
                       'Predicted Topic' : pred_topic,
                       'Predicted Probability' : max_pred,
                       'Correct' : correct}
        output_list.append(output_dict)
        if i % 10000 == 0:
            print(f'Completed {i} of {len(test_y_array)}')
    return pd.DataFrame(output_list)
     
avp_df = actual_vs_predicted(test_y_array = test_y, pred_y_array = pred_test_y)

temp = avp_df[avp_df['Predicted Probability'] > 0.8]



np.mean(temp['Correct'])
    

plt.hist(avp_df['Predicted '])

actual_index = list(test_y[0]).index(1)
pred_test_y[0][actual_index]


pred_list = list(model_object.predict(vec_input_text)[0])
topic_labels = sorted(config.config_product_issue_list)
max_pred = max(pred_list)
max_pred_index = pred_list.index(max_pred)
max_pred_label = str(round(max_pred * 100, 2)) + '%'
pred_topic = topic_labels[max_pred_index]

















