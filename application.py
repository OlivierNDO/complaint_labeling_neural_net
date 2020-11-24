### Configuration
########################################################################################################
# System Path Insert
import sys
sys.path.insert(1, 'D:/complaint_labeling_neural_net/src')

# Python Module Imports
from collections import Counter
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import sklearn
import time
from werkzeug.utils import secure_filename

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
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Project Modules
import configuration as config
import text_processing as tp
import misc_functions as mf
import modeling as m


# File paths & allowed extensions
app_dir = 'D:/complaint_labeling_neural_net/'
UPLOAD_FOLDER = app_dir + 'uploads/'
model_file_path = config.config_product_model_save_name
tfid_file_path = config.config_tokenizer_save_name_product

model = keras.models.load_model(model_file_path)
tfid_vectorizer = pickle.load(open(tfid_file_path, 'rb'))

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

### Flask App Functions
########################################################################################################
def predict(input_text, model_object = model, vectorizer = tfid_vectorizer, confidence_threshold = 0.3, max_sequence_length = 300):
    text_pipeline = tp.TextProcessingPipeline(string_list = [input_text])
    cleaned_input_text = text_pipeline.get_cleaned_train_text()
    vec_input_text = vectorizer.texts_to_sequences(cleaned_input_text)
    vec_input_text = pad_sequences(vec_input_text, maxlen = max_sequence_length)
    pred_list = list(model_object.predict(vec_input_text)[0])
    topic_labels = ['Checking or savings account',
                    'Credit card or prepaid card',
                    'Credit reporting, credit repair services, or other personal consumer reports',
                    'Debt collection',
                    'Student loan']
    max_pred = max(pred_list)
    max_pred_index = pred_list.index(max_pred)
    max_pred_label = str(round(max_pred * 100, 2)) + '%'
    
    pred_topic = topic_labels[max_pred_index]
    if max_pred < confidence_threshold:
        output_val = "I'm not confident about this one ... "
    else:
        if max_pred > 0.8:
            conf_str = 'high'
        elif max_pred > confidence_threshold:
            conf_str = 'medium'
        else:
            conf_str = 'low'
        output_val = 'Product associated with complaint: <br>' + pred_topic + ' (' + conf_str + f' confidence -- {max_pred_label})'
    return output_val


app = Flask(__name__, static_folder = app_dir + 'static/')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def template_test():
    return render_template('home.html', label='', imagesource='file://null')


@app.route('/', methods=['GET', 'POST'])
def process_text_input():
    if request.method == 'POST':
        text = request.form['text']
        output = predict(input_text = text)
        #print('<br>Input Text:<br>'  + text)
    return render_template("home.html", label = output + '<br>', print_text = '<br>' + text)

if __name__ == "__main__":
    app.run(debug=False, threaded=False)

#if __name__ == "__main__":
#    app.run(host = '0.0.0.0', port = 8080, debug=False, threaded=False)








