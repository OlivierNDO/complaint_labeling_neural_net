### Configuration
########################################################################################################
# System Path Insert
import sys
sys.path.insert(1, 'D:/complaint_labeling_neural_net/src')

# Python Modules
import datetime
import nltk
import numpy as np
import pandas as pd
import pickle
import re
import sklearn
from tensorflow.keras.preprocessing.text import Tokenizer
import time

# Tensorflow / Keras Modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization, GlobalAvgPool2D
from tensorflow.keras.layers import Add, ZeroPadding2D, AveragePooling2D, GaussianNoise, SeparableConv2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.layers import add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Project Modules
import configuration as config



### Define Functions & Classes
########################################################################################################
def print_timestamp_message(message, timestamp_format = '%Y-%m-%d %H:%M:%S'):
    """
    Print formatted timestamp followed by custom message
    
    Args:
        message (str): string to concatenate with timestamp
        timestamp_format (str): format for datetime string. defaults to '%Y-%m-%d %H:%M:%S'
    """
    ts_string = datetime.datetime.fromtimestamp(time.time()).strftime(timestamp_format)
    print(f'{ts_string}: {message}')
    

def count_each_unique(lst):
    """
    Count occurences of each unique element of a list
    Args:
        lst (list): list or other iterable object
    Returns:
        pandas.DataFrame() object with fields 'element', 'count', and 'percent'
    """
    unique_values = list(set(lst))
    uv_counts = [len([l for l in lst if l == uv]) for uv in unique_values]
    uv_percent_counts = [uvc / sum(uv_counts) for uvc in uv_counts]
    output_df = pd.DataFrame({'element' : unique_values,
                              'count' : uv_counts,
                              'percent' : uv_percent_counts})
    return output_df



def remove_stopwords(input_string, word_delimiter = ' ', use_lowercase = True,
                     stopword_list = nltk.corpus.stopwords.words('english')):
   """
   Remove stopwords from a string
   Args:
       input_string (str): string
       word_delimiter (str): delimiter used in .split() to separate words in a string. defaults to ' '.
       use_lowercase (bool): boolean value indicating whether or not to convert characters to lowercase. defaults to True.
       stopword_list (list): list of stopwords to omit from string
   Returns:
       string
   """
   return word_delimiter.join([w for w in input_string.lower().split(word_delimiter) if w not in stopword_list])


def remove_punctuation(input_string, replace_punct_with = ' ', punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'):
   """
   Remove punctuation from a string
   Args:
       input_string (str): string
       replace_punct_with (str): string to replace punctuation with. defaults to ' '.
       punctuation (str): string of concatenated punctuation marks to remove
   Returns:
       str
   """
   punct_regex = re.compile('[%s]' % re.escape(punctuation))
   return punct_regex.sub(replace_punct_with, input_string)


def remove_numerics(input_string):
   """
   Remove numeric characters  from a string
   Args:
       input_string (str): string
   Returns:
       str
   """
   return ''.join([i for i in input_string if not i.isdigit()])


def remove_multiple_substrings(input_string, substring_removals, use_lowercase = True):
    """
    Remove list of substrings from a string
    Args:
        input_string (string): string
        substring_removals (list): list of substrings to remove
    Returns:
        string
    """
    your_new_string = input_string
    replace_dict = dict(zip(substring_removals, ['' for ssr in substring_removals]))
    for removal, blank in replace_dict.items():
        your_new_string = your_new_string.replace(removal, blank)
    return your_new_string


def stem_string_porter(input_string, word_delimiter = ' '):
   """
   Use NLTKs PorterStemmer() class to stem every word in a string
   Args:
       input_string (str): string
       word_delimiter (str): delimiter used in .split() to separate words in a string. defaults to ' '.
   Returns:
       string
   """
   stemmer_object = nltk.stem.PorterStemmer()
   return word_delimiter.join([stemmer_object.stem(w) for w in input_string.split(word_delimiter)])


def wordnet_lemmatize_string(input_string, word_delimiter = ' ', pos = 'n'):
   """
   Use NLTKs PorterStemmer() class to stem every word in a string
   Args:
       input_string (str): string
       word_delimiter (str): delimiter used in .split() to separate words in a string. defaults to ' '.
       pos (str): part of speech fed into wordnet lemmatization class. defaults to 'n' (noun)
   Returns:
       string
   """
   lemmatizer_object = nltk.stem.WordNetLemmatizer()
   return word_delimiter.join([lemmatizer_object.lemmatize(w, pos = pos) for w in input_string.split(word_delimiter)])




def clean_text(string_list, replace_punct_with = ' ', punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~', word_delimiter = ' ', use_lowercase = True, pos = 'n'):
    """
    Apply sequential text cleaning functions to a list of strings
    Args:
       string_list (list): list of strings string
       replace_punct_with (str): string to replace punctuation with. defaults to ' '.
       punctuation (str): string of concatenated punctuation marks to remove
       word_delimiter (str): delimiter used in .split() to separate words in a string. defaults to ' '.
       use_lowercase (bool): boolean value indicating whether or not to convert characters to lowercase. defaults to True.
       pos (str): part of speech fed into wordnet lemmatization class. defaults to 'n' (noun)
    Returns:
       list
    """
    sl_rm_punct = [remove_punctuation(s, replace_punct_with, punctuation) for s in string_list]
    sl_rm_nums = [remove_numerics(s) for s in sl_rm_punct]
    sl_rm_stopwords = [remove_stopwords(s, word_delimiter, use_lowercase) for s in sl_rm_nums]
    sl_lem_strings = [wordnet_lemmatize_string(s, word_delimiter, pos) for s in sl_rm_stopwords]
    sl_stem_strings = [stem_string_porter(s, word_delimiter) for s in sl_lem_strings]
    return sl_stem_strings



class TextProcessingPipeline:
    def __init__(self,
                 string_list,
                 test_string_list = None,
                 max_df = 0.7,
                 max_sequence_length = 300,
                 min_df = 10,
                 max_features = 1000,
                 ngram_range = (1,3),
                 pos = 'n',
                 punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~',
                 replace_punct_with = ' ',
                 stopword_list = nltk.corpus.stopwords.words('english'),
                 use_lowercase = True,
                 word_delimiter = ' ',
                 save_tfid_name = f'{config.config_vectorizer_folder}tfid_vectorizer.pkl',
                 save_token_name = f'{config.config_vectorizer_folder}keras_tokenizer.pkl',
                 train_x_save_name = config.config_train_x_save_name,
                 test_x_save_name = config.config_test_x_save_name,
                 train_y_save_name = config.config_train_y_save_name,
                 test_y_save_name = config.config_test_y_save_name,
                 train_x_save_name_issue = config.config_train_x_save_name_issue,
                 test_x_save_name_issue = config.config_test_x_save_name_issue,
                 train_y_save_name_issue = config.config_train_y_save_name_issue,
                 test_y_save_name_issue = config.config_test_y_save_name_issue
                 ):    
        self.string_list = string_list
        self.test_string_list = test_string_list
        self.max_df = max_df
        self.max_sequence_length = max_sequence_length
        self.min_df = min_df
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.pos = pos
        self.punctuation = punctuation
        self.replace_punct_with = replace_punct_with
        self.stopword_list = stopword_list
        self.use_lowercase = use_lowercase
        self.word_delimiter = word_delimiter
        self.save_tfid_name = save_tfid_name
        self.save_token_name = save_token_name
        self.train_x_save_name = train_x_save_name
        self.test_x_save_name = test_x_save_name
        self.train_y_save_name = train_y_save_name
        self.test_y_save_name = test_y_save_name
        self.train_x_save_name_issue = train_x_save_name_issue
        self.test_x_save_name_issue = test_x_save_name_issue
        self.train_y_save_name_issue = train_y_save_name_issue
        self.test_y_save_name_issue = test_y_save_name_issue
        
    def get_cleaned_train_text(self):
        print_timestamp_message('Removing punctuation (1/5)')
        sl_rm_punct = [remove_punctuation(s, self.replace_punct_with, self.punctuation) for s in self.string_list]
        print_timestamp_message('Removing numeric values (2/5)')
        sl_rm_nums = [remove_numerics(s) for s in sl_rm_punct]
        print_timestamp_message('Removing stopwords (3/5)')
        sl_rm_stopwords = [remove_stopwords(s, self.word_delimiter, self.use_lowercase, self.stopword_list) for s in sl_rm_nums]
        print_timestamp_message('Performing lemmatization (4/5)')
        sl_lem_strings = [wordnet_lemmatize_string(s, self.word_delimiter, self.pos) for s in sl_rm_stopwords]
        print_timestamp_message('Stemming words (5/5)')
        sl_stem_strings = [stem_string_porter(s, self.word_delimiter) for s in sl_lem_strings]
        return sl_stem_strings
    
    def get_cleaned_test_text(self):
        print_timestamp_message('Removing punctuation (1/5)')
        sl_rm_punct = [remove_punctuation(s, self.replace_punct_with, self.punctuation) for s in self.test_string_list]
        print_timestamp_message('Removing numeric values (2/5)')
        sl_rm_nums = [remove_numerics(s) for s in sl_rm_punct]
        print_timestamp_message('Removing stopwords (3/5)')
        sl_rm_stopwords = [remove_stopwords(s, self.word_delimiter, self.use_lowercase, self.stopword_list) for s in sl_rm_nums]
        print_timestamp_message('Performing lemmatization (4/5)')
        sl_lem_strings = [wordnet_lemmatize_string(s, self.word_delimiter, self.pos) for s in sl_rm_stopwords]
        print_timestamp_message('Stemming words (5/5)')
        sl_stem_strings = [stem_string_porter(s, self.word_delimiter) for s in sl_lem_strings]
        return sl_stem_strings
        
    def get_vectorized_text_and_feature_names(self):
        clean_text = self.get_cleaned_text()
        vectorizer_object = sklearn.feature_extraction.text.TfidfVectorizer(max_df = self.max_df,
                                                                            min_df = self.min_df,
                                                                            max_features = self.max_features,
                                                                            ngram_range = self.ngram_range)
        return vectorizer_object.fit_transform(clean_text), vectorizer_object.get_feature_names()
        
    def get_vectorized_text_and_feature_names_train_test(self):
        clean_text = self.get_cleaned_train_text()
        clean_text_test = self.get_cleaned_test_text()
        vectorizer_object = sklearn.feature_extraction.text.TfidfVectorizer(max_df = self.max_df,
                                                                            min_df = self.min_df,
                                                                            max_features = self.max_features,
                                                                            ngram_range = self.ngram_range)
        
        train_vec = vectorizer_object.fit_transform(clean_text)
        test_vec = vectorizer_object.fit_transform(clean_text_test)
        feat_names = vectorizer_object.get_feature_names()
        return train_vec, test_vec, feat_names
        
    def vectorizer_fit_and_save(self):
        clean_text = self.get_cleaned_train_text()
        vectorizer_object = sklearn.feature_extraction.text.TfidfVectorizer(max_df = self.max_df,
                                                                            min_df = self.min_df,
                                                                            max_features = self.max_features,
                                                                            ngram_range = self.ngram_range)
        
        train_vec = vectorizer_object.fit(clean_text)
        print(f'Saving tfid vectorizer in file: {self.save_tfid_name}')
        pickle.dump(train_vec, open(self.save_tfid_name, "wb"))
    
    def tokenizer_fit_and_save(self):
        clean_text = self.get_cleaned_train_text()
        tokenizer_object = Tokenizer(num_words = self.max_features)
        tokenizer_object.fit_on_texts(clean_text)
        print(f'Saving keras tokenizer in file: {self.save_token_name}')
        with open(self.save_token_name, 'wb') as handle:
            pickle.dump(tokenizer_object, handle, protocol = pickle.HIGHEST_PROTOCOL)
        
    def tokenizer_load_and_transform_train(self):
        print_timestamp_message('Processing training set text')
        clean_text = self.get_cleaned_train_text()
        print_timestamp_message('Loading saved tokenizer')
        tokenizer_object = pickle.load(open(self.save_token_name, 'rb'))
        print_timestamp_message('Converting text to sequences')
        train_sequences = tokenizer_object.texts_to_sequences(clean_text)
        train_sequences= pad_sequences(train_sequences, maxlen = self.max_sequence_length)
        print_timestamp_message('Returning training set sequences')
        return train_sequences
    
    def tokenizer_load_and_transform_test(self):
        print_timestamp_message('Processing test set text')
        clean_text_test = self.get_cleaned_test_text()
        print_timestamp_message('Loading saved tokenizer')
        tokenizer_object = pickle.load(open(self.save_token_name, 'rb'))
        print_timestamp_message('Converting text to sequences')
        test_sequences = tokenizer_object.texts_to_sequences(clean_text_test)
        test_sequences= pad_sequences(test_sequences, maxlen = self.max_sequence_length)
        print_timestamp_message('Returning test set sequences')
        return test_sequences
    
    def get_tokenizer_word_index(self):
        tokenizer_object = pickle.load(open(self.save_token_name, 'rb'))
        return tokenizer_object.word_index
    
    def load_transformed_train_test_product_data(self):
        train_x = np.load(self.train_x_save_name)
        test_x = np.load(self.test_x_save_name)
        train_y = np.load(self.train_y_save_name)
        test_y = np.load(self.test_y_save_name)
        return train_x, train_y, test_x, test_y
    
    def load_transformed_train_test_issue_data(self):
        train_x = np.load(self.train_x_save_name_issue)
        test_x = np.load(self.test_x_save_name_issue)
        train_y = np.load(self.train_y_save_name_issue)
        test_y = np.load(self.test_y_save_name_issue)
        return train_x, train_y, test_x, test_y
        
        
        
### To Do
### N-grams
### remove contractions
### part of speech tokenized features & counts
### n - LDA topic models outputs as features
### 
        
        
        
        
        
        
        
        
        
        




