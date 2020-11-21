### Configuration
########################################################################################################
# System Path Insert
import sys
sys.path.insert(1, 'D:/complaint_labeling_neural_net/src')

# Python Modules
import datetime
import functools
import nltk
import numpy as np
import pandas as pd
import pickle
import re
import sklearn
from tensorflow.keras.preprocessing.text import Tokenizer
import time
import tqdm

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
import misc_functions as mf


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


def replace_contractions(input_string, contraction_dict = config.config_contraction_dict, delimiter = ' '):
    """
    Replace contractions with full words using a dictionary
    Args:
        input_string (str): string within which to replace contractions
        contraction_dict (dictionary): dictionary with contractions and spelled out alternatives
    Returns:
        string
    """
    lower_string = input_string.lower()
    for w, i in contraction_dict.items():
        lower_string = lower_string.replace(w.lower(), i)
    return lower_string


def remove_stopwords(input_string, word_delimiter = ' ', use_lowercase = True,
                     stopword_list = nltk.corpus.stopwords.words('english') + config.config_custom_stopwords):
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


def get_pos_wordnet(word_string):
    """
    Get part of speech tag from nltk wordnet lemmatizer
    Args:
        word_string (str): string representing single word
    Returns:
        string
    """
    pos_tag = nltk.pos_tag([word_string])[0][1][0].upper()
    pos_tag_dict = {'J' : nltk.corpus.wordnet.ADJ,
                    'N' : nltk.corpus.wordnet.NOUN,
                    'V' : nltk.corpus.wordnet.VERB,
                    'R' : nltk.corpus.wordnet.ADV}
    
    return pos_tag_dict.get(pos_tag, nltk.corpus.wordnet.NOUN)


def wordnet_lemmatize_string(input_string, word_delimiter = ' ', maxsize = 50000):
   """
   Use NLTKs PorterStemmer() class to stem every word in a string
   Args:
       input_string (str): string
       word_delimiter (str): delimiter used in .split() to separate words in a string. defaults to ' '
       maxsize (int): maximum cache size for lemmatizer. defaults to 50000.
   Returns:
       string
   """
   lemmatizer = nltk.stem.WordNetLemmatizer()
   lemmatize = functools.lru_cache(maxsize = maxsize)(lemmatizer.lemmatize)
   return word_delimiter.join([lemmatize(w, pos = get_pos_wordnet(w)) for w in nltk.word_tokenize(input_string)])


def remove_non_alpha(input_string, replace_with = ' '):
    """
    Remove non-alphabetic characters from a string
    Args:
        input_string (str): string from which to remove non-alphabetic characters
        replace_with (str): string to replace non-alphabetic characters. defaults to ' '.
    Returns:
        string
    """
    return re.compile('[^a-zA-Z]').sub(replace_with, input_string)


def get_word_counts(input_string_list, delimiter = ' '):
    """
    Get counts of each unique word in a list of strings
    Args:
        input_string_list (list): list of strings
        delimiter (str): delimiter used in .split() to separate words in a string. defaults to ' '.
    """
    lower_string_list = mf.unnest_list_of_lists([s.lower().split(delimiter) for s in tqdm.tqdm(input_string_list)])
    return mf.get_unique_counts(lower_string_list)


def clean_text(string_list, contraction_dict = config.config_contraction_dict,
               replace_punct_with = ' ', punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~',
               word_delimiter = ' ', use_lowercase = True):
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
    output = []
    n_strings = len(string_list)
    print_timestamp_message(f'Cleaning {n_strings} strings')
    for i, s in enumerate(string_list):
        sx = replace_contractions(s.lower().strip(), contraction_dict, word_delimiter)
        sx = remove_non_alpha(sx)
        sx = remove_stopwords(sx, word_delimiter, use_lowercase)
        sx = wordnet_lemmatize_string(sx, word_delimiter)
        output.append(sx)
        if (i % 1000 == 0 and i != 0):
            print_timestamp_message(f'Completed {i} of {n_strings}')
    return output
    
    
class TextProcessingPipeline:
    def __init__(self,
                 string_list,
                 test_string_list = None,
                 save_token_name = config.config_tokenizer_save_name_product,
                 train_x_save_name = config.config_train_x_save_name_product,
                 test_x_save_name = config.config_test_x_save_name_product,
                 train_y_save_name = config.config_train_y_save_name_product,
                 test_y_save_name = config.config_test_y_save_name_product,
                 validation_size = 0.2,
                 random_state = 11202020,
                 max_df = 0.7,
                 max_sequence_length = 300,
                 min_df = 10,
                 max_features = 1000,
                 ngram_range = (1,3),
                 punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~',
                 replace_punct_with = ' ',
                 stopword_list = nltk.corpus.stopwords.words('english'),
                 use_lowercase = True,
                 word_delimiter = ' ',
                 contraction_dict = config.config_contraction_dict,
                 ):    
        self.string_list = string_list
        self.test_string_list = test_string_list
        self.save_token_name = save_token_name
        self.train_x_save_name = train_x_save_name
        self.test_x_save_name = test_x_save_name
        self.train_y_save_name = train_y_save_name
        self.test_y_save_name = test_y_save_name
        self.validation_size = validation_size
        self.random_state = random_state
        self.max_df = max_df
        self.max_sequence_length = max_sequence_length
        self.min_df = min_df
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.punctuation = punctuation
        self.replace_punct_with = replace_punct_with
        self.stopword_list = stopword_list
        self.use_lowercase = use_lowercase
        self.word_delimiter = word_delimiter
        self.contraction_dict = contraction_dict
        
        
    def get_cleaned_train_text(self):
        cl_txt =  clean_text(self.string_list,
                             contraction_dict = self.contraction_dict,
                             replace_punct_with = self.replace_punct_with,
                             punctuation = self.punctuation,
                             word_delimiter = self.word_delimiter,
                             use_lowercase = self.use_lowercase)
        return cl_txt
    
    def get_cleaned_test_text(self):
        cl_txt =  clean_text(self.test_string_list,
                             contraction_dict = self.contraction_dict,
                             replace_punct_with = self.replace_punct_with,
                             punctuation = self.punctuation,
                             word_delimiter = self.word_delimiter,
                             use_lowercase = self.use_lowercase)
        return cl_txt
    
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
    
    def load_transformed_train_test_data(self):
        train_x = np.load(self.train_x_save_name)
        test_x = np.load(self.test_x_save_name)
        train_y = np.load(self.train_y_save_name)
        test_y = np.load(self.test_y_save_name)
        return train_x, train_y, test_x, test_y
    
    def load_transformed_train_test_valid_data(self):
        train_x, train_y, test_x, test_y = self.load_transformed_train_test_data()
        train_x, valid_x, train_y, valid_y = sklearn.model_selection.train_test_split(train_x, train_y, test_size = self.validation_size, random_state = self.random_state)
        return train_x, train_y, valid_x, valid_y, test_x, test_y
        
        
        
        
### To Do
### N-grams
### remove contractions
### part of speech tokenized features & counts
### n - LDA topic models outputs as features
### 

        
        




