#!/usr/bin/env python3

#
# Description
#

# Libraries ------------------------------------------------------------------------------------------------------------

## General
import logging

## Text data
import string
import nltk
from nltk.tokenize import word_tokenize
from .helpers import remove_sw, stemmer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# DataPreparer --------------------------------------------------------------------------------------------------------
class DataPreparer:

    def __init__(self, config):
        self.random_state = config['random_state']
        self.lang = config['lang']
        self.text_var = config['text']
        self.text_process = config['text-processed']

    def transform(self, data):
        logging.info('Initiating Text Preparation processing...')
        interim_data = data.copy()
        logging.info(f'Lower {self.text_var}...')
        interim_data[self.text_process] = data[self.text_var].apply(lambda x: x.lower())
        logging.info(f'Remove punctualizations...')
        interim_data[self.text_process] = interim_data[self.text_process].apply(
            lambda x: x.translate(str.maketrans('', '', string.punctuation)))
        logging.info(f'Remove digits...')
        interim_data[self.text_process] = interim_data[self.text_process].apply(lambda x: x.translate(
            str.maketrans('', '', string.digits)))
        logging.info(f'Tokenize words...')
        interim_data[self.text_process] = interim_data[self.text_process].apply(word_tokenize)
        logging.info(f'Remove stopwords...')
        interim_data[self.text_process] = interim_data[self.text_process].apply(remove_sw)
        logging.info(f'Stemming...')
        interim_data[self.text_process] = interim_data[self.text_process].apply(stemmer)
        logging.info('Text data successfully processed!')
        return interim_data
