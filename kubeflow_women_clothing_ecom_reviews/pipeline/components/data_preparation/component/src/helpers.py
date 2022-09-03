#!/usr/bin/env python3

#
# Description
#

# Libraries ------------------------------------------------------------------------------------------------------------

import os
import logging
from pathlib import Path

import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem import PorterStemmer

from tensorflow.io import gfile


# Helpers ------------------------------------------------------------------------------------------------------------

def remove_sw(words_list):
    stop_words = stopwords.words("english")
    return [word for word in words_list if word not in stop_words]


def stemmer(words_list):
    ps = PorterStemmer()
    return [ps.stem(word) for word in words_list]

def load_data(input_data, mode):
    logging.info(f'Loading data to {input_data}...')
    if mode == 'cloud':
        with gfile.GFile(name=input_data, mode='r') as file:
            df = pd.read_csv(file)
    else:
        df = pd.read_csv(input_data)
    logging.info(f'{input_data} successfully loaded!')
    return df

def save_data(df, path, out_data, mode, bucket):
    if mode == 'cloud':
        out_csv_gcs = f'{bucket}/{path}/{out_data}'
        logging.info(f'Writing {out_csv_gcs} file...')
        with gfile.GFile(name=out_csv_gcs, mode='w') as file:
            df.to_csv(file, index=False)
        logging.info(f'{out_csv_gcs} successfully loaded!')
        return out_csv_gcs
    else:
        p = Path(path)
        if not p.exists():
            os.mkdir(path)
        out_csv = f'{path}/{out_data}'
        logging.info(f'Writing {out_csv} file...')
        df.to_csv(out_csv, index=False)
        logging.info(f'{out_csv} successfully loaded!')
        return out_csv
