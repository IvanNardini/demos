#!/usr/bin/env python3

#
# Description
#

# Libraries ------------------------------------------------------------------------------------------------------------

import logging
import sys

import pandas as pd
from tensorflow.io import gfile
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import os
from pathlib import Path


# Helpers --------------------------------------------------------------------------------------------------------------

def load_data(input_data, mode):
    logging.info(f'Loading data to {input_data}...')
    if mode == 'cloud':
        with gfile.GFile(name=input_data, mode='r') as file:
            df = pd.read_csv(file)
    else:
        df = pd.read_csv(input_data)
    logging.info(f'{input_data} successfully loaded!')
    return df


def get_count_words(s):
    return len(str(s).split(" "))


def get_count_char(s):
    return sum(len(w) for w in str(s).split(" "))


def get_count_sents(s):
    return len(str(s).split("."))


def get_count_exc_marks(s):
    return s.count('!')


def get_count_question_marks(s):
    return s.count('?')


def get_count_pct(s):
    return len([w for w in s if w in '"#$%&\'()*+,-./:;<=>@[\\]^_`{|}~'])


def get_count_cap(s):
    return sum(1 for w in s if w.isupper())


def get_polarity(s):
    tb = TextBlob(s)
    return tb.sentiment.polarity


def get_subjectivity(s):
    tb = TextBlob(s)
    return tb.sentiment.subjectivity


def get_text_features(df, text_var):
    df_copy = df.copy()
    # word count
    logging.info('Get count words feature...')
    df_copy['word_count'] = df_copy[text_var].apply(get_count_words)
    # character count
    logging.info('Get count characters feature...')
    df_copy['char_count'] = df_copy[text_var].apply(get_count_char)
    # sentence count
    logging.info('Get count sentences feature...')
    df_copy['sentence_count'] = df_copy[text_var].apply(get_count_sents)
    # count capitals
    logging.info('Get capitals words feature...')
    df_copy['capitals_count'] = df_copy[text_var].apply(get_count_cap)
    # count puncts
    logging.info('Get count punctuation features...')
    df_copy['punc_count'] = df_copy[text_var].apply(get_count_pct)
    df_copy['exc_marks_count'] = df_copy[text_var].apply(get_count_exc_marks)
    df_copy['question_marks_count'] = df_copy[text_var].apply(get_count_question_marks)
    # avg word len
    logging.info('Get word density feature...')
    df_copy['avg_word_len'] = df_copy['char_count'] / df_copy['word_count']
    # avg sentence len
    logging.info('Get sentence density feature...')
    df_copy['avg_sentence_len'] = df_copy['word_count'] / df_copy['sentence_count']
    # avg cap
    logging.info('Get capitals density feature...')
    df_copy['avg_cap_len'] = df_copy.apply(lambda row: float(row['capitals_count']) / float(row['word_count']),
                                           axis=1)
    return df_copy


def get_nlp_features(df, text_var):
    df_copy = df.copy()
    # polarity
    logging.info('Get polarity feature...')
    df_copy['polarity'] = df_copy[text_var].apply(get_polarity)
    # subjectivity
    logging.info('Get subjectivity feature...')
    df_copy['subjectivity'] = df_copy[text_var].apply(get_subjectivity)
    return df_copy


def fit_tf_idf(data, text_var, params=None):
    logging.info('Train TfidfTransformer...')
    try:
        if params:
            tf_idf_vectorizer = TfidfVectorizer(**params)
        else:
            tf_idf_vectorizer = TfidfVectorizer()
        tf_idf_vectorizer = tf_idf_vectorizer.fit(data[text_var])
    except RuntimeError as error:
        logging.error(error)
        sys.exit(1)
    else:
        logging.info('TfidfTransformer successfully trained!')
    return tf_idf_vectorizer


def fit_min_max_scaler(data, params=None):
    logging.info('Train MinMaxScaler...')
    try:
        if params:
            scaler = MinMaxScaler(**params)
        else:
            scaler = MinMaxScaler()
        scaler = scaler.fit(data)
    except RuntimeError as error:
        logging.error(error)
        sys.exit(1)
    else:
        logging.info('MinMaxScaler successfully trained!')
    return scaler


def get_tfidf_df(df, text_cols, tfidf_matrix, cols):
    df_copy = df.copy()

    logging.info('Get Tf-Idf dataframe...')
    df_copy = df_copy.drop(text_cols, axis=1)
    tfidf_plain = tfidf_matrix.toarray()
    tfidf = pd.DataFrame(tfidf_plain, columns=cols)
    tfidf_df = pd.merge(df_copy, tfidf, how="left", left_index=True, right_index=True)
    logging.info('Tf-Idf successfully created!')

    return tfidf_df


def get_scaled_df(matrix, df):
    scaled_df = pd.DataFrame(matrix, columns=df.columns)
    return scaled_df


def save_data(x_df, y_df, path, out_data, mode, bucket):
    df = pd.merge(x_df, y_df, how="left", left_index=True, right_index=True)
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
