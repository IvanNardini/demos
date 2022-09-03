#!/usr/bin/env python3

#
# Description
#

# Libraries ------------------------------------------------------------------------------------------------------------

import logging
import sys

import pandas as pd
from tensorflow.io import gfile
from sklearn.metrics import accuracy_score, balanced_accuracy_score, \
    average_precision_score, brier_score_loss, f1_score, log_loss, \
    precision_score, recall_score, jaccard_score, roc_auc_score
import os
from pathlib import Path


# Helpers --------------------------------------------------------------------------------------------------------------

def load_data(input_data, mode):
    logging.info(f'Loading data from {input_data}...')
    if mode == 'cloud':
        with gfile.GFile(name=input_data, mode='r') as file:
            df = pd.read_csv(file)
    else:
        df = pd.read_csv(input_data)
    logging.info(f'{input_data} successfully loaded!')
    return df


def get_acc_score(y_true, y_pred, params=None):
    if params:
        return accuracy_score(y_true, y_pred, **params)
    return round(accuracy_score(y_true, y_pred), 3)

def get_balanced_acc_score(y_true, y_pred, params=None):
    if params:
        return balanced_accuracy_score(y_true, y_pred, **params)
    return round(balanced_accuracy_score(y_true, y_pred), 3)

def get_average_precision_score(y_true, y_pred, params=None):
    if params:
        return average_precision_score(y_true, y_pred, **params)
    return round(average_precision_score(y_true, y_pred), 3)

def get_brier_score_loss(y_true, y_pred, params=None):
    if params:
        return brier_score_loss(y_true, y_pred, **params)
    return round(brier_score_loss(y_true, y_pred), 3)

def get_f1_score(y_true, y_pred, params=None):
    if params:
        return f1_score(y_true, y_pred, **params)
    return round(f1_score(y_true, y_pred), 3)

def get_log_loss(y_true, y_pred, params=None):
    if params:
        return log_loss(y_true, y_pred, **params)
    return round(log_loss(y_true, y_pred), 3)

def get_precision_score(y_true, y_pred, params=None):
    if params:
        return precision_score(y_true, y_pred, **params)
    return round(precision_score(y_true, y_pred), 3)

def get_recall_score(y_true, y_pred, params=None):
    if params:
        return recall_score(y_true, y_pred, **params)
    return round(recall_score(y_true, y_pred),3)

def get_jaccard_score(y_true, y_pred, params=None):
    if params:
        return jaccard_score(y_true, y_pred, **params)
    return round(jaccard_score(y_true, y_pred), 3)

def get_roc_auc_score(y_true, y_pred, params=None):
    if params:
        return roc_auc_score(y_true, y_pred, **params)
    return round(roc_auc_score(y_true, y_pred), 3)

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

def save_model():
    pass
