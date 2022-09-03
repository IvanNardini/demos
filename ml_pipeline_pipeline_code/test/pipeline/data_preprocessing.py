'''
data_preprocessing module contains machine learning object templates
'''
# Data Preparation
import numpy as np
import pandas as pd

# Model Training
from sklearn.preprocessing import MinMaxScaler

#Sklearn pipeline
from sklearn.base import BaseEstimator, TransformerMixin

#Utils
import sys
import logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
import joblib
import ruamel.yaml as yaml
import warnings
warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)

class Dropper(BaseEstimator, TransformerMixin):

    """ A transformer that returns dataframe
    without selected columns.

    Parameters
    ----------
    dropped_columns : list, default=None

    """
    
    def __init__(self, dropped_columns=None):
        if not isinstance(dropped_columns, list):
            logging.error('The config file is corrupted either dropped_columns key!')
            sys.exit(1)
        else:
            self.dropped_columns = dropped_columns

    # We have fit method cause Sklearn Pipeline
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X.drop(self.dropped_columns, axis=1, inplace=True)
        return X

class Renamer(BaseEstimator, TransformerMixin):
    """ A transformer that returns dataframe
    with renamed columns.

    Parameters
    ----------
    renamed_columns : list, default=None

    """
    
    def __init__(self, renamed_columns=None):
        if not isinstance(renamed_columns, dict):
            logging.error('The config file is corrupted either renamed_columns key!')
            sys.exit(1)
        else:
            self.renamed_columns = renamed_columns

    # We have fit method cause Sklearn Pipeline
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X.rename(columns=self.renamed_columns, inplace=True)
        return X


class Missing_Imputer(BaseEstimator, TransformerMixin):

    """ A transformer that returns DataFrame
    with missings imputed.

    Parameters
    ----------
    missing_predictors : list, default=None
    replace : str, default=missing

    """

    def __init__(self, missing_predictors=None, replace='missing'):
        if not isinstance(missing_predictors, list):
            logging.error('The config file is corrupted in missing_predictors key!')
            sys.exit(1)
        else:
            self.missing_predictors = missing_predictors
            self.replace = replace

    # We have fit method cause Sklearn Pipeline
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.missing_predictors] = X[self.missing_predictors].replace('?', self.replace)
        return X