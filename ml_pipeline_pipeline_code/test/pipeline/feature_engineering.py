'''
data_preprocessing module contains machine learning object templates
'''
# Data Preparation
import numpy as np
import pandas as pd

# Model Training
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

#Sklearn pipeline
from sklearn.base import BaseEstimator, TransformerMixin

#Utils
import logging
import joblib
import ruamel.yaml as yaml
import warnings
warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)

class Binner(BaseEstimator, TransformerMixin):

    """ A transformer that returns DataFrame 
    with bins based on variable distributions.

    Parameters
    ----------
    binning_meta : list, default=None

    """

    def __init__(self, binning_meta=None):
        if not isinstance(binning_meta, dict):
            logging.error('The config file is corrupted in binning_meta key!')
            sys.exit(1)
        else:
            self.binning_meta = binning_meta

    # We have fit method cause Sklearn Pipeline
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for var, meta in self.binning_meta.items():
            X[meta['var_name']] = pd.cut(X[var], bins = meta['bins'], labels = meta['bins_labels'], include_lowest = True)
            X.drop(var, axis=1, inplace=True)
        return X

class Encoder(BaseEstimator, TransformerMixin):

    """ A transformer that returns DataFrame 
    with variable encoded.

    Parameters
    ----------
    encoding_meta : list, default=None

    """
    
    def __init__(self, encoding_meta=None):
        if not isinstance(encoding_meta, dict):
            logging.error('The config file is corrupted in encoding_meta key!')
            sys.exit(1)
        else:
            self.encoding_meta = encoding_meta


    # We have fit method cause Sklearn Pipeline
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for var, meta in self.encoding_meta.items():
            if var not in X.columns.values.tolist():
                pass
            X[var] = X[var].map(meta)
        return X

class Dumminizer(BaseEstimator, TransformerMixin):

    """ A transformer that returns DataFrame 
    with dummies.

    Parameters
    ----------
    columns_to_dummies: list, default=None
    dummies_meta : dict, default=None

    """

    def __init__(self, columns_to_dummies):
        if not isinstance(columns_to_dummies, list):
            logging.error('The config file is corrupted in nominal_predictors keys!')
            sys.exit(1)
        else:
            self.columns_to_dummies = columns_to_dummies

    # We have fit method cause Sklearn Pipeline
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = pd.get_dummies(X, columns=self.columns_to_dummies)
        return X

class Scaler(BaseEstimator, TransformerMixin):

    """ A transformer that returns DataFrame 
    with scaled variables via MinMaxScaler sklearn class.

    Parameters
    ----------
    columns_to_scale: list, default=None

    """

    def __init__(self, features):
        if not isinstance(features, list):
            logging.error('The config file is corrupted in features key!')
            sys.exit(1)
        else:
            self.features = features
            self.scaler = MinMaxScaler()

    # We have fit method cause Sklearn Pipeline
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        scaler_fit = self.scaler.fit(X[self.features]) 
        X[self.features] = scaler_fit.transform(X[self.features])
        return X

class Feature_selector(BaseEstimator, TransformerMixin):

    """ A transformer that returns DataFrame 
    with selected features

    Parameters
    ----------
    features_selected: list, default=None

    """

    def __init__(self, features_selected):
        if not isinstance(features_selected, list):
            logging.error('The config file is corrupted in features_selected key!')
            sys.exit(1)
        else:
            self.features_selected = features_selected

    # We have fit method cause Sklearn Pipeline
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X[self.features_selected]
        return X