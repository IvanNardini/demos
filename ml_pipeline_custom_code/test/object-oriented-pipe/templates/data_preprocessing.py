'''
data_preprocessing module contains preprocessing object templates
'''
# Data Preparation
import numpy as np
import pandas as pd

# Model Training
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

#Utils
import logging
import joblib
import ruamel.yaml as yaml
import warnings
warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)

class Preprocessing:

    def dropper(self, data, columns_to_drop):
        '''
        Drop columns
        :params: data, columns_to_drop
        :return: DataFrame
        '''
        data = data.copy()
        data.drop(columns_to_drop, axis=1, inplace=True)
        return data

    def renamer(self, data, columns_to_rename):
        '''
        Rename columns
        :params: data, columns_to_rename
        :return: DataFrame
        '''
        data = data.copy()
        data.rename(columns=columns_to_rename, inplace=True)
        return data

    def anomalizier(self, data, anomaly_var):
        '''
        Drop anomalies 
        :params: data, anomaly_var
        :return: DataFrame
        '''
        data = data.copy()
        flt = data[anomaly_var]>=0
        return data[flt]

    def missing_imputer(self, data, columns_to_impute, replace='missing'):
        '''
        Imputes '?' character with 'missing' label
        :params:data, columns_to_impute, replace
        :return: Series
        '''
        data = data.copy()
        data[columns_to_impute] = data[columns_to_impute].replace('?', replace)
        return data

    def data_splitter(self, data, target, predictors, test_size, random_state):
        '''
        Split data in train and test samples
        :params: data, target, predictors, test_size, random_state
        :return: X_train, X_test, y_train, y_test
        '''
        data = data.copy()
        X_train, X_test, y_train, y_test = train_test_split(data[predictors],
                                                            data[target],
                                                            test_size=test_size,
                                                            random_state=random_state)
        return X_train, X_test, y_train, y_test
    
