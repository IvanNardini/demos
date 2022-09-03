'''
features_engineering module contains feature engineering object templates
'''
# Data Preparation
import numpy as np
import pandas as pd

# Model Training
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

#Utils
import logging
import joblib
import ruamel.yaml as yaml
import warnings
warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)

class FeatureEngineering:

    def target_encoder(self, target, labels_dic):
        '''
        Encode target
        :params: target, labels_dic
        :return: target_encoded
        '''
        target_encoded = target.map(labels_dic).astype('category')
        return target_encoded

    def binner(self, data, binning_meta):
        '''
        Create bins based on variable distributions
        :params: data, binning_meta
        :return: DataFrame
        '''
        data = data.copy()
        for var, meta in binning_meta.items():
            data[meta['var_name']] = pd.cut(data[var], bins = meta['bins'], labels=meta['bins_labels'], include_lowest=True)
            data.drop(var, axis=1, inplace=True)
        return data

    def encoder(self, data, encoding_meta):
        '''
        Encode all variables for training
        :params: data, encoding_meta
        :return: DataFrame
        '''
        data = data.copy()
        for var, meta in encoding_meta.items():
            if var not in data.columns.values.tolist():
                pass
            data[var] = data[var].map(meta)
        return data

    def dumminizer(self, data, columns_to_dummies):
        '''
        Generate dummies for nominal variables
        :params: data, columns_to_dummies, dummies_meta
        :return: DataFrame
        '''
        data = data.copy()
        data = pd.get_dummies(data, columns=columns_to_dummies)
        return data

    def scaler_trainer(self, data):
        '''
        Fit the scaler on predictors
        :params: data
        :return: scaler
        '''
        data = data.copy()
        scaler = MinMaxScaler()
        scaler.fit(data)
        return scaler
    
    def scaler_transformer(self, data, features, scaler):
        '''
        Trasform the data 
        :params: data, scaler
        :return: DataFrame
        ''' 
        data = data.copy()
        data[features] = scaler.transform(data[features])
        return data[features]

    def features_selector(self, data, features_selected):
        '''
        Select features
        :params: data, features_selected
        :return: DataFrame
        '''
        data = data.copy()
        data = data[features_selected]
        return data

    def balancer(self, data, target, random_state):
        '''
        Balance data with SMOTE
        :params: data, target, random_state
        : X, y
        '''
        data = data.copy()
        smote = SMOTE(random_state=random_state)
        X, y = smote.fit_resample(data, target)
        return X,y
