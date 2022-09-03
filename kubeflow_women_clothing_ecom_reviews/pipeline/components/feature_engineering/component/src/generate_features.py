#!/usr/bin/env python3

#
# Description
#

# Libraries ------------------------------------------------------------------------------------------------------------

## General
import sys
import logging
from .helpers import fit_tf_idf, fit_min_max_scaler, get_text_features, get_nlp_features, get_tfidf_df, get_scaled_df


# FeatureEngineer ------------------------------------------------------------------------------------------------------


class FeaturesGenerator:

    def __init__(self, config):
        self.random_state = config['random_state']
        self.text_variables = config['text_variables']
        self.tf_idf_vectorizer = None
        self.min_max_scaler = None

    def fit(self, x, est_param, params=None):
        if est_param == 'tf_idf':
            tf_idf_vectorizer = fit_tf_idf(x, self.text_variables[1], params=params)
            self.tf_idf_vectorizer = tf_idf_vectorizer
        elif est_param == 'min_max_scale':
            min_max_scaler = fit_min_max_scaler(x, params=params)
            self.min_max_scaler = min_max_scaler

    def transform(self, x, est_param):
        try:
            if est_param == 'tf_idf':
                logging.info('Processing text features...')
                df_text_feats = get_text_features(x, self.text_variables[0])
                logging.info('Processing nlp features...')
                df_nlp_feats = get_nlp_features(df_text_feats, self.text_variables[0])
                logging.info('Processing tf-idf matrix...')
                tf_idf_matrix = self.tf_idf_vectorizer.transform(df_nlp_feats[self.text_variables[1]])
                tfidf_df = get_tfidf_df(df_nlp_feats, self.text_variables, tf_idf_matrix,
                                        self.tf_idf_vectorizer.get_feature_names())
                return tfidf_df

            elif est_param == 'min_max_scale':
                logging.info('Scaling data...')
                scaled_matrix = self.min_max_scaler.transform(x)
                scaled_df = get_scaled_df(scaled_matrix, x)
                return scaled_df

        except RuntimeError as error:
            logging.error(error)
            sys.exit(1)
