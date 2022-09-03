#!/usr/bin/env python3

#
# Description
#

import argparse
from typing import NamedTuple
from kfp.dsl.types import GCSPath


# Main -----------------------------------------------------------------------------------------------------------------
# TODO: Search or Ask How GCSPath works. Because I need to install kfp SDK to run this code
def run_generate_features(config: str,
                          mode: str,
                          bucket: str,
                          train_path: 'GCSPath',
                          test_path: 'GCSPath',
                          val_path: 'GCSPath') -> NamedTuple('output_paths', [('train', 'GCSPath'),
                                                                              ('test', 'GCSPath'),
                                                                              ('val', 'GCSPath')]):
    # Libraries --------------------------------------------------------------------------------------------------------
    import logging.config
    import yaml
    import sys
    import os
    from src.generate_features import FeaturesGenerator
    from src.helpers import load_data, save_data

    # Settings ---------------------------------------------------------------------------------------------------------
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    try:
        logging.info('Initializing configuration...')
        stream = open(config, 'r')
        config = yaml.load(stream=stream, Loader=yaml.FullLoader)
        feats_generator = FeaturesGenerator(config=config)

        logging.info('Initiating features engineering process...')
        if mode == 'cloud':
            output_paths_gcs = []

            # Train ----------------------------------------------------------------------------------------------------
            train_data = load_data(input_data=train_path, mode=mode)
            y_train = train_data[[config['target']]]
            x_train = train_data[train_data.columns.difference([config['target']])]

            feats_generator.fit(x=x_train, est_param=config['est_params'][0])
            x_train_tf_idf_df = feats_generator.transform(x=x_train, est_param=config['est_params'][0])

            feats_generator.fit(x=x_train_tf_idf_df, est_param=config['est_params'][1])
            x_train_scaled = feats_generator.transform(x=x_train_tf_idf_df, est_param=config['est_params'][1])

            train_path_gcs = save_data(x_df=x_train_scaled, y_df=y_train, path=config['featured_path'],
                                       out_data=config['featured_data'][0], mode=mode, bucket=bucket)
            output_paths_gcs.append(train_path_gcs)

            # Test - Val -----------------------------------------------------------------------------------------------
            for input_path, out_filename in zip([test_path, val_path], config['featured_data'][1:]):
                data = load_data(input_data=input_path, mode=mode)
                y = data[[config['target']]]
                x = data[data.columns.difference([config['target']])]

                x_tf_idf_matrix = feats_generator.transform(x=x, est_param=config['est_params'][0])
                x_scaled = feats_generator.transform(x=x_tf_idf_matrix, est_param=config['est_params'][1])

                x_path_gcs = save_data(x_df=x_scaled, y_df=y, path=config['featured_path'],
                                       out_data=out_filename, mode=mode, bucket=bucket)

                output_paths_gcs.append(x_path_gcs)

            return tuple(output_paths_gcs)
        else:
            output_paths = []
            data_path = os.path.join(config['processed_path'], config['processed_data'][0])

            # Train ----------------------------------------------------------------------------------------------------
            train_data = load_data(input_data=data_path, mode=mode)
            y_train = train_data[[config['target']]]
            x_train = train_data[train_data.columns.difference([config['target']])]

            feats_generator.fit(x=x_train, est_param=config['est_params'][0])
            x_train_tf_idf_df = feats_generator.transform(x=x_train, est_param=config['est_params'][0])

            feats_generator.fit(x=x_train_tf_idf_df, est_param=config['est_params'][1])
            x_train_scaled = feats_generator.transform(x=x_train_tf_idf_df, est_param=config['est_params'][1])

            train_path = save_data(x_df=x_train_scaled, y_df=y_train, path=config['featured_path'],
                                   out_data=config['featured_data'][0], mode=mode, bucket=bucket)
            output_paths.append(train_path)

            # Test - Val -----------------------------------------------------------------------------------------------
            for input_path, out_filename in zip(config['processed_data'][1:], config['featured_data'][1:]):
                data_path = os.path.join(config['processed_path'], input_path)
                data = load_data(input_data=data_path, mode=mode)
                y = data[[config['target']]]
                x = data[data.columns.difference([config['target']])]

                x_tf_idf_matrix = feats_generator.transform(x=x, est_param=config['est_params'][0])
                x_scaled = feats_generator.transform(x=x_tf_idf_matrix, est_param=config['est_params'][1])

                x_path = save_data(x_df=x_scaled, y_df=y, path=config['featured_path'],
                                   out_data=out_filename, mode=mode, bucket=bucket)

                output_paths.append(x_path)
            return tuple(output_paths)

    except RuntimeError as error:
        logging.info(error)
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run data collector")
    parser.add_argument('--config',
                        default='config.yaml',
                        help='path to configuration yaml file')
    parser.add_argument('--mode',
                        required=False,
                        help='where you run the pipeline')
    parser.add_argument('--bucket',
                        required=False,
                        help='if cloud, the staging bucket')
    parser.add_argument('--train-path',
                        required=False,
                        help='if cloud, the path to train data')
    parser.add_argument('--test-path',
                        required=False,
                        help='if cloud, the path to test data')
    parser.add_argument('--val-path',
                        required=False,
                        help='if cloud, the path to val data')

    args = parser.parse_args()
    CONFIG = args.config
    MODE = args.mode
    BUCKET = args.bucket
    TRAIN_PATH = args.train_path
    TEST_PATH = args.test_path
    VAL_PATH = args.val_path
    run_generate_features(config=CONFIG,
                          mode=MODE,
                          bucket=BUCKET,
                          train_path=TRAIN_PATH,
                          test_path=TEST_PATH,
                          val_path=VAL_PATH)
