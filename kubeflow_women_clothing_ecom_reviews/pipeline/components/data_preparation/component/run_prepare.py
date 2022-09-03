#!/usr/bin/env python3

#
# Description
#

import argparse
from typing import NamedTuple
from kfp.dsl.types import GCSPath

# Main -----------------------------------------------------------------------------------------------------------------
#TODO: Search or Ask How GCSPath works. Because I need to install kfp SDK to run this code
def run_prepare(config: str,
                mode: str,
                bucket: str,
                train_path: 'GCSPath',
                test_path: 'GCSPath',
                val_path: 'GCSPath') -> NamedTuple('output_paths', [('train', 'GCSPath'), ('test', 'GCSPath'), ('val', 'GCSPath')]):
    # Libraries --------------------------------------------------------------------------------------------------------
    import logging.config
    import yaml
    import sys
    import os
    from src.prepare import DataPreparer
    from src.helpers import load_data, save_data

    # Settings ---------------------------------------------------------------------------------------------------------
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    try:
        logging.info('Initializing configuration...')
        stream = open(config, 'r')
        config = yaml.load(stream=stream, Loader=yaml.FullLoader)
        preparer = DataPreparer(config=config)
        input_paths = [train_path, test_path, val_path]

        if mode == 'cloud':
            output_paths_gcs = []
            for input_path, out_filename in zip(input_paths, config['processed_data']):
                data = load_data(input_data=input_path, mode=mode)
                processed_data = preparer.transform(data=data)
                # TODO: Add metadata in the pipeline
                print(processed_data.head(5))
                out_path_gcs = save_data(df=processed_data, path=config['processed_path'],
                                         out_data=out_filename, mode=mode, bucket=bucket)
                output_paths_gcs.append(out_path_gcs)
            return tuple(output_paths_gcs)

        else:
            output_paths = []
            for input_filename, out_filename in zip(config['interim_data'], config['processed_data']):
                data_path = os.path.join(config['interim_path'], input_filename)
                data = load_data(input_data=data_path, mode=mode)
                processed_data = preparer.transform(data=data)
                # TODO: Add metadata in the pipeline
                print(processed_data.head(5))
                out_path = save_data(df=processed_data, path=config['processed_path'],
                                     out_data=out_filename, mode=mode, bucket=bucket)
                output_paths.append(out_path)
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
    run_prepare(config=CONFIG,
                mode=MODE,
                bucket=BUCKET,
                train_path=TRAIN_PATH,
                test_path=TEST_PATH,
                val_path=VAL_PATH)
