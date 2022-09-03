#!/usr/bin/env python3

#
# Description
#

import argparse
from typing import NamedTuple

# Main -----------------------------------------------------------------------------------------------------------------
#TODO: Search or Ask How GCSPath works
def run_collect(config: str,
                mode: str,
                bucket: str) -> NamedTuple('output_paths',
                                           [('train', 'GCSPath'),
                                            ('test', 'GCSPath'),
                                            ('val', 'GCSPath')]):
    # Libraries --------------------------------------------------------------------------------------------------------
    import logging
    import yaml
    from collections import namedtuple
    import sys
    from src.collect import DataCollector

    # Settings ---------------------------------------------------------------------------------------------------------
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    try:
        logging.info('Initializing configuration...')
        stream = open(config, 'r')
        config = yaml.load(stream=stream, Loader=yaml.FullLoader)
        collector = DataCollector(config=config)
        raw_df = collector.extract(mode=mode, bucket=bucket)
        # TODO: Add metadata in the pipeline
        print(raw_df.head(5))
        x_train, x_test, x_val, y_train, y_test, y_val = collector.transform(raw_df)

        if mode == 'cloud':
            (train_path_gcs, test_path_gcs, val_path_gcs) = collector.load(x_train, x_test, x_val,
                                                                           y_train, y_test, y_val, mode=mode,
                                                                           bucket=bucket)
            out_gcs = namedtuple('output_paths', ['train', 'test', 'val'])
            return out_gcs(train_path_gcs, test_path_gcs, val_path_gcs)
        else:
            (train_path, test_path, val_path) = collector.load(x_train, x_test, x_val,
                                                               y_train, y_test, y_val, mode=mode, bucket=bucket)
            out_path = namedtuple('output_paths', ['train', 'test', 'val'])
            return out_path(train_path, test_path, val_path)
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
                        help='if cloud, the bucket to stage output')

    args = parser.parse_args()
    CONFIG = args.config
    MODE = args.mode
    BUCKET = args.bucket
    run_collect(config=CONFIG, mode=MODE, bucket=BUCKET)
