#!/usr/bin/env python3

#
# Description
#

import argparse
from typing import NamedTuple


# Main -----------------------------------------------------------------------------------------------------------------
def run_validate(config: str,
                 mode: str,
                 bucket: str):

    # Libraries --------------------------------------------------------------------------------------------------------
    import logging
    import yaml
    from src.validate import DataValidator
    from src.helpers import load_data
    import sys

    # Settings ---------------------------------------------------------------------------------------------------------
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    try:
        logging.info('Initializing configuration...')
        stream = open(config, 'r')
        config = yaml.load(stream=stream, Loader=yaml.FullLoader)
        validator = DataValidator(config=config)
        df = load_data(input_path=config['raw_path'], input_data=config['raw_data'], mode=mode, bucket=bucket)
        validation_status = validator.validate(df=df)
        validator.check_validity(validation_status=validation_status)
    except RuntimeError as error:
        logging.info(error)
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run data validation")
    parser.add_argument('--config',
                        default='config.yaml',
                        help='path to configuration yaml file')
    parser.add_argument('--mode',
                        required=False,
                        help='where you run the pipeline')
    parser.add_argument('--bucket',
                        required=False,
                        help='if cloud, the staging bucket')

    args = parser.parse_args()
    CONFIG = args.config
    MODE = args.mode
    BUCKET = args.bucket
    run_validate(config=CONFIG,
                 mode=MODE,
                 bucket=BUCKET)
