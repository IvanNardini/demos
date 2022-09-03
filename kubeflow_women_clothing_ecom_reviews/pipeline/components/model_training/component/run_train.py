#!/usr/bin/env python3

#
# Description
#

import argparse
from typing import NamedTuple
from kfp.dsl.types import GCSPath


# Main -----------------------------------------------------------------------------------------------------------------
# TODO: Search or Ask How GCSPath works. Because I need to install kfp SDK to run this code
def run_train(config: str,
              mode: str,
              bucket: str,
              train_path: 'GCSPath',
              test_path: 'GCSPath',
              classifier='logit') -> NamedTuple('output_paths', [('train', 'GCSPath'),
                                                                 ('test', 'GCSPath'),
                                                                 ('model', 'GCSPath')]):
    # Libraries --------------------------------------------------------------------------------------------------------

    import logging.config
    import yaml
    import sys
    import os
    import pprint
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from src.train_model import Modeler
    from src.helpers import load_data, save_data

    # Settings ---------------------------------------------------------------------------------------------------------
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    logging.info('Initializing configuration...')
    stream = open(config, 'r')
    config = yaml.load(stream=stream, Loader=yaml.FullLoader)

    logging.info("Initializing model...")
    if classifier == 'logit':
        modeler = Modeler(LogisticRegression, params={'max_iter': config['models']['logit']['max_iter'],
                                                      'random_state': config['random_state']})
    if classifier == 'dtree':
        modeler = Modeler(DecisionTreeClassifier, params={'random_state': config['random_state']})
    if classifier == 'rf':
        modeler = Modeler(RandomForestClassifier, params={'random_state': config['random_state']})
    if classifier == 'gb':
        modeler = Modeler(GradientBoostingClassifier, params={'random_state': config['random_state']})
    if classifier == 'xgb':
        modeler = Modeler(XGBClassifier, params={'use_label_encoder': config['models']['xgb']['use_label_encoder'],
                                                 'random_state': config['random_state']})
    if classifier == 'lightgb':
        modeler = Modeler(LGBMClassifier, params={'random_state': config['random_state']})
    logging.info(f"{classifier} model successfully initialized!")

    try:
        logging.info('Starting model training...')
        if not mode:
            train_path = os.path.join(config['featured_path'], config['featured_data'][0])
            test_path = os.path.join(config['featured_path'], config['featured_data'][1])

        # Train --------------------------------------------------------------------------------------------------------
        logging.info(f'Training {classifier} model...')
        train_data = load_data(input_data=train_path, mode=mode)
        y_train = train_data[config['target']]
        x_train = train_data[train_data.columns.difference([config['target']])]
        modeler.train(x_train, y_train)
        logging.info(f'{classifier} model successfully trained!')

        logging.info(f'Testing {classifier} model...')
        # Predict and Evalutate ----------------------------------------------------------------------------------------
        test_data = load_data(input_data=test_path, mode=mode)
        x_test = test_data[train_data.columns.difference([config['target']])]
        y_pred = modeler.predict(x_test)
        metrics = modeler.evaluate(y_train, y_pred)
        # TODO: Store metrics. Figure out how to consume in the next stage for model validation.
        pprint.pprint(metrics)

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
    parser.add_argument('--classifier',
                        default='logit',
                        required=False,
                        help='The classifier to train')

    args = parser.parse_args()
    CONFIG = args.config
    MODE = args.mode
    BUCKET = args.bucket
    TRAIN_PATH = args.train_path
    TEST_PATH = args.test_path
    CLASSIFIER = args.classifier
    run_train(config=CONFIG,
              mode=MODE,
              bucket=BUCKET,
              train_path=TRAIN_PATH,
              test_path=TEST_PATH,
              classifier=CLASSIFIER)
