#!/usr/bin/env python3

#
# Description
#

# Libraries ------------------------------------------------------------------------------------------------------------

## General
import sys
import logging

from .helpers import *


class Modeler:
    def __init__(self, clf, params=None):
        # TODO: Add hyper-tuning strategy
        if params:
            self.clf = clf(**params)
        else:
            self.clf = clf()

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x_test):
        return self.clf.predict(x_test)

    def evaluate(self, y_true, y_pred, params=None):
        metrics_names = ["accuracy", "balanced_accuracy", "average_precision", "neg_brier_score",
                         "f1", "neg_log_loss", "precision", "recall", "jaccard", "roc_auc"]
        if params:
            metrics_values = [
                get_acc_score(y_true, y_pred, params),
                get_balanced_acc_score(y_true, y_pred, params),
                get_average_precision_score(y_true, y_pred, params),
                get_brier_score_loss(y_true, y_pred, params),
                get_f1_score(y_true, y_pred, params),
                get_log_loss(y_true, y_pred, params),
                get_precision_score(y_true, y_pred, params),
                get_recall_score(y_true, y_pred, params),
                get_jaccard_score(y_true, y_pred, params),
                get_roc_auc_score(y_true, y_pred, params)
            ]
            metrics = dict(zip(metrics_names, metrics_values))
        else:
            metrics_values = [
                get_acc_score(y_true, y_pred),
                get_balanced_acc_score(y_true, y_pred),
                get_average_precision_score(y_true, y_pred),
                get_brier_score_loss(y_true, y_pred),
                get_f1_score(y_true, y_pred),
                get_log_loss(y_true, y_pred),
                get_precision_score(y_true, y_pred),
                get_recall_score(y_true, y_pred),
                get_jaccard_score(y_true, y_pred),
                get_roc_auc_score(y_true, y_pred)
            ]
            metrics = dict(zip(metrics_names, metrics_values))
        return metrics
