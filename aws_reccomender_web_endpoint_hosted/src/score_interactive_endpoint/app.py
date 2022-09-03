# -*- coding: utf-8 -*-

'''
The Dash instance module
'''

import os
import logging
from collections import defaultdict
import pandas as pd
from surprise import dump

import dash

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions = True)
server = app.server

def locate_model(dest):
    
    """ 
    return path of binary model
    args:
       dest: folder for searching
    returns:
       model_path
    """

    for dirpath, dirnames, filenames in os.walk(dest):
        for filename in [f for f in filenames if f.endswith((".pkl", ".pickle"))]:
            model_path = os.path.join(dirpath, filename)
            return model_path
    return None

def model_reader(model_path):
    """ 
    return predictions and model class
    args:
       model_path: pickle file path
    returns:
       predictions and model
    """
    predictions, algo = dump.load(model_path)
    return predictions, algo

def get_top(predictions, n=10):
    
    '''
    Returns the the top-N recommendation from a set of predictions
    args:
       predictions: predictions generated in testing phase
       n: number of items to suggest (default=10)
    return:
       top_n dictionary: user-prediction dictionaries
    '''
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
        
    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
        
    return top_n

def get_top_n_ui(top, uid):
    '''
    Returns the list of selected items
    args:
       top: user-prediction dictionaries
       uid: user id for filtering
    return:
       top_n dictionary: user-prediction dictionaries
    '''
    try:
        top_n_ui = [[iid for (iid, _) in user_ratings] for UID, user_ratings in top.items() if UID==uid][0]
        return top_n_ui
    except ValueError: 
        return 0
