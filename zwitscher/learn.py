#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A script to learn classification
"""
import pickle
import os
import uuid

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier

from utils.PCC import load as load_pcc
from gold_standard import pcc_to_gold
from features import discourse_connective_text_featurizer

__author__ = 'arkadi'


def load_data(connector_folder='/media/arkadi/arkadis_ext/NLP_data/ger_twitter/' +
                               'potsdam-commentary-corpus-2.0.0/connectors'):
    # Load PCC data
    pcc = load_pcc(connector_folder)
    # creating a pd.DataFrame
    return pcc_to_gold(pcc)

def clean_data(dataframe):
    """ Transforms all Nones int np.NaN and drops rows where connective_position or sentences is NaN

    :param dataframe: Gold data from the PCC
    :type dataframe: pd.DataFrame
    :return: Cleaned data
    :rtype: pd.DataFrame
    """
    dataframe[dataframe.isnull()] = np.NaN
    dataframe = dataframe.dropna(subset=['connective_positions', 'sentences'])
    return dataframe

def featurize_data(dataframe, feature_function):
    """ Creates features as specified by the featurizer that creates a feature dict from
    sentences and connective positions

    Sentences are a list of lists with tokens
    Connective positions is a list with pairs of sentence and token indices
    :param dataframe: data with columns 'sentences' and 'connective_positions'
    :type dataframe: pd.DataFrame
    :return: features
    :rtype: pd.DataFrame
    """
    features = dataframe.apply(lambda row: pd.Series(feature_function(row['sentences'], row['connective_positions'])),
                               axis=1, reduce=False)
    return features


def main(feature_list=['connective_lexical', 'length_connective',
                       'length_prev_sent', 'length_same_sent', 'length_next_sent',
                       'tokens_before', 'tokens_after', 'tokens_between'],
         label_features=['connective_lexical'],
         connector_folder='/media/arkadi/arkadis_ext/NLP_data/ger_twitter/' +
                          'potsdam-commentary-corpus-2.0.0/connectors',
         pickle_folder='data',
         unpickle_gold=True,
         pickle_classifier=True):
    """ The main learning function for a classifier

    Runs a random forest. Prints out scores from a 5-fold cross validation. Returns the classifier.
    :param connector_folder: folder with the gold data. Can be left empty when unpickling the gold data
    :type connector_folder: str
    :param pickle_folder: The folder where pickles are put. The names are dealt with internally
    :type pickle_folder: str
    :type unpickle_gold: bool
    :type pickle_classifier: bool
    :return: Classifier and Label encoder to decode labels for prediction
    :rtype: (sklearn.ensemble.forest.RandomForestClassifier, sklearn.preprocessing.LabelEncoder)
    """
    # Some checks on the function call
    if pickle_classifier or unpickle_gold:
        assert os.path.exists(pickle_folder), 'Pickle folder has to exist when using pickling'
    if not unpickle_gold:
        assert os.path.exists(connector_folder), 'Connector folder has to exist, when not unpickling connectors'

    print 'Loading data...'
    if unpickle_gold:
        print 'Unpickling gold data from %s' % os.path.join(pickle_folder, 'PCC_disc.pickle')
        if os.path.exists(os.path.join(pickle_folder, 'PCC_disc.pickle')):
            with open(os.path.join(pickle_folder, 'PCC_disc.pickle'), 'rb') as f:
                pcc_df = pcc_to_gold(pickle.load(f))
        else:
            print 'Could not find file %s' % os.path.join(pickle_folder, 'PCC_disc.pickle')
            return
    else:
        print 'Parsing gold data from %s' % connector_folder
        pcc_df = load_data(connector_folder)  # Loaded X and y into one dataframe
    print 'Loaded data'

    print 'Cleaning data...'
    clean_pcc = clean_data(pcc_df)
    print 'Cleaned data'

    print 'Calculating features...'
    # Taking our favorite featurizer
    featurizer = lambda sents, conn_pos: discourse_connective_text_featurizer(sents, conn_pos,
                                                                              feature_list=feature_list)
    features = featurize_data(clean_pcc, featurizer)  # Got features of X
    print 'Calculated all features'

    # We need to encode the non-numerical labels
    le = LabelEncoder()
    if label_features:
        print 'Encoding labels...'
        le.fit(features[label_features])
        features[label_features] = le.transform(features[label_features])
        print 'Encoded label'

    print 'Cross validating classifier...'
    clf = RandomForestClassifier(min_samples_leaf=5, n_jobs=-1, verbose=2)
    scores = cross_val_score(clf, features, clean_pcc['sentence_dist'], cv=5)
    print 'Cross validated classifier\nscores: %s\nmean score: %f' % (str(scores), scores.mean())

    print 'Learning classifier on the whole data set...'
    clf.fit(features, clean_pcc['sentence_dist'])
    print 'Learned classifier on the whole data set'

    if pickle_classifier:
        id = uuid.uuid4().get_hex()
        print 'Pickling classifier to %s' % os.path.join(pickle_folder, '%s_classifier.pickle' % str(id))
        with open(os.path.join(pickle_folder, '%s_classifier.pickle' % str(id)), 'wb') as f:
            pickle.dump(clf, f, pickle.HIGHEST_PROTOCOL)
        print 'Pickling label encoder to %s' % os.path.join(pickle_folder, '%s_encoder.pickle' % str(id))
        with open(os.path.join(pickle_folder, '%s_encoder.pickle' % str(id)), 'wb') as f:
            pickle.dump(clf, f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(pickle_folder, 'classifier.log'), 'a') as f:
            f.write('%s\t%f\t%s\n' % (id, scores.mean(), str(feature_list)))
    return clf, le

if __name__ == '__main__':

    feature_list = ['connective_lexical', 'length_prev_sent', 'length_connective',
                    'length_same_sent', 'tokens_before', 'length_next_sent']
    # 'tokens_before', 'tokens_after', ,             'length_next_sent',]

    main(feature_list=feature_list, label_features=['connective_lexical'], unpickle_gold=True, pickle_classifier=True)