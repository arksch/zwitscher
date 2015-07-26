#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Methods needed for learning classifiers
"""
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier

from utils.PCC import load_connectors as load_pcc
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


def learn_sentdist(clean_pcc,
                   feature_list=['connective_lexical', 'length_connective',
                                 'length_prev_sent', 'length_same_sent', 'length_next_sent',
                                 'tokens_before', 'tokens_after', 'tokens_between'],
                   label_features=['connective_lexical']):
    """ Learning a classifier for the distance of arguments from a connective

    Runs a random forest. Prints out accuracy scores from a 5-fold cross validation.
    Returns the classifier and the label encoder that was used.
    :param clean_pcc: Cleaned PCC data, no NaNs
    :type clean_pcc: pd.DataFrame
    :param feature_list: list of features that shall be calculated with discourse_connective_text_featurizer
    :param label_features: list of features that have to be encoded as labels
    :return: trained classifier, score array and label encoder
    :rtype: tuple
    """
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
        # LabelEncoder only deals with 1 dim np.arrays
        le.fit(features[label_features].values.ravel())
        for feat in label_features:
            features[feat] = le.transform(features[feat])
        print 'Encoded label'

    print 'Cross validating classifier...'
    clf = RandomForestClassifier(min_samples_leaf=5, n_jobs=-1, verbose=1)
    scores = cross_val_score(clf, features, clean_pcc['sentence_dist'], cv=5)
    print 'Cross validated classifier\nscores: %s\nmean score: %f' % (str(scores), scores.mean())

    print 'Learning classifier on the whole data set...'
    clf.fit(features, clean_pcc['sentence_dist'])
    print 'Learned classifier on the whole data set'

    return clf, scores, le


def learn_main_arg_node(clean_pcc,
                          feature_list=['connective_lexical', 'length_connective',
                                        'length_prev_sent', 'length_same_sent', 'length_next_sent',
                                        'tokens_before', 'tokens_after', 'tokens_between'],
                          internal_argument=True):
    # ToDo: Filter out connectives with arguments in the same sentence
    # ToDo: Train a logistic regression classifier on all the nodes of the given sentences
    # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    # ToDo: Implement method to find nodes with maximal probability
    # ToDo: Implement tree subtraction

    # ToDo: Evaluate this method
    # ToDo: Get baseline by labeling everything after the connective as arg0, everything else as arg1
    # ToDo: Get baseline for previous sentence by labeling the full sentence as arg1.
    return clf, scores, le