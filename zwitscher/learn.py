#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Methods needed for learning classifiers
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from features import discourse_connective_text_featurizer, node_featurizer



__author__ = 'arkadi'


def sentdist_feature_dataframe(dataframe, feature_function):
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


def node_feature_dataframe(dataframe,
                        feature_function,
                        syntax_dict=None,
                        node_dict=None,
                        feature_list=None
                        ):
    """ Creates features as specified by the featurizer that creates a feature dict from
    sentences and connective positions

    Sentences are a list of lists with tokens
    Connective positions is a list with pairs of sentence and token indices
    :param dataframe: data with columns 'sentences' and 'connective_positions'
    :type dataframe: pd.DataFrame
    :param syntax_dict:
    :type syntax_dict: dict
    :param node_dict:
    :type node_dict: dict
    :return: features
    :rtype: pd.DataFrame
    """
    if syntax_dict is None or node_dict is None:
        raise ValueError('Need syntax and node dict to look up values')
    # Defining a helper method
    def row_reduce(row):
        ser = pd.Series(feature_function(node_dict[row['node_id']],
                                         row['connective_positions'],
                                         syntax_dict[row['syntax_id']],
                                         feature_list=feature_list))
        return ser
    # Applying the helper to every row in the dataframe
    features = dataframe.apply(lambda row: row_reduce(row), axis=1, reduce=False)
    return features


def encode_label_features(features, le, label_features):
    """ Helper to encode non numerical labels

    Maps previously unseen labels to '<unknown>', note that after fitting you
    have to call
    le.classes_ = np.append(le.classes_, '<unknown>')
    :param features: Features data
    :type features: pd.DataFrame
    :param le: encoder
    :type le: LabelEncoder
    :param label_features: Names of features that are labels
    :type label_features: list
    :return: encoded feature data
    :rtype: pd.DataFrame

    :Example:
    le = LabelEncoder()
    le.fit(features[label_features].values.ravel())
    le.classes_ = np.append(le.classes_, '<unknown>')
    encoded_features = encode_label_features(features, le, label_features)
    """
    for feat in label_features:
        features[feat] = features[feat].map(lambda s: '<unknown>' if s not in le.classes_ else s)
        features[feat] = le.transform(features[feat])
    return features


def binarize_features(encoded_features, ohe, label_features):
    """ Helper to binarize features

    :param encoded_features: Features data already encoded into floats
    :type encoded_features: pd.DataFrame
    :param ohe: encoder
    :type ohe: OneHotEncoder
    :param label_features: Names of features that are labels
    :type label_features: list
    :return: binarized feature data
    :rtype: pd.DataFrame
    """
    binarized_features = ohe.transform(encoded_features[label_features].values)
    feature_list = encoded_features.columns
    cont_features = [feat for feat in feature_list if
                     feat not in label_features]
    logit_features = np.concatenate([encoded_features[cont_features].values, binarized_features], axis=1)
    return logit_features


def learn_sentdist(clean_pcc,
                   feature_list=None,
                   label_features=None):
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
    features = sentdist_feature_dataframe(clean_pcc, featurizer)  # Got features of X
    print 'Calculated all features'

    # We need to encode the non-numerical labels
    le = LabelEncoder()
    # LabelEncoder only deals with 1 dim np.arrays
    le.fit(features[label_features].values.ravel())
    # Dealing with unknowns
    le.classes_ = np.append(le.classes_, '<unknown>')
    features = encode_label_features(features, le, label_features)

    print 'Cross validating classifier...'
    clf = RandomForestClassifier(min_samples_leaf=5, n_jobs=-1, verbose=0)
    scores = cross_val_score(clf, features, clean_pcc['sentence_dist'], cv=5)
    print 'Cross validated classifier\nscores: %s\nmean score: %f' % (str(scores), scores.mean())

    print 'Learning classifier on the whole data set...'
    clf.fit(features, clean_pcc['sentence_dist'])
    print 'Learned classifier on the whole data set'

    return clf, scores, le


def learn_main_arg_node(node_df,
                        syntax_dict,
                        node_dict,
                        precalc_features=None,
                          feature_list=None,
                          label_features=None):
    """ Learn a classifier for a node being arg0 or arg1

    :param node_df: node data with tree and node ids
    :type node_df: pd.DataFrame
    :param syntax_dict: to look up the syntax trees by their id
    :type syntax_dict:
    :param node_dict: to look up the nodes by their id
    :type node_dict:
    :param feature_list:
    :type feature_list:
    :param internal_argument:
    :type internal_argument:
    :param label_features:
    :type label_features:
    :return:
    :rtype:
    """

    def featurizer(node_df, syntax_dict, node_dict):
        return node_feature_dataframe(node_df, node_featurizer,
                                      syntax_dict=syntax_dict,
                                      node_dict=node_dict,
                                      feature_list=feature_list)

    if precalc_features is None:
        print 'Calculating features'
        features = featurizer(node_df, syntax_dict, node_dict)
        print 'done'
    else:
        features = precalc_features

    # We need to encode the non-numerical labels
    print 'Encoding labels...'
    le = LabelEncoder()
    # LabelEncoder only deals with 1 dim np.arrays
    le.fit(features[label_features].values.ravel())
    # Dealing with unknowns
    le.classes_ = np.append(le.classes_, '<unknown>')
    encoded_features = encode_label_features(features, le, label_features)
    print 'Encoded label'
    # We need to binarize the data for logistic regression
    print 'Binarizing features for logistic regression...'
    ohe = OneHotEncoder(sparse=False)
    ohe.fit(encoded_features[label_features].values)
    logit_features = binarize_features(encoded_features, ohe, label_features)
    print 'Binarized features.'

    print 'Training classifiers for arg0 labeling'
    print '======================================'
    nr_of_nodes = float(len(node_df))
    baseline = (nr_of_nodes - sum(node_df['is_arg0_node'])) / nr_of_nodes
    print 'Majority baseline: %f' % baseline
    print 'Cross validating Logistic regression classifier...'
    # C is the inverse of the regularization strength
    # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    logit_arg0_clf = LogisticRegression(C=1.0)
    scores = cross_val_score(logit_arg0_clf, logit_features,
                             node_df['is_arg0_node'], cv=5)
    print 'Cross validated Logistic Regression classifier\nscores: %s\nmean score: ' \
          '%f' % (str(scores), scores.mean())

    print ''
    print 'Training classifiers for arg1 labeling'
    print '======================================'
    baseline = (nr_of_nodes - sum(node_df['is_arg1_node'])) / nr_of_nodes
    print 'Majority baseline: %f' % baseline
    print 'Cross validating Logistic regression classifier...'
    # C is the inverse of the regularization strength
    logit_arg1_clf = LogisticRegression(C=1.0)
    scores = cross_val_score(logit_arg1_clf, logit_features,
                             node_df['is_arg1_node'], cv=5)
    print 'Cross validated Logistic Regression classifier\nscores: %s\nmean score: ' \
          '%f' % (
              str(scores), scores.mean())

    print 'Learning classifiers on the whole data set...'
    logit_arg0_clf.fit(logit_features, node_df['is_arg0_node'])
    logit_arg1_clf.fit(logit_features, node_df['is_arg1_node'])
    print 'Learned classifier on the whole data set'


    # ToDo: Design features (see Lin et al p. 17, Connective_syntactic!)

    # ToDo: Evaluate this method (remember not to count punctuation)
    # ToDo: Get baseline by labeling everything after the connective as
    # arg0, everything else as arg1
    # ToDo: Get baseline for previous sentence by labeling the full sentence
    #  as arg1.

    return_dict = {'logit_arg0_clf': logit_arg0_clf,
                   'logit_arg1_clf': logit_arg1_clf,
                   'feature_list': feature_list,
                   'label_features': label_features,
                   'label_encoder': le,
                   'binary_encoder': ohe,
                   'node_featurizer': featurizer}
    return return_dict

