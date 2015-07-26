#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A script to learn classification of sentence distance between connective arguments
"""
import os
import uuid
import pickle

from gold_standard import pcc_to_gold
from learn import load_data, clean_data, learn_sentdist

__author__ = 'arkadi'

# ToDo: Clickify this!
def main(feature_list=['connective_lexical', 'length_connective',
                       'length_prev_sent', 'length_same_sent', 'length_next_sent',
                       'tokens_before', 'tokens_after', 'tokens_between'],
         label_features=['connective_lexical'],
         connector_folder='/media/arkadi/arkadis_ext/NLP_data/ger_twitter/' +
                          'potsdam-commentary-corpus-2.0.0/connectors',
         pickle_folder='data',
         unpickle_gold=True,
         pickle_classifier=True
    ):


    """ The main learning function for a classifier

    Runs a random forest. Prints out accuracy scores from a 5-fold cross validation.
    Returns the classifier and the label encoder that was used.
    :param feature_list: list of features that shall be calculated with discourse_connective_text_featurizer
    :param label_features: list of features that have to be encoded as labels
    :return: trained classifier, score array and label encoder
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
    print 'Learning classifier for sentence distance between arguments of connectors.'
    print '=========================================================================='
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

    clf, scores, le = learn_sentdist(clean_pcc, feature_list=feature_list, label_features=label_features)

    if pickle_classifier:
        classifier_folder = os.path.join(pickle_folder, 'classifiers/')
        if not os.path.exists(classifier_folder):
            os.mkdir(classifier_folder)
        id = uuid.uuid4().get_hex()
        print 'Pickling classifier to %s' % os.path.join(classifier_folder, '%s_classifier_sent_dist.pickle' % str(id))
        with open(os.path.join(classifier_folder, '%s_classifier.pickle' % str(id)), 'wb') as f:
            pickle.dump(clf, f, pickle.HIGHEST_PROTOCOL)
        print 'Pickling label encoder to %s' % os.path.join(classifier_folder, '%s_encoder.pickle' % str(id))
        with open(os.path.join(classifier_folder, '%s_encoder.pickle' % str(id)), 'wb') as f:
            pickle.dump(clf, f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(classifier_folder, 'classifier.log'), 'a') as f:
            f.write('%s\t%f\t%s\n' % (id, scores.mean(), str(feature_list)))


if __name__ == '__main__':

    feature_list = ['connective_lexical', 'length_prev_sent', 'length_connective',
                    'length_same_sent', 'tokens_before', 'length_next_sent', 'prev_token', 'next_token', '2prev_token']
    # 'tokens_before', 'tokens_after', ,             'length_next_sent',]

    main(feature_list=feature_list,
         label_features=['connective_lexical', 'prev_token', 'next_token', '2prev_token'],
         unpickle_gold=True,
         pickle_classifier=True)

