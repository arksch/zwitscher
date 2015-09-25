#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A script to learn classification of sentence distance between connective arguments
"""
import os
import uuid
import pickle

import click

from gold_standard import pcc_to_gold
from learn import learn_sentdist
from gold_standard import load_gold_data, clean_data

__author__ = 'arkadi'


@click.command(help='Learning and storing classifiers for sentence distances between arguments'
                    ' of discourse connectives.\n'
                    'The output file can be passed used by the pipeline with\n'
                    'python pipeline.py -as 123uuid_sentdist_classification_dict.pickle')
@click.option('--feature_list', '-f',
              help='A comma separated list of features. By default: '
                   'connective_lexical,length_prev_sent,length_connective,length_same_sent,'
                   'tokens_before,length_next_sent,prev_token,next_token,2prev_token',
              default=None)
@click.option('--label_features', '-lf',
              help='A comma separated list of features that are labels. By default: '
                   'connective_lexical,prev_token,next_token,2prev_token',
              default=None)
@click.option('--connector_folder', '-cf',
              help='The folder to find the connectors from the potsdam commentary corpus'
                   'Can be left empty if unpickle_gold is True',
              default='/media/arkadi/arkadis_ext/NLP_data/ger_twitter/'
                      'potsdam-commentary-corpus-2.0.0/connectors')
@click.option('--pickle_folder', '-pf',
              help='The folder for all the pickles',
              default='data')
@click.option('--unpickle_gold', '-ug',
              is_flag=True,
              help='Unpickle gold connector data. Useful for dev',
              default=False)
@click.option('--pickle_classifier', '-pc',
              is_flag=True,
              help='Pickle the classifier for later use, e.g. in the pipeline.'
                   'Note that this adds an uuid to the output, so it doesnt overwrite former output'
                   'Otherwise this script will just print the evaluation',
              default=True)
def main(feature_list=None,
         label_features=None,
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
    if feature_list is None:
        feature_list = ['connective_lexical', 'length_prev_sent', 'length_connective',
                        'length_same_sent', 'tokens_before', 'length_next_sent', 'prev_token',
                        'next_token', '2prev_token']
    else:
        feature_list = feature_list.split(',')
    if label_features is None:
        label_features = ['connective_lexical', 'prev_token', 'next_token', '2prev_token']
    else:
        label_features = feature_list.split(',')

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
                pcc_df, syntax_dict = pcc_to_gold(pickle.load(f))
        else:
            print 'Could not find file %s' % os.path.join(pickle_folder, 'PCC_disc.pickle')
            return
    else:
        print 'Parsing gold data from %s' % connector_folder
        pcc_df, syntax_dict = load_gold_data(connector_folder)  # Loaded X and y into one dataframe
    print 'Loaded data'

    print 'Cleaning data...'
    clean_pcc = clean_data(pcc_df)
    print 'Cleaned data'

    clf, scores, le = learn_sentdist(clean_pcc, feature_list=feature_list, label_features=label_features)
    classification_dict = {'sent_dist_classifier': clf,
                           'feature_list': feature_list,
                           'label_features': label_features,
                           'label_encoder': le}
    if pickle_classifier:
        classifier_folder = os.path.join(pickle_folder, 'classifiers/')
        if not os.path.exists(classifier_folder):
            os.mkdir(classifier_folder)
        id_ = uuid.uuid4().get_hex()
        classifier_path = os.path.join(classifier_folder, '%s_sent_dist_classification_dict.pickle' % str(id_))
        print 'Pickling sent_dist classifier to %s' % classifier_path
        with open(classifier_path, 'wb') as f:
            pickle.dump(classification_dict, f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(classifier_folder, 'classifier.log'), 'a') as f:
            f.write('sent_dist\t%s\t%f\t%s\n' % (id_, scores.mean(), str(feature_list)))


if __name__ == '__main__':
    main()

