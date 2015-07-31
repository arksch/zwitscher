#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A script to learn classification of the argument spans of a connective,
whose sentence distance between arguments has already been classified
"""
import os
import uuid
import pickle

from gold_standard import pcc_to_gold
from learn import load_gold_data, clean_data, same_sentence, \
    pcc_to_arg_node_gold, learn_main_arg_node

__author__ = 'arkadi'

# ToDo: Clickify this!
def main(feature_list=['connective_lexical',
                       'nr_of_siblings',
                     #'nr_of_left_C_siblings',
                     #'nr_of_right_C_siblings',
                     'path_to_node',
                     'node_cat',
                     #'relative_pos_of_N_to_C'
                     ],
         label_features=['connective_lexical'],
         connector_folder='/media/arkadi/arkadis_ext/NLP_data/ger_twitter/' +
                          'potsdam-commentary-corpus-2.0.0/connectors',
         pickle_folder='data',
         unpickle_gold=True,
         pickle_classifier=True
         ):
    """ The main learning function for a classifier

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
    print 'Learning classifier for finding the main nodes of connectors.'
    print '============================================================='
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

    same_sentence_connectives = same_sentence(clean_pcc)
    node_gold_df, node_dict = pcc_to_arg_node_gold(same_sentence_connectives, syntax_dict)
    print '%i incorrectly parsed trees' % len([tree for tree in syntax_dict.values() if isinstance(tree, basestring)])
    print '%i incorrectly parsed nodes' % len(
        [node for node in node_dict.values() if
         isinstance(node, basestring)])
    print node_gold_df.head()
    import ipdb; ipdb.set_trace()
    print 'Cleaned data'


    clf, scores, le = learn_main_arg_node(node_gold_df, syntax_dict, node_dict,
                                          feature_list=feature_list,
                                          label_features=label_features)


if __name__ == '__main__':
    main(unpickle_gold=False)