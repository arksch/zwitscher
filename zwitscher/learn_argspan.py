#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A script to learn classification of the argument spans of a connective,
whose sentence distance between arguments has already been classified
"""
import os
import uuid
import pickle

import pandas as pd

from gold_standard import pcc_to_gold
from learn import learn_main_arg_node
from gold_standard import load_gold_data, clean_data, \
    pcc_to_arg_node_gold, same_sentence
from evaluation import evaluate_argspan_prediction, random_train_test_split

__author__ = 'arkadi'

# ToDo: Clickify this!
def main(feature_list=['connective_lexical',
                       'nr_of_siblings',
                       'path_to_node',
                       'node_cat'],
         label_features=['connective_lexical', 'node_cat', 'path_to_node'],
         connector_folder='/media/arkadi/arkadis_ext/NLP_data/ger_twitter/' +
                          'potsdam-commentary-corpus-2.0.0/connectors',
         pickle_folder='data',
         unpickle_features=True,
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
    if pickle_classifier or unpickle_gold or unpickle_features:
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

    ### Splitting the data connective wise and creating node data after
    ### the split
    same_sentence_connectives = same_sentence(clean_pcc)  # a pd.DataFrame)
    train_df, test_df = random_train_test_split(same_sentence_connectives)
    train_node_df, train_node_dict = pcc_to_arg_node_gold(train_df, syntax_dict)
    test_node_df, test_node_dict = pcc_to_arg_node_gold(test_df, syntax_dict)

    print 'Cleaned data'

    features = None
    if unpickle_features:
        hdf_path = os.path.join(pickle_folder, 'features.h5')
        features = pd.read_hdf(hdf_path, 'argspan')


    classification_dict = learn_main_arg_node(train_node_df,
                                              syntax_dict,
                                              train_node_dict,
                                              precalc_features=features,
                                              feature_list=feature_list,
                                              label_features=label_features)

    arg0_clf = classification_dict['logit_arg0_clf']
    arg1_clf = classification_dict['logit_arg1_clf']
    feature_list = classification_dict['feature_list']
    label_features = classification_dict['label_features']
    le = classification_dict['label_encoder']
    ohe = classification_dict['binary_encoder']
    node_featurizer = classification_dict['node_featurizer']
    eval_results = evaluate_argspan_prediction(eval_node_df=test_node_df,
                                               syntax_dict=syntax_dict,
                                               logit_arg0_clf=arg0_clf,
                                               logit_arg1_clf=arg1_clf,
                                               feature_list=feature_list,
                                               node_featurizer=node_featurizer,
                                               label_features=label_features,
                                               label_encoder=le,
                                               binary_encoder=ohe)
    scores = 'arg0-bool: %f, arg1-bool: %f, arg0-f1: %f, arg1-f1: %f' %\
             (eval_results['arg0_overlap_bool'], eval_results['arg1_overlap_bool'],
              eval_results['arg0_overlap_f1'], eval_results['arg1_overlap_f1'])
    print 'Scores: %s' % scores

    print 'Learning classification on full data set...'
    node_df, node_dict = pcc_to_arg_node_gold(same_sentence_connectives, syntax_dict)
    classification_dict = learn_main_arg_node(node_df,
                                              syntax_dict,
                                              node_dict,
                                              precalc_features=features,
                                              feature_list=feature_list,
                                              label_features=label_features)
    print '...done'

    if pickle_classifier:
        classification_dict.pop('node_featurizer')  # Cannot pickle a function
        classifier_folder = os.path.join(pickle_folder, 'classifiers/')
        if not os.path.exists(classifier_folder):
            os.mkdir(classifier_folder)
        id_ = uuid.uuid4().get_hex()
        classification_path = os.path.join(classifier_folder, '%s_argspan_classification_dict.pickle' % str(id_))
        print 'Pickling arg span classification data to %s' % classification_path
        import ipdb; ipdb.set_trace()
        with open(classification_path, 'wb') as f:
            pickle.dump(classification_dict, f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(classifier_folder, 'classifier.log'), 'a') as f:
            f.write('arg_span\t%s\t%s\t%s\n' % (
                id_, scores, str(feature_list)))


if __name__ == '__main__':
    main(unpickle_gold=True, unpickle_features=False)