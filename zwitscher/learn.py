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

from utils.PCC import load_connectors
from utils.tree import ConstituencyTree
from gold_standard import pcc_to_gold, label_arg_node
from features import discourse_connective_text_featurizer, node_featurizer

__author__ = 'arkadi'


def load_gold_data(connector_folder='/media/arkadi/arkadis_ext/NLP_data/ger_twitter/' +
                               'potsdam-commentary-corpus-2.0.0/connectors'):
    # Load PCC data
    pcc = load_connectors(connector_folder)
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


def featurize_sentdist_data(dataframe, feature_function):
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


def featurize_node_data(dataframe, feature_function):
    """ Creates features as specified by the featurizer that creates a feature dict from
    sentences and connective positions

    Sentences are a list of lists with tokens
    Connective positions is a list with pairs of sentence and token indices
    :param dataframe: data with columns 'sentences' and 'connective_positions'
    :type dataframe: pd.DataFrame
    :return: features
    :rtype: pd.DataFrame
    """
    def row_reduce(row):
        ser = pd.Series(feature_function(row['node_id'],
                                         row['sentence'],
                                         row['connective_positions'],
                                         row['syntax_id']))
        return ser
    features = dataframe.apply(lambda row: row_reduce(row), axis=1, reduce=False)
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
    features = featurize_sentdist_data(clean_pcc, featurizer)  # Got features of X
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


def same_sentence(clean_pcc):
    """ Filter out those connectives that have the argument in the same sentence

    :param clean_pcc:
    :type clean_pcc: pd.DataFrame
    :return:
    :rtype: pd.DataFrame
    """
    return clean_pcc[clean_pcc['sentence_dist'] == 0]


def pcc_to_arg_node_gold(same_sent_pcc, syntax_dict):
    """ Get all the nodes from the pcc dataframe

    :param same_sent_pcc:
    :type same_sent_pcc: pd.DataFrame
    :return: Dataframe with nodes as index and arg0, arg1, connective_positions,
    sentence and syntax as columns
    Also a node dict, since objects in pd.DataFrames seem to break
    :rtype: pair
    """
    data = {}
    index = 0
    node_dict = dict()
    for i in range(0, len(same_sent_pcc)):
        sent_data = {}
        conn_series = same_sent_pcc.iloc[i, :]
        conn_nested_pos = conn_series['connective_positions']
        sents = [sent for (sent, tok) in conn_nested_pos]
        if len(set(sents)) != 1:
            print 'Found %i sentences in %s' % (len(sents), str(conn_series))
        sent = sents[0]
        sentence = conn_series['sentences'][sent]
        syntax_id = conn_series['syntax_ids'][sent]
        syntax_tree = syntax_dict[syntax_id]
        insent_arg0 = [tok for (sent, tok) in conn_series['arg0']]
        insent_arg1 = [tok for (sent, tok) in conn_series['arg1']]
        if not isinstance(syntax_tree, ConstituencyTree):
            # Fixme: Apparantely there is some bug that makes a tree a string
            print 'Found string %s instead of syntax tree' % str(syntax_tree)
            continue
        label_arg_node(insent_arg0, syntax_tree, label=0)
        label_arg_node(insent_arg1, syntax_tree, label=1)
        connective_pos = [tok for sent, tok in conn_series['connective_positions']]
        syntax_dict[syntax_id] = syntax_id
        sent_data = {'sentence': sentence,
                     'arg0': insent_arg0,
                     'arg1': insent_arg1,
                     'connective_positions': connective_pos,
                     'syntax_id': syntax_id}
        for node in [node for node in syntax_tree.nodes if not node.terminal]:
        #for node in set(list(syntax_tree.iter_nodes(include_terminals=False))):
            # Don't have node as a index, since it is easier to access with int
            node_data = {}
            node_id = node.id_str
            node_dict[node_id] = node
            node_data.update(sent_data)
            node_data['node_id'] = node_id
            node_data['is_arg0_node'] = node.arg0
            node_data['is_arg1_node'] = node.arg1
            data[node_id] = node_data
            index += 1
    return pd.DataFrame().from_dict(data, orient='index'), node_dict


def learn_main_arg_node(node_df,
                        syntax_dict,
                        node_dict,
                          feature_list=['connective_lexical',
                                        'nr_of_siblings',
                                        #'nr_of_left_C_siblings',
                                        #'nr_of_right_C_siblings',
                                        'node_cat',
                                        'path_to_node',
                                        #'relative_pos_of_N_to_C'
                                        ],
                          internal_argument=True,
                          label_features=['connective_lexical', 'path_to_node', 'relative_pos_of_N_to_C']):
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
    def featurizer(node, sent, connective_pos, tree):
        return node_featurizer(node, sent, connective_pos, tree,
                               syntax_dict=syntax_dict,
                               node_dict=node_dict,
                               feature_list=feature_list)

    print 'Calculating features'
    import ipdb; ipdb.set_trace()
    features = featurize_node_data(node_df, featurizer)  # Got features of X
    print 'done'
    # ToDo: Design features (see Lin et al p. 17, Connective_syntactic!)
    # ToDo: Chose the correct nodes (by some heuristic: first chose the node that maximizes the overlap between prediction and true)
    # ToDo: Train a logistic regression classifier on all the nodes of the given sentences

    # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    # ToDo: Implement method to find nodes with maximal probability

    # ToDo: Evaluate this method (remember not to count punctuation)
    # ToDo: Get baseline by labeling everything after the connective as
    # arg0, everything else as arg1
    # ToDo: Get baseline for previous sentence by labeling the full sentence
    #  as arg1.

    # ToDo: Try to use a random forest classifier as well

    # ToDo: Implement tree subtraction (for this need to refine the finding of the correct nodes)
    # ToDo: Or find another method that trains with a scoring method. Write evaluation method on node base (given the optimal other argument, if we wrap over it its fine, else penalize)


    return clf, scores, le

if __name__ == '__main__':
    learn_main_arg_node()