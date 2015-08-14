#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module includes methods to predict the argument span given a classifier
"""
import pandas as pd

from gold_standard import subset
from features import discourse_connective_text_featurizer, node_featurizer
from learn import sentdist_feature_dataframe, node_feature_dataframe
from learn import encode_label_features, binarize_features

__author__ = 'arkadi'


def predict_sentence_dist(clean_pcc,
                          sent_dist_clf,
                          feature_list,
                          label_features,
                          label_encoder):
    featurizer = lambda sents, conn_pos: discourse_connective_text_featurizer(sents, conn_pos, feature_list=feature_list)

    features = sentdist_feature_dataframe(clean_pcc, featurizer)

    features = encode_label_features(features, label_encoder, label_features)

    sent_dists = sent_dist_clf.predict(features)
    return sent_dists


def tree_to_node_df(connective_positions, tree):
    """ Dataframe from a tree and a connective

    :param tree: Dependency Tree
    :type tree: zwitscher.utils.tree.DependencyTree
    :param connective_positions: non-nested positions of the connective
    :type connective_positions: list
    :return: Dataframe and a node_id dictionary
    :rtype: tuple
    """
    data = dict()
    node_dict = dict()
    sentence = [ter.word for ter in tree.terminals]
    sent_data = {'sentence': sentence,
                 'connective_positions': connective_positions,
                 'syntax_id': tree.id_str}
    for node in [node for node in tree.nodes if not node.terminal]:
        # for node in set(list(syntax_tree.iter_nodes(
        # include_terminals=False))):
        # Don't have node as a index, since it is easier to access with int
        node_data = {}
        node_id = node.id_str
        node_dict[node_id] = node
        node_data.update(sent_data)
        node_data['node_id'] = node_id
        node_data['is_arg0_node'] = node.arg0
        node_data['is_arg1_node'] = node.arg1
        data[node_id] = node_data
    return pd.DataFrame().from_dict(data, orient='index'), node_dict


def predict_arg_node(conn_pos, tree, clf, featurizer,
                     label_features=None,
                     label_encoder=None,
                     binary_encoder=None,
                     argument=0):
    """ Predicting the most probable node to be the argument node

    :param conn_pos: Positions of the connective (non nested)
    :type conn_pos: list
    :param tree: syntax data
    :type tree: zwitscher.utils.tree.ConstituencyTree
    :param clf: trained classifier
    :type clf: sklearn classifier
    :param featurizer: method to featurize with featurizer(node, conn_pos, tree)
    :type featurizer: function
    :param label_features: which features are label features
    :type label_features: list
    :param label_encoder: how to encode the labeled features, so that the
    classifier can read them
    :type label_encoder: LabelEncoder
    :param binary_encoder: how to encode features into binary features, so that
    the classifier can read them
    :type binary_encoder: OneHotEncoder
    :param argument: Is this an argument 1 or argument 0
    :type argument: int
    :return: Most probable node to be the argument node
    :rtype: zwitscher.utils.tree.Node
    """
    arg_str = 'arg%i_proba' % argument
    node_df, node_dict = tree_to_node_df(conn_pos, tree)
    syntax_dict = {tree.id_str: tree}
    features = featurizer(node_df, syntax_dict, node_dict)
    if label_encoder is not None:
        # We need to encode the non-numerical labels
        features = encode_label_features(features, label_encoder, label_features)
    if binary_encoder is not None:
        # We need to binarize the data for logistic regression
        features = binarize_features(features, binary_encoder, label_features)
    # Zeroth column of probabilities is False, first is true
    node_df[arg_str] = clf.predict_proba(features)[:, 1]
    node_id = node_df[arg_str].argmax()
    return node_dict[node_id]


def label_argspan(node0, node1, tree_subtraction=True):
    """ Label argspan when the nodes are known

    :param node0: node of argument 0
    :type node0: zwitscher.utils.tree.Node
    :param node1: node of argument 1
    :type node1:  zwitscher.utils.tree.Node
    :param tree_subtraction: use tree subtraction or not
    :type tree_subtraction: bool
    :return: argument spans 0 and 1 (internal, external)
    :rtype: tuple
    """
    # ToDo: Might include that connective is always part of arg0
    argspan0 = node0.terminal_indices()
    argspan1 = node1.terminal_indices()
    if tree_subtraction:
        if subset(argspan0, argspan1):
            argspan1 = [i for i in argspan1 if i not in argspan0]
        if subset(argspan1, argspan0):
            argspan0 = [i for i in argspan1 if i not in argspan1]
    if not argspan0:
        argspan0 = node0.terminal_indices()
    if not argspan1:
        all_indices = node0.tree.root.terminal_indices()
        argspan1 = [i for i in all_indices if i not in argspan0]
    return sorted(argspan0), sorted(argspan1)


def predict_argspans(node_df, syntax_dict, logit_arg0_clf, logit_arg1_clf, feature_list,
                     label_features, label_encoder, binary_encoder):
    def featurizer(node_df, syntax_dict, node_dict):
        return node_feature_dataframe(node_df, node_featurizer,
                                      syntax_dict=syntax_dict,
                                      node_dict=node_dict,
                                      feature_list=feature_list)

    predicted_arg0spans = {}
    predicted_arg1spans = {}
    evaluated_connectives = list()
    for i in range(len(node_df)):
        syntax_id = node_df.ix[i, 'syntax_id']
        conn_pos = node_df.ix[i, 'connective_positions']
        conn_id = (syntax_id, tuple(conn_pos))
        if conn_id not in evaluated_connectives:
            tree = syntax_dict[syntax_id]
            arg0_node = predict_arg_node(conn_pos=conn_pos, tree=tree,
                                         clf=logit_arg0_clf,
                                         featurizer=featurizer,
                                         label_features=label_features,
                                         label_encoder=label_encoder,
                                         binary_encoder=binary_encoder,
                                         argument=0)
            arg1_node = predict_arg_node(conn_pos=conn_pos, tree=tree,
                                         clf=logit_arg1_clf,
                                         featurizer=featurizer,
                                         label_features=label_features,
                                         label_encoder=label_encoder,
                                         binary_encoder=binary_encoder,
                                         argument=1)
            predicted_arg0, predicted_arg1 = label_argspan(arg0_node,
                                                           arg1_node)
            predicted_arg0spans[conn_id] = predicted_arg0
            predicted_arg1spans[conn_id] = predicted_arg1
            evaluated_connectives.append(conn_id)
    return predicted_arg0spans, predicted_arg1spans