#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module includes methods to predict the argument span given a classifier
"""
import pandas as pd

from gold_standard import subset
from learn import encode_label_features, binarize_features

__author__ = 'arkadi'




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
    import ipdb; ipdb.set_trace()
    features = featurizer(node_df)
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
    return argspan0, argspan1


# ToDo: Write a pipeline to predict given only syntax trees