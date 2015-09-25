#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Some scoring methods for our problem
http://scikit-learn.org/stable/modules/model_evaluation.html
"""

import numpy as np

from predict import predict_arg_node, label_argspan
from features import node_featurizer as node_feat
from learn import node_feature_dataframe

__author__ = 'arkadi'


def random_train_test_split(df, eval_frac=0.2):
    """ Splits a dataframe into train and test

    :param df: dataset
    :type df: pandas.DataFrame
    :param eval_size: fraction of the evaluation size
    :type eval_size: float
    :return: train and test df
    :rtype: tuple
    """
    eval_size = np.ceil(len(df) * eval_frac)
    eval_indices = np.random.choice(df.index, eval_size, replace=False)
    eval_df = df.ix[eval_indices]
    train_df = df.drop(eval_indices)
    return train_df, eval_df


def overlap_bool(ground_truth, predictions):
    """ Evaluation method
    Returns percentage of the ground truth instances that have any overlap with the predicted argument span

    :param ground_truth: An iterable with ground truth argument spans, each argument span is again an iterable
    :type ground_truth: iterable
    :param predictions: An iterable with predicted argument spans, each argument span is again an iterable
    :type predictions: iterable
    :return:
    :rtype: float
    """
    n = len(ground_truth)
    recalled = 0.0
    match = False
    assert len(predictions) == n, 'Ground Truth and predictions have to have same length!'
    for i in range(0, n):
        for gt_pos in ground_truth[i]:
            for pred_pos in predictions[i]:
                if gt_pos == pred_pos:
                    match = True
                    # Breaking the innermost loop
                    break
            if match:
                recalled += 1
                match = False
                # Breaking the middle loop
                break
    return recalled/float(n)


def overlap_f1(ground_truth, predictions):
    """ Evaluation method
    Takes the F1 micro value of argument spans,
    i.e. calculates the F1 value per pair of test and predicted argument spans and averages over the whole test set.

    :param ground_truth: An iterable with ground truth argument spans, each argument span is again an iterable
    :type ground_truth: iterable
    :param predictions: An iterable with predicted argument spans, each argument span is again an iterable
    :type predictions: iterable
    :return:
    :rtype: float
    """
    n = len(ground_truth)
    total_f1 = 0.0
    assert len(predictions) == n, 'Ground Truth and predictions have to have same length!'
    for i in range(0, n):
        gt_len = len(ground_truth[i])
        pred_len = len(predictions[i])
        tp = 0.0
        for gt_pos in ground_truth[i]:
            for pred_pos in predictions[i]:
                if gt_pos == pred_pos:
                    tp += 1
        total_f1 += 2 * tp / (gt_len + pred_len - tp)  # = 2 * tp / (tp + fp + fn)
    return total_f1/float(n)


def evaluate_argspan_prediction(eval_node_df,
                                syntax_dict,
                                logit_arg0_clf,
                                logit_arg1_clf,
                                feature_list=None,
                                node_featurizer=None,
                                label_features=None,
                                label_encoder=None,
                                binary_encoder=None):
    """ Evaluate how well the argument spans were predicted by classifiers

    Apart from the evaluation data and its synax, every input is returned by the
    zwitscher.learn.learn_main_arg_node method

    :param eval_node_df: Node data including columns 'syntax_id',
    'connective_positions', 'arg0', 'arg1'.
    :type eval_node_df: pandas.DataFrame
    :param syntax_dict: To look up the syntactic information
    :type syntax_dict: dict
    :param logit_arg0_clf: Classifier for the arg0
    :type logit_arg0_clf:
    :param logit_arg1_clf: Classifier for the arg1
    :type logit_arg1_clf:
    :param node_featurizer: Create features from the nodes and syntax trees
    :type node_featurizer:
    :param label_features:
    :type label_features:
    :param label_encoder:
    :type label_encoder:
    :param binary_encoder:
    :type binary_encoder:
    :return: results with boolean and f1 overlap
    :rtype: dict
    """

    if node_featurizer is None:
        def node_featurizer(node_df, syntax_dict, node_dict):
            return node_feature_dataframe(node_df, node_feat,
                                          syntax_dict=syntax_dict,
                                          node_dict=node_dict,
                                          feature_list=feature_list)
    # Evaluation

    eval_results = {'arg0_overlap_bool': 0.0,
                    'arg1_overlap_bool': 0.0,
                    'arg0_overlap_f1': 0.0,
                    'arg1_overlap_f1': 0.0}
    gt_arg0spans = []
    predicted_arg0spans = []
    gt_arg1spans = []
    predicted_arg1spans = []
    evaluated_connectives = list()  # tuples (syntax_id, conn_pos)
    for i in range(len(eval_node_df)):
        syntax_id = eval_node_df.ix[i, 'syntax_id']
        conn_pos = eval_node_df.ix[i, 'connective_positions']
        conn_id = (syntax_id, conn_pos)
        if conn_id not in evaluated_connectives:
            tree = syntax_dict[syntax_id]
            gt_arg0spans.append(eval_node_df.ix[i, 'arg0'])
            gt_arg1spans.append(eval_node_df.ix[i, 'arg1'])
            arg0_node = predict_arg_node(conn_pos=conn_pos, tree=tree,
                                         clf=logit_arg0_clf,
                                         featurizer=node_featurizer,
                                         label_features=label_features,
                                         label_encoder=label_encoder,
                                         binary_encoder=binary_encoder,
                                         argument=0)
            arg1_node = predict_arg_node(conn_pos=conn_pos, tree=tree,
                                         clf=logit_arg1_clf,
                                         featurizer=node_featurizer,
                                         label_features=label_features,
                                         label_encoder=label_encoder,
                                         binary_encoder=binary_encoder,
                                         argument=1)
            predicted_arg0, predicted_arg1 = label_argspan(arg0_node,
                                                           arg1_node)
            predicted_arg0spans.append(predicted_arg0)
            predicted_arg1spans.append(predicted_arg1)
            evaluated_connectives.append(conn_id)

    eval_results['arg0_overlap_bool'] = overlap_bool(gt_arg0spans,
                                                     predicted_arg0spans)
    eval_results['arg1_overlap_bool'] = overlap_bool(gt_arg1spans,
                                                     predicted_arg1spans)
    eval_results['arg0_overlap_f1'] = overlap_f1(gt_arg0spans,
                                                 predicted_arg0spans)
    eval_results['arg1_overlap_f1'] = overlap_f1(gt_arg1spans,
                                                 predicted_arg1spans)
    return eval_results
