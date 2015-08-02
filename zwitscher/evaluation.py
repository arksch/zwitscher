#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Some scoring methods for our problem
http://scikit-learn.org/stable/modules/model_evaluation.html
"""
from sklearn.metrics import make_scorer

from predict_argspan import label_argspan, predict_arg_node
__author__ = 'arkadi'



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
            #import ipdb; ipdb.set_trace()
            for pred_pos in predictions[i]:
                if gt_pos == pred_pos:
                    tp += 1
        total_f1 += 2 * tp / (gt_len + pred_len - tp)  # = 2 * tp / (tp + fp + fn)
    return total_f1/float(n)


overlap_bool_scorer = make_scorer(overlap_bool, greater_is_better=True)
overlap_f1_scorer = make_scorer(overlap_f1, greater_is_better=True)


