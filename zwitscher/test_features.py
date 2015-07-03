#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module tests the feature functions
"""
from features import connective_raw, discourse_connective_text_featurizer

__author__ = 'arkadi'


def test_connective_raw():
    sents = [['Das', 'stimmt', 'zwar', '.'], ['Aber', 'egal']]
    conn_pos0 = [(0, 2), (1, 0)]
    assert connective_raw(sents, conn_pos0) == 'zwar_Aber', "Sentence boundaries are separated by '_'"
    conn_pos0 = [(0, 0), (0, 1)]
    assert connective_raw(sents, conn_pos0) == 'Das stimmt', "Phrases are separated by ' '"
    conn_pos0 = [(0, 1), (0, 0), (0, 2)]
    assert connective_raw(sents, conn_pos0) == 'Das stimmt zwar', 'Sorting works properly'
    conn_pos0 = [(0, 0), (0, 2)]
    assert connective_raw(sents, conn_pos0) == 'Das_zwar', "Discontinuous connectives are separated by '_'"

def test_discourse_connective_text_featurizer():
    sents = [['Das', 'stimmt', 'zwar', '.'], ['Aber', 'egal']]
    conn_pos = [(0, 2), (1, 0)]
    # ToDo: Add global variables for dimlex and pcc paths
    results = discourse_connective_text_featurizer(sents, conn_pos, feature_list=['connective_lexical'])
    assert results['connective_lexical'] == 'zwar_aber'

if __name__ == '__main__':
    test_discourse_connective_text_featurizer()