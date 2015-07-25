#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module tests the feature functions
"""
__author__ = 'arkadi'


def test_connective_raw():
    from features import connective_raw
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
    from features import discourse_connective_text_featurizer
    sents = [['Das', 'stimmt', 'zwar', '.'], ['Aber', 'egal']]
    conn_pos = [(0, 2), (1, 0)]
    # ToDo: Add global variables for dimlex and pcc paths
    results = discourse_connective_text_featurizer(sents, conn_pos)
    assert results['connective_lexical'] == 'zwar_aber'
    assert results['length_prev_sent'] == 4
    assert results['length_same_sent'] == 2
    assert results['length_next_sent'] == 0
    assert results['tokens_before'] == 2
    assert results['tokens_between'] == 1
    assert results['tokens_after'] == 1
    assert results['length_connective'] == 2

# ToDo: Add tests for other feature functions

if __name__ == '__main__':
    test_discourse_connective_text_featurizer()