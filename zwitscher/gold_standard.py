#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module has helper functions to create a gold standard
"""

__author__ = 'arkadi'

import pandas as pd

from analysis import analyse_argument_positions

def pcc_to_gold(pcc):
    """ Helper to transform the PCC into a pd.DataFrame

    The data are the 'sentences' and 'connective_positions'
    The targets are the complete argument positions in the columns 'arg0', 'arg1' and the denoted 'relation'
    :param pcc: list of Discourse objects
    :type pcc: list
    :return: sentences with the discourse positions and arg0, arg1 as target
    :rtype: pd.DataFrame
    """
    data = []
    for disc in pcc:
        # Gold standard stores sentence boundaries and tokens in one list of lists
        sents = [disc.tokens[sent[0]: sent[1]] for sent in sorted(disc.sentences)]
        sent_to_index = dict([(i, sent) for sent, i in enumerate(disc.sentences)])
        for conn in disc.connectives:
            # This is just the index in the whole text. We want a pair of sent,token indices
            nested_positions = flat_index_to_nested(disc.sentences, conn.positions, sent_to_index_dict=sent_to_index)
            nested_arg0 = flat_index_to_nested(disc.sentences, conn.arg0, sent_to_index_dict=sent_to_index)
            nested_arg1 = flat_index_to_nested(disc.sentences, conn.arg1, sent_to_index_dict=sent_to_index)
            # For the middle step we try to predict the sentences that include the arguments
            token_dist, sent_dist, involved_sents = analyse_argument_positions(disc.sentences, conn)
            data.append({'sentences': sents, 'syntax': disc.syntax, 'connective_positions': nested_positions,
                         'sentence_dist': sent_dist, 'arg0': nested_arg0, 'arg1': nested_arg1,
                         'relation': conn.relation})
    return pd.DataFrame(data=data)


def flat_index_to_nested(flat_sents, flat_indices, sent_to_index_dict=None):
    """ Helper to create a nested index list from a list of flat indices and sentence boundaries

    [(0, 3), (3, 6)], [2,5] --> [(0, 2), (1, 2)]
    :param flat_sents: list of pairs (start, end), where end is non-inclusive
    :type flat_sents: list
    :param flat_indices: list of indices (integers)
    :type flat_indices: list
    :param sent_to_index_dict: for a faster look up this should be precalculated
    :type sent_to_index_dict: dict
    :return: nested index
    :rtype: list
    """
    if not flat_indices:
        return None
    if sent_to_index_dict is None:
        sent_to_index_dict = dict([(i, sent) for sent, i in enumerate(flat_sents)])
    nested_indices = []
    candidate_sents = [sent for sent in flat_sents if
                       min(flat_indices) < sent[1] and sent[0] <= max(flat_indices)]
    for pos in flat_indices:  # This is just the index in the whole text. We want a pair of sent,token indices
        corresp_sent = [sent for sent in candidate_sents if sent[0] <= pos < sent[1]][0]
        nested_indices.append((sent_to_index_dict[corresp_sent], pos - corresp_sent[0]))
    return nested_indices
