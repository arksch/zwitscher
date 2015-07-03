#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module includes feature methods to create features from discourse connectives in a text to classify
their argument positions
"""
import re
import string

import numpy as np
import pandas as pd

from utils.dimlex import load as load_dimlex
__author__ = 'arkadi'


def discourse_connective_text_featurizer(sents, connective_positions,
                                         feature_list=['connective_lexical', 'length_connective',
                                                       'length_prev_sent', 'length_same_sent', 'length_next_sent',
                                                       'words_before', 'words_after', 'words_between'],
                                         dimlex_path='data/dimlex.xml'):
    """ A very simple featurizer for connectives in a text to classify argument positions

    :param sents: A list of sentences. Each sentence is a list of tokens.
    :type sents: list
    :param connective_positions: list of indices, each index is a pair (sentence_index, token_index)
    :type connective_positions: list
    :param feature_list: which features are calculated. You can also pass your own function to calculate the features
    'connective_lexical'
    'length_connective'
    'length_prev_sent'
    'length_same_sent'
    'length_next_sent
    'words_before'
    'words_after'
    'words_between'
    :type feature_list: list
    :return: Features:
    :rtype: pd.DataFrame
    """
    results = dict()
    for feature in feature_list:
        if isinstance(feature, basestring):
            if feature == 'connective_lexical':
                assert dimlex_path, 'Need to specify location of dimlex'
                dimlex = load_dimlex(dimlex_path)
                # create a function that transforms a text and connective_pos into the lexicon entry
                feature_fct = lambda s, c: connective_lexical(connective_raw(s, c), lexicon=dimlex)
                feature_name = feature
        results[feature_name] = feature_fct(sents, connective_positions)
    return results

def connective_raw(text, connective_positions):
    """ Helper to calculate the raw connective from the text and the positions

    :param text: list of sentences. Each sentence a list of tokens
    :param connective_positions: list of pairs of (sent_index, token_index)
    :type connective_positions: list
    :return: concatenate the strings. Discontinuous jumps are concatenated with a '_'
    """
    chunks = []
    current_chunk = []
    connective_positions = sorted(connective_positions)
    for i in range(0, len(connective_positions)):
        # Finding all the words in the sentence and
        if not current_chunk:
            current_chunk.append(connective_positions[i])
        else:
            if connective_positions[i - 1][0] != connective_positions[i][0]:
                # Crossed sentence boundary
                chunks.append(current_chunk)
                current_chunk = [connective_positions[i]]
            else:
                if connective_positions[i - 1][1] + 1 != connective_positions[i][1]:
                    # Skipped words
                    chunks.append(current_chunk)
                    current_chunk = [connective_positions[i]]
                else:
                    # We are continuing our connective
                    current_chunk.append(connective_positions[i])
    chunks.append(current_chunk)
    return '_'.join([' '.join([text[sent][tok] for (sent, tok) in chunk]) for chunk in chunks])


def connective_lexical(connective_raw, lexicon):
    """ Look up canonical spelling, create own normalized version if the word doesn't exist in the lexicon

    :param connective_raw: string of the connective. Discontinuous parts are separated with '_'
    :type connective_raw: basestring
    :return: lexical version of the raw string
    :rtype: basestring
    """
    if connective_raw in lexicon.orthography_variants:
        return lexicon.orthography_variants[connective_raw]
    # 'Außer ,  wenn ' --> 'außer , wenn'
    lower_conn_raw = re.sub(' +', ' ', connective_raw.strip().lower())
    if lower_conn_raw in lexicon.orthography_variants:
        return lexicon.orthography_variants[lower_conn_raw]
    # 'außer , wenn' --> 'außer wenn'
    normalized_conn_raw = ' '.join(re.findall('\w+', lower_conn_raw))
    if normalized_conn_raw in lexicon.orthography_variants:
        return lexicon.orthography_variants[normalized_conn_raw]
    else:
        return normalized_conn_raw

