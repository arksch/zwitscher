#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module includes feature methods to create features from discourse connectives in a text to classify
their argument positions
"""
import re
import sys
import os.path

import pandas as pd

from utils.dimlex import load as load_dimlex
__author__ = 'arkadi'


def discourse_connective_text_featurizer(sents, nested_connective_positions,
                                         feature_list=['connective_lexical', 'length_connective',
                                                       'length_prev_sent', 'length_same_sent', 'length_next_sent',
                                                       'tokens_before', 'tokens_after', 'tokens_between',
                                                       'prev_token', '2prev_token', 'next_token'],
                                         dimlex_path=os.path.join(os.path.dirname(__file__), 'data/dimlex.xml')):
    """ A very simple featurizer for connectives in a text to classify argument positions

    :param sents: A list of sentences. Each sentence is a list of tokens.
    :type sents: list
    :param nested_connective_positions: list of nested indices, each index is a pair (sentence_index, token_index)
    :type nested_connective_positions: list
    :param feature_list: which features are calculated. You can also pass your own function to calculate the features
    'connective_lexical': lexical entry of the connective (or normalized if not present in lexicon)
    'length_connective': number of words in the connective
    'length_prev_sent': length of previous sentence. If the connective spreads over two sentence, take the first one.
    'length_same_sent': length of the sentence that includes the connective. If it spreads over two sentences, take the
    second one
    'length_next_sent: length of the sentence after the connective ends
    'words_before': number of words inside the sentence before the sentence begins
    'words_after': number of words inside the sentence after the connective ends
    'words_between': number of words between the parts of the connective (will be 0 for continuous connectives)
    :type feature_list: list
    :return: Features:
    :rtype: pd.DataFrame
    """
    results = dict()
    for feature in feature_list:
        if isinstance(feature, basestring):
            feature_name = feature
            if feature == 'connective_lexical':
                assert dimlex_path, 'Need to specify location of dimlex'
                dimlex = load_dimlex(dimlex_path)
                # create a function that transforms a text and connective_pos into the lexicon entry
                feature_fct = lambda s, c: connective_lexical(connective_raw(s, c), lexicon=dimlex)
            elif feature == 'length_connective':
                feature_fct = lambda s, c: len(c)
            elif feature == 'length_prev_sent':
                feature_fct = lambda s, c: sent_length(s, c, direction='prev')
            elif feature == 'length_same_sent':
                feature_fct = lambda s, c: sent_length(s, c, direction='same')
            elif feature == 'length_next_sent':
                feature_fct = lambda s, c: sent_length(s, c, direction='next')
            elif feature == 'tokens_before':
                feature_fct = lambda s, c: number_of_tokens(s, c, direction='before')
            elif feature == 'tokens_after':
                feature_fct = lambda s, c: number_of_tokens(s, c, direction='after')
            elif feature == 'tokens_between':
                feature_fct = lambda s, c: number_of_tokens(s, c, direction='between')
            elif feature == 'prev_token':
                feature_fct = lambda s, c: token(s, c, direction=-1)
            elif feature == '2prev_token':
                feature_fct = lambda s, c: token(s, c, direction=-2)
            elif feature == 'next_token':
                feature_fct = lambda s, c: token(s, c, direction=1)
            else:
                raise ValueError('%s is an unknown feature' % feature)
        else:
            feature_fct = feature
            feature_name = feature.__name__
        results[feature_name] = feature_fct(sents, nested_connective_positions)
    return results


def node_featurizer(node, sent, connective_pos, tree,
                    feature_list=['connective_lexical', 'nr_of_left_C_siblings', 'nr_of_right_C_siblings',
                              'path_to_node', 'relative_pos_of_N_to_C'],
                    dimlex_path=os.path.join(os.path.dirname(__file__), 'data/dimlex.xml')):
    results = dict()
    for feature in feature_list:
        if isinstance(feature, basestring):
            feature_name = feature
            if feature == 'connective_lexical':
                assert dimlex_path, 'Need to specify location of dimlex'
                dimlex = load_dimlex(dimlex_path)
                # create a function that transforms a text and connective_pos into the lexicon entry
                feature_fct = lambda n, s, c, t: connective_lexical(connective_raw_flat(s, c), lexicon=dimlex)  # Fixme: slow!
            elif feature == 'nr_of_left_C_siblings':
                feature_fct = lambda n, s, c, t: siblings(t.terminals[c[0]], own_pos=c[0], direction='left')  # Fixme: slow! use the full c
            elif feature == 'nr_of_right_C_siblings':
                feature_fct = lambda n, s, c, t: siblings(t.terminals[c[0]], own_pos=c[0], direction='right')  # Fixme: slow! use the full c
            elif feature == 'path_to_node':
                feature_fct = lambda n, s, c, t: t.terminals[c[0]].path_to_other(n)  # Fixme: fast but seems to have odd results, use the full c
            elif feature == 'relative_pos_of_N_to_C':
                feature_fct = lambda n, s, c, t: relative_pos(n, t.terminals[c[0]])  # Fixme: ives always NaN, use the full c
            else:
                raise ValueError('%s is an unknown feature' % feature)
        else:
            feature_fct = feature
            feature_name = feature.__name__
        results[feature_name] = feature_fct(node, sent, connective_pos, tree)
    sys.stdout.write('.')
    return results

#########################################
#### All the actual feature methods #####
#########################################

def chunk_connective(nested_connective_positions):
    """ Helper to chunk a connective into its continuous parts

    :param nested_connective_positions: list of pairs of (sent_index, token_index)
    :type nested_connective_positions: list
    :return: list of chunks, each chunk is a list of pairs of (sent_index, token_index)
    :rtype: list
    """
    chunks = []
    current_chunk = []
    nested_connective_positions = sorted(nested_connective_positions)
    for i in range(0, len(nested_connective_positions)):
        # Finding all the words in the sentence and
        if not current_chunk:
            current_chunk.append(nested_connective_positions[i])
        else:
            if nested_connective_positions[i - 1][0] != nested_connective_positions[i][0]:
                # Crossed sentence boundary
                chunks.append(current_chunk)
                current_chunk = [nested_connective_positions[i]]
            else:
                if nested_connective_positions[i - 1][1] + 1 != nested_connective_positions[i][1]:
                    # Skipped words
                    chunks.append(current_chunk)
                    current_chunk = [nested_connective_positions[i]]
                else:
                    # We are continuing our connective
                    current_chunk.append(nested_connective_positions[i])
    chunks.append(current_chunk)
    return chunks


def connective_raw(sents, nested_connective_positions):
    """ Helper to calculate the raw connective from the text and the positions

    :param sents: list of sentences. Each sentence a list of tokens
    :param nested_connective_positions: list of pairs of (sent_index, token_index)
    :type nested_connective_positions: list
    :return: concatenate the strings. Discontinuous jumps are concatenated with a '_'
    """
    chunks = chunk_connective(nested_connective_positions)
    return '_'.join([' '.join([sents[sent][tok] for (sent, tok) in chunk]) for chunk in chunks])


def connective_raw_flat(sent, indices):
    """ Helper to get the raw connective from within a sentence

    :param sent:
    :type sent:
    :param indices:
    :type indices:
    :return: concatenate the strings. Discontinuous jumps are concatenated with a '_'
    :rtype: basestring
    """
    sents = [sent]
    nested_indices = [(0, i) for i in indices]
    return  connective_raw(sents, nested_indices)


def connective_lexical(connective_raw, lexicon):
    """ Feature function
    Look up canonical spelling, create own normalized version if the word doesn't exist in the lexicon

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


def sent_length(sents, nested_connective_positions, direction='prev'):
    """ Feature function
    Calculates lengths of sentences around a connective

    Use direction 'prev', 'same' or 'next'
    For connectives that spread over more than one sentence, 'prev' denotes the first sentence, 'same' the second
    :param sents:
    :type sents:
    :param nested_connective_positions:
    :type nested_connective_positions:
    :param direction: 'prev', 'same' or 'next'
    :type direction:
    :return: length of the sentence in the given direction
    :rtype: int
    """
    conn_sents, conn_pos = zip(*nested_connective_positions)  # unzipping the sentences from the nested positions
    if len(set(conn_sents)) == 1:
        # The connective spreads only over one sentence
        prev_ind = conn_sents[0] - 1
        same_ind = conn_sents[0]
    else:
        # The connective spreads over more than one sentence
        prev_ind = min(conn_sents)
        same_ind = max(conn_sents)
    next_ind = same_ind + 1

    # Returning the requested values
    if direction == 'prev':
        if prev_ind >= 0:
            return len(sents[prev_ind])
        else:
            # There is no previous sentence
            return 0
    if direction == 'same':
        return len(sents[same_ind])
    if direction == 'next':
        if next_ind < len(sents):
            return len(sents[next_ind])
        else:
            # There is no next sentence
            return 0


def number_of_tokens(sents, nested_connective_positions, direction='before'):
    """ Feature function

    Counting tokens inside sentences before/after/between connectives. 'between' is 0 for continuous connectives.
    :param sents:
    :type sents:
    :param nested_connective_positions:
    :type nested_connective_positions:
    :param direction:
    :type direction:
    :return:
    :rtype: int
    """
    nested_connective_positions = sorted(nested_connective_positions)
    if direction == 'before':
        sent, tok = nested_connective_positions[0]
        return tok
    if direction == 'after':
        sent, tok = nested_connective_positions[-1]
        return len(sents[sent]) - tok - 1
    if direction == 'between':
        chunks = chunk_connective(nested_connective_positions)
        if len(chunks) == 0:
            return None  # Missing value
        elif len(chunks) == 1:
            return 0
        elif len(chunks) == 2:
            # The left boundary of the gap is the last token of the first chunk
            # The right boundary of the gap is the first token of the last chunk
            left = chunks[0][-1]
            right = chunks[-1][0]
            if left[0] == right[0]:
                # The chunks are in the same sentence
                return right[1] - left[1] - 1
            else:
                return (number_of_tokens(sents, [left], direction='after') +  # Tokens from the left sentence
                        number_of_tokens(sents, [right], direction='before') +  # Tokens from the right sentence
                        sum([len(sents[i]) for i in range(left[0] + 1, right[0])]))  # Tokens from sentences inbetween
        else:
            return None


def token(sents, nested_connective_positions, direction=0):
    """ Feature function

    Returns the token before or after the connective as specified by the direction parameter
    If the connective is bounded by a sentence, always a '.' is returned
    :param sents:
    :type sents:
    :param nested_connective_positions:
    :type nested_connective_positions:
    :param direction:
    :type direction: int
    :return:
    :rtype: str
    """
    if direction < 0:
        sent, pos = min(nested_connective_positions)
        if pos == 0:
            return '.'
        else:
            return sents[sent][pos + direction]
    elif direction > 0:
        sent, pos = max(nested_connective_positions)
    else:
        return None
    if pos + direction < 0 or pos + direction >= len(sents[sent]):
        return '.'
    else:
        return sents[sent][pos + direction]


def siblings(terminal, own_pos=None, direction='left'):
    """ Feature function

    Counts how many siblings there are
    :param terminal: The terminal in question
    :type terminal: zwitscher.utils.tree.Node
    :param direction:
    :type direction: basestring
    :return: Number of siblings
    :rtype: int
    """
    if own_pos is None:
        own_pos = terminal.tree.terminals.index(terminal)
    siblings_positions = terminal.parent.terminal_indices()
    if direction == 'left':
        directed_sibs = [sib for sib in siblings_positions if sib < own_pos]
    elif direction == 'right':
        directed_sibs = [sib for sib in siblings_positions if sib > own_pos]
    else:
        raise ValueError('Need to call with direction "left" or "right"')
    return len(directed_sibs)


def relative_pos(node, connective_positions):
    """ Feature function

    Returns whether more descendants of the node are to the right or left
        or the terminal node is in the center
    :param node:
    :type node: zwitscher.utils.tree.Node
    :param connective_positions: positions of the connective in the sentence
    :type connective_positions: list
    :return: 'left', 'center', 'right'
    :rtype: basestring
    """
    node.terminal_indices()
    # ToDo: Finish this
