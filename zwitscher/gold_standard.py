#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module has helper functions to create a gold standard
"""
import numpy as np
from zwitscher.utils.PCC import load_connectors

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
    syntax_dict = dict()
    for disc in pcc:
        # Gold standard stores sentence boundaries and tokens in one list of lists
        sents = [disc.tokens[sent[0]: sent[1]] for sent in sorted(disc.sentences)]
        sent_to_index = dict([(i, sent) for sent, i in enumerate(disc.sentences)])
        syntax_ids = []
        for tree in disc.syntax:
            syntax_dict[tree.id_str] = tree
            syntax_ids.append(tree.id_str)  # Keep it sorted!
        for conn in disc.connectives:
            # This is just the index in the whole text. We want a pair of sent,token indices
            nested_positions = flat_index_to_nested(disc.sentences, conn.positions, sent_to_index_dict=sent_to_index)
            nested_arg0 = flat_index_to_nested(disc.sentences, conn.arg0, sent_to_index_dict=sent_to_index)
            nested_arg1 = flat_index_to_nested(disc.sentences, conn.arg1, sent_to_index_dict=sent_to_index)
            # For the middle step we try to predict the sentences that include the arguments
            token_dist, sent_dist, involved_sents = analyse_argument_positions(disc.sentences, conn)
            data.append({'sentences': sents, 'syntax_ids': syntax_ids, 'connective_positions': nested_positions,
                         'sentence_dist': sent_dist, 'arg0': nested_arg0, 'arg1': nested_arg1,
                         'relation': conn.relation})
    return pd.DataFrame(data=data), syntax_dict


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


def subset(list1, list2):
    """ Helper to tell whether list1 is a subset of list2
    """
    for val in list1:
        if val not in list2:
            return False
    return True

def label_arg_node(arg_pos, tree, label=0):
    """ Helper to label the argument node

    the node that is probably the closest ancestor of the full argument
    will get a label
    :param arg_pos: flat indices of the argument terminals in the sentence
    :type arg_pos: list
    :param label: The label to put to the node
    0 will set node.arg0 = True, 1 will set node.arg1 = True
    :param tree:
    :type tree: zwitscher.utils.tree.ConstituencyTree
    """
    node = tree.terminals[arg_pos[0]].parent
    if node is None:
        # First part of the argument was probably punctuation
        if len(arg_pos) == 1:
            print 'Only punctuation in the argument of this tree'
            print arg_pos
            print str(tree)
            node = tree.root
        else:
            # Maybe the latter parts of the argument can still be labeled
            print 'Removed punctuation from argument'
            label_arg_node(arg_pos[1:], tree, label=label)
            return
    while subset(arg_pos, node.terminal_indices()):
        node = node.parent
        if node is None:
            # Something went wrong, backoff to full sentence
            node = tree.root
            break
    if label == 0:
        node.arg0 = True
    elif label == 1:
        node.arg1 = True
    node.label = label


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
        label_arg_node(insent_arg0, syntax_tree, label=0)
        label_arg_node(insent_arg1, syntax_tree, label=1)
        connective_pos = [tok for sent, tok in conn_series['connective_positions']]
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
    return pd.DataFrame().from_dict(data, orient='index'), node_dict


def same_sentence(clean_pcc):
    """ Filter out those connectives that have the argument in the same sentence

    :param clean_pcc:
    :type clean_pcc: pd.DataFrame
    :return:
    :rtype: pd.DataFrame
    """
    return clean_pcc[clean_pcc['sentence_dist'] == 0]