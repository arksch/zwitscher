#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module includes methods to analyse connectives and their use
"""
from __future__ import print_function
import os
import pickle
import sys
import re
import string

from utils.PCC import load as load_pcc
from utils.dimlex import load as load_dimlex

__author__ = 'arkadi'

def analyse_argument_positions(sentences, connective):
    """ Analyze the argument positions of a connective in a sequence of sentences

    :param sentences: List of integer pairs for start and end
    :type sentences: list
    :type connective: zwitscher.utils.connectives.DiscConnective
    :return: token distance, sentence distance and involved sentences of arguments
    :rtype: tuple
    """
    if connective.arg0 and connective.arg1:
        min0 = min(connective.arg0)
        max0 = max(connective.arg0)
        min1 = min(connective.arg1)
        max1 = max(connective.arg1)
    else:
        raise ValueError('No arguments set for connective %s' % str(connective))
    # Find out whether arg0 is before, after or around arg0
    # Note that right sentence boundaries are exclusive, whereas the min/max are inclusive
    if max1 < min0:
        # arg1 is before arg0, negative token_dist
        token_dist = max1 - min0
        # Sentences that end after arg0 starts and start before arg0 ends
        involved_sents = -len([sent for sent in sentences if min1 < sent[1] and sent[0] < max0])
        # Sentences that start before arg0 starts and end after arg1 ends. Distance subtracts one. Sign is switched
        sent_dist = -len([sent for sent in sentences if max1 < sent[1] and sent[0] <= min0]) + 1
    elif max0 < min1:
        # arg1 is after arg0, positive token_dist
        # As before with switched roles of arg0 and arg1 and switched signs
        token_dist = min1 - max0
        involved_sents = len([sent for sent in sentences if min0 < sent[1] and sent[0] < max1])
        sent_dist = len([sent for sent in sentences if max0 < sent[1] and sent[0] <= min1]) - 1
    else:
        #arg1 is surrounding arg0, set token_dist to zero
        token_dist = 0
        involved_sents = len([sent for sent in sentences if min1 < sent[1] and sent[0] < max1])
        sent_dist = 0
    return token_dist, sent_dist, involved_sents


def arg_pos_in_PCC(connector_folder='/media/arkadi/arkadis_ext/NLP_data/ger_twitter/' +
                                       'potsdam-commentary-corpus-2.0.0/connectors',
                   pickle_folder='data'):
    """ Analyze the argument positions in the PCC and optionally pickle the results

    :param connector_folder:
    :type connector_folder:
    :param pickle_folder:
    :type pickle_folder:
    :return: results of the analysis
    :rtype: dict
    """
    results = list()
    errors = dict()
    pcc = load_pcc(connector_folder, pickle_folder=pickle_folder)
    for discourse in pcc:
        for conn in discourse.connectives:
            conn_txt, arg0_txt, arg1_txt = discourse.get_connective_text(conn)
            try:
                token_dist, sent_dist, involved_sents = analyse_argument_positions(discourse.sentences, conn)
                cont_type, phrase_type = conn.get_type()
                if cont_type == 'cont':
                    result_dict = {'conn_txt': conn_txt, 'arg0_txt': arg0_txt, 'arg1_txt': arg1_txt,
                                   'token_dist': token_dist, 'sent_dist': sent_dist, 'involved_sents': involved_sents,
                                   'cont_type': cont_type, 'phrase_type_0': phrase_type}
                else:
                    result_dict = {'conn_txt': conn_txt, 'arg0_txt': arg0_txt, 'arg1_txt': arg1_txt,
                                   'token_dist': token_dist, 'sent_dist': sent_dist, 'involved_sents': involved_sents,
                                   'cont_type': cont_type, 'phrase_type_0': phrase_type[0], 'phrase_type_1': phrase_type[1]}
                results.append(result_dict)
            except Exception, e:
                errors[discourse.name] = e
    if pickle_folder:
        with open(os.path.join(pickle_folder,'PCC_argdist.pickle'), 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stderr.write(str(errors))
    return results


def compare_PCC_dimlex(PCC_discourse, dimlex):
    """ Answers the following questions:
    Which connectives are present in both?

    :param PCC_discourse:
    :type PCC_discourse: list
    :param dimlex:
    :type dimlex: zwitscher.utils.connectives.Lexicon
    :return:
    :rtype:
    """
    disc_conn_words = []
    for discourse in pcc:
        for disc_conn in discourse.connectives:
            connected_pieces = disc_conn.connected_pieces()
            # Joining connected pieces with white space
            conn_pieces_txt = [' '.join([discourse.tokens[i] for i in piece]) for piece in connected_pieces]
            # Joining non-connected pieces with _
            conn_txt = '_'.join(conn_pieces_txt)
            disc_conn_words.append(conn_txt)
    return disc_conn_words


def analyse_disambiguity(text, lexicon):
    # Creating regexps that can find discontinuous connectives - ungreedy up to a distance of 100 characters
    connective_words = ['.{0,100}? '.join(word.split('_')) for word in lexicon.orthography_variants.keys()]
    # ToDo: Create better regexps to find discontinuous connectives and overlapping matches
    matches = re.findall(('[ %s]|[ %s]' % (string.punctuation, string.punctuation)).join(lexicon.orthography_variants),
                         text)
    print("%i of %i matches are disambiguous" % (len([word for word in matches if lexicon.disambi(word[1:-1])]),
                                                 len(matches)))

if __name__ == '__main__':
    pickle_folder = 'data'
    #pcc = load_pcc(pickle_folder=pickle_folder)
    with open(os.path.join(pickle_folder, 'PCC_disc.pickle'), 'rb') as f:
        pcc = pickle.load(f)
    lexicon = load_dimlex('data/dimlex.xml')
    all_conn = []
    for disc in pcc:
        all_conn.extend(disc.connectives)

    import ipdb; ipdb.set_trace()
    words = compare_PCC_dimlex(pcc, lexicon)
    print(str(words))
