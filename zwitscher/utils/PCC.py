#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import re
import os
import sys
import pickle

import xmltodict
from bs4 import BeautifulSoup

from connectives import DiscConnective, Discourse
from tree import ConstituencyTree

__author__ = 'arkadi'

class Parser(object):
    """
    This is a parser for the connectors of the Potsdam Commentary Corpus (PCC)

    """
    def __init__(self):
        pass

    def numerate_tokens(self, xml, tokens=None):
        """
        This helper function enumerates the tokens in the xml in the form token_000017
        :param xml: xml string
        :type xml: basestring
        :param tokens: The tokens that shall be numerated. If this is None, than the tokens are parsed from the file
        :type tokens: list
        :return: xml string
        :rtype: basestring
        """
        if tokens is None:
            # Get all the tokens outside < > paranthesis
            tokens = re.sub('<[^>]*>', '', xml).split()
        # We take a pointer to search through the text
        pointer = xml.find('<discourse>') + len('<discourse>')
        assert pointer != -1, "Have to find <discourse> in the xml"
        # We split off the segments that were already searched...
        segments = [xml[:pointer]]
        for number, token in enumerate(tokens):
            # Note all tokens are alphanumerical. Dealing with tokens that would mess up regex
            token = re.escape(token)
            # Searching for the first match of the token in the rest of the text
            # The regex is looking only for tokens that are first followed by a < before a >,
            # meaning they are outside < >
            m = re.search(token + '(?=[^>]*<)', xml[pointer:])
            segments.append(xml[pointer:pointer+m.end()] + '__' + str(number).zfill(6))
            pointer += m.end()
        # ..get the last segment and join everything back together
        segments.append(xml[pointer:])
        xml = ''.join(segments)
        return xml

    def get_text_pos(self, numerated_text):
        # Split into tuples of tokens and indices
        tokens_indexed = [token_pos.split('__') for token_pos in numerated_text.split()]
        # Sort this list by position
        tokens_indexed = sorted(tokens_indexed, key=lambda tok_num: tok_num[1])
        # unzip it
        tokens, positions = zip(*tokens_indexed)
        # get the raw text
        text = ' '.join(tokens)
        # converting the positions back into integers
        positions = [int(pos) for pos in positions]
        return text, positions

    def get_unit_text(self, unit):
        """
        Return the full text of a unit. This will change the order.
        :param unit:
        :type unit:
        :return:
        :rtype:
        """
        text = ""
        units = self.get_children(unit)
        for unit in units:
            if unit.has_key('#text'):
                text += unit['#text'] + ' '
        return text

    def get_children(self, unit, include_root=True):
        """
        Gets all children, including units, connectives and the root itself
        :param unit:
        :type unit:
        :return:
        :rtype:
        """
        units = [unit]
        pointer = 0
        while pointer < len(units):
            unit = units[pointer]
            if unit.has_key('unit'):
                children = unit['unit']
                if isinstance(children, list):
                    units.extend(children)
                else:
                    units.append(children)
            if unit.has_key('connective'):
                children = unit['connective']
                if isinstance(children, list):
                    units.extend(children)
                else:
                    units.append(children)
            pointer += 1
        if not include_root:
            units = units[1:]
        return units


    def parse_discourse(self, discourse_xml, syntax_xml=None):
        """
        Parses an xml string into a Discourse object
        :param discourse_xml: xml string as in the PCC
        :type discourse_xml: basestring
        :param syntax_xml: TigerXML string with syntactical information of the discourse
        :type syntax_xml: basestring
        :return: dictionary of TextConnective objects
        :rtype: Discourse
        """
        connectives = dict()
        tokens = re.sub('<[^>]*>', '', discourse_xml).split()
        discourse_xml = self.numerate_tokens(discourse_xml, tokens=tokens)
        discourse_dict = xmltodict.parse(discourse_xml)
        units = self.get_children(discourse_dict['discourse'], include_root=False)
        for unit in units:
            # Each unit is an ordered dict.
            # If it has a @type key it is a unit
            if unit.has_key('@type'):
                # Read the text
                text_numerated = self.get_unit_text(unit)
                if text_numerated is not None:
                    # Add the position information to the connectives
                    unit_text, unit_pos = self.get_text_pos(text_numerated)
                    unit_type = unit['@type']
                    unit_id = unit['@id']
                    if not connectives.has_key(unit_id):
                        # Create new connective object with the given argument
                        connective = DiscConnective()
                    else:
                        # Add arguments to existing connective
                        connective = connectives[unit_id]
                    # Extend the list of positions with the new position
                    if unit_type == 'int':
                        connective.arg0.extend(unit_pos)
                    else:
                        connective.arg1.extend(unit_pos)
                    connectives[unit_id] = connective
                else:
                    # No text saved in the unit. This is curious
                    print('No text in the following unit')
                    print(unit)

            # Connectives are units that have the @relation key
            if unit.has_key('@relation'):
                id = unit['@id']
                relation = unit['@relation']
                text = self.get_unit_text(unit)
                raw_conn, positions = self.get_text_pos(text)
                # Checking whether we already added the connective
                if connectives.has_key(id):
                    # Update a connective
                    connective = connectives[id]
                else:
                    connective = DiscConnective()
                if connective.relation:
                    # Since there already is a relation present, we are dealing with a discontinuous connective
                    assert connective.relation == relation, 'Discontinuous connectives have to have the same relation'
                else:
                    connective.relation = relation
                connective.positions.extend(positions)
                connectives[id] = connective
        # Wrap the results in a Discourse object
        discourse = Discourse()
        discourse.tokens = tokens
        discourse.rawtext = ' '.join(tokens)
        discourse.connectives = connectives.values()
        if syntax_xml is not None:
            tiger_syntax_dict = parse_syntax(syntax_xml)
            discourse.load_tiger_syntax(tiger_syntax_dict)
        else:
            # No good way to find sentence boundaries, so we just take [.?!]
            discourse.set_sentences()
        return discourse


def parse_syntax(xml_text):
    soup = BeautifulSoup(xml_text)
    sentences_xml = soup.find_all('s')
    annotations = []  # A list with a Constituency Tree for each sentence
    # Get the constituency trees
    for sent_xml in sentences_xml:
        annotations.append(ConstituencyTree(str(sent_xml)))
    # Sort the output just in case
    # annotations = sorted(annotations, key=lambda tree: int(tree.id_str[1:]))  # ids are of the form s1234
    return annotations


def load_connectors(connector_folder='/media/arkadi/arkadis_ext/NLP_data/ger_twitter/' +
                                     'potsdam-commentary-corpus-2.0.0/connectors',
                    syntax_folder='/media/arkadi/arkadis_ext/NLP_data/ger_twitter/' +
                                  'potsdam-commentary-corpus-2.0.0/syntax',
                    pickle_folder='data/'):
    """ Load the whole PCC connectors into a discourse objects

    :param connector_folder:
    :type connector_folder:
    :param pickle_folder:
    :type pickle_folder:
    :return: PCC discourses
    :rtype: list
    """
    parser = Parser()
    errors = dict()
    pcc = []
    for conn_file in os.listdir(connector_folder):
        with open(os.path.join(connector_folder, conn_file), 'r') as f:
            discourse_xml = unicode(f.read(), 'utf-8')
        if syntax_folder:
            with open(os.path.join(syntax_folder, conn_file), 'r') as f:
                syntax_xml = unicode(f.read(), 'utf-8')
        else:
            syntax_xml = None
        try:
            discourse = parser.parse_discourse(discourse_xml, syntax_xml=syntax_xml)
            discourse.name = conn_file
            pcc.append(discourse)
        except Exception, e:
            errors[conn_file] = e
    if pickle_folder:
        with open(os.path.join(pickle_folder, 'PCC_disc.pickle'), 'wb') as f:
            pickle.dump(pcc, f, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stderr.write(str(errors))
    return pcc

