#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module hold helper classes to deal with discourses, connectives and lexica
"""
import re

__author__ = 'arkadi'


class DiscConnective(object):
    """
    This class bundles information about an explicit connective in a text.
    """
    def __init__(self):
        self.relation = ""  # Which kind of relation does the connective represent
        self.impl = False  # Implicit or explicit connective
        self.positions = list()  # Token positions of the connective itself
        self.arg0 = list()  # Token positions of the argument span the discourse connective is part of
        self.arg1 = list()  # Token positions of the argument span the discourse connective is *not* part of

    def __str__(self):
        fmt = u"connective:%s \n" % str(self.positions)
        fmt += u"relation: %s\n" % self.relation
        fmt += u"arg0: %s\n" % str(self.arg0)
        fmt += u"arg1: %s" % str(self.arg1)
        return fmt

    def get_type(self):
        """
        Helper function to check whether the connective is a a continuous sequence or a discontinuous sequence,
        and whether each part of the sequence is a single word or phrasal.
        E.g. umso weniger ... als is disc, phrasal, single
        """
        if len(self.positions) == 1:
            return 'cont', 'single'
        elif len(self.positions) > 1:
            if max(self.positions) - min(self.positions) == len(self.positions) - 1:
                return 'cont', 'phrasal'
            else:
                # Assume that every discontinuous connective has at most two parts. Find out which of those are phrasal.
                if min(self.positions) + 1 in self.positions:
                    a = 'phrasal'
                else:
                    a = 'single'
                if max(self.positions) - 1 in self.positions:
                    b = 'phrasal'
                else:
                    b = 'single'
                return 'discont', (a, b)
        else:
            raise Exception('No positions saved for the connective.')

    def connected_pieces(self):
        """ Helper to get the connected pieces
        :return: A list of lists with indices
        :rtype: list
        """
        if not self.positions:
            # No positions set for this discourse
            return [[]]
        self.positions = sorted(self.positions)
        counter = 0
        all_pieces = []
        this_piece = []
        while counter < len(self.positions) - 1:
            this_piece.append(self.positions[counter])
            if self.positions[counter] + 1 != self.positions[counter + 1]:
                all_pieces.append(this_piece)
                this_piece = []
            counter += 1
        this_piece.append(self.positions[counter])
        all_pieces.append(this_piece)
        return all_pieces

    def is_complete(self):
        """ Was everything parsed for the discourse or are we missing stuff
        :rtype: bool
        """
        return self.positions and self.arg0 and self.arg1 and self.relation


class Discourse(object):
    """
    This class bundles information about a discourse. Mainly this is a paragraph.
    """
    def __init__(self):
        self.name = ""
        self.rawtext = ""
        self.tokens = list()
        self.sentences = list()  # A list of tuples with start and end points
        self.connectives = list()  # A list of DiscConnective objects
        self.annotations = {}
        self.syntax = []  # A list with ConstituencyTrees for each sentence

    def __iter__(self):
        """
        :return: Connectives of the discourse
        :rtype: iterator
        """
        for connective in self.connectives:
            yield connective

    def __str__(self):
        fmt = u"Discourse '%s':\n==========\n" % self.name
        for conn in sorted(self.connectives, key=lambda x: x.positions[0]):
            try:
                fmt += u"connective: %s\n" % ' '.join([self.tokens[i] for i in sorted(conn.positions)])
                fmt += u"arg0: %s\n" % ' '.join([self.tokens[i] for i in sorted(conn.arg0)])
                fmt += u"arg1: %s\n\n" % ' '.join([self.tokens[i] for i in sorted(conn.arg1)])
            except UnicodeEncodeError, e:
                print e
                print "Showing everything that can still be decoded without errors."
        return fmt

    def set_sentences(self, sentences=[]):
        """
        Set the sentences field. Sentences are a sorted list of tuples that cover the whole token length. [(0,3),(3,10)]
        If no sentences are given, points are taken as sentence boundaries.
        A sentence can then be directly queried from the tokens by
        s, e = self.sentences[0]
        sent_tok = self.tokens[s:e]
        """
        if sentences:
            self.sentences = sentences
        else:
            # Find the sentence beginnings by points (doesn't account for abbreviations)
            # ToDo: Make this smarter
            points = [i + 1 for (i, tok) in enumerate(self.tokens) if re.match('[.!?]+', tok) is not None]
            if not len(self.tokens) in points:
                # The last sentence might not have been closed
                points.append(len(self.tokens))
            # Shift them back by one and add 0 to get the sentence beginnings and zip them
            prev_points = [0]
            prev_points.extend([pt for pt in points[:-1]])
            self.sentences = zip(prev_points, points)

    # METHOD IS NOT USED, DUE TO COMPATIBILITY PROBLEMS
    # def load_annotations(self, annotations):
    #     """
    #     Loading annotated data from the PCC files or the TreeTagger
    #     :param annotations: a dict with sentence index as key and a list with a dict per token as values
    #     :type annotations: dict
    #     """
    #     self.annotations = annotations
    #
    #     # Setting the sentences as they are from the annotations
    #     sentence_boundaries = []
    #     left_boundary = 0
    #     for sentence in annotations.itervalues():
    #         right_boundary = left_boundary + len(sentence)
    #         sentence_boundaries.append((left_boundary, right_boundary))
    #         left_boundary = right_boundary
    #     self.set_sentences(sentence_boundaries)
    #
    #     # Sanity check: Do we have as many annotated tokens as tokens in the discourse?
    #     if not len(self.tokens) == sum([len(anno) for anno in self.annotations.itervalues()]):
    #         print 'Annotated tokens should match discourse tokens!'
    #         print [[a['word'] for a in anno] for anno in self.annotations.itervalues()]
    #         print [self.tokens[s[0]:s[1]] for s in self.sentences]

    def load_tiger_syntax(self, tiger_trees, set_sentences=True, load_tokens=False):
        """
        Loading annotations from the syntax files
        :param tiger_trees: Trees with syntactic information, see zwitscher.utils.tree.ConstituencyTree
        :type tiger_trees: list
        :param set_sentences: set sentences as given by the trees
        :param load_tokens: load the tokens from the trees
        """
        self.syntax = tiger_trees
        # Setting the sentences as they are given by the trees
        sentence_boundaries = []
        left_boundary = 0
        for tree in self.syntax:
            right_boundary = left_boundary + len(tree.terminals)
            sentence_boundaries.append((left_boundary, right_boundary))
            left_boundary = right_boundary

        if set_sentences:
            self.set_sentences(sentence_boundaries)

        if load_tokens:
            self.tokens = []
            for tree in tiger_trees:
                self.tokens.extend([ter.word for ter in tree.terminals])

        # Sanity check: Do we have as many annotated tokens as tokens in the discourse?
        if not len(self.tokens) == sum([len(tree.terminals) for tree in self.syntax]):
            print 'Total number of annotated tokens from syntax trees should match discourse tokens!'
            print [[a['word'] for a in anno] for anno in self.annotations.itervalues()]
            print [self.tokens[s[0]:s[1]] for s in self.sentences]


    def get_connective_text(self, connective):
        """
        Helper function to get the text that a connective refers to
        """
        conn_txt = ' '.join([self.tokens[i] for i in connective.positions])
        arg0_txt = ' '.join([self.tokens[i] for i in connective.arg0])
        arg1_txt = ' '.join([self.tokens[i] for i in connective.arg1])
        return conn_txt, arg0_txt, arg1_txt

    def connective_in_one_sentence(self, connective):
        """
        Helper function to check, whether the connective is in the same sentence
        This is trivially true for connectives of length one
        :return: None if the connective doesnt have positions. True if the connective is in the same sentence.
        False otherwise
        """
        if len(connective.positions) < 1:
            return None
        elif len(connective.positions) == 1:
            return True
        elif len(connective.positions) > 1:
            # Finding the sentence of the first word of the connective
            sent = [sent for sent in self.sentences if sent[0] <= connective.positions[0] < sent[1]][0]
            if len([pos for pos in connective.positions if not (sent[0] <= pos < sent[1])]) > 0:
                # There are words outside this sentence
                return False
            else:
                return True


class LexConnective(object):
    """
    This class bundles general information about connectives, as they are in the dimlex lexicon.
    """
    def __init__(self):
        self.id_str = ''
        self.disambi = True
        self.canonical = ''
        self.orths = []  # List with orthographical variants. Discontinous "entweder_oder"

    def __str__(self):
        fmt = "LexConnective\n"
        fmt += "%s\n" % self.canonical
        fmt += "Disambiguous: %s\n" % str(self.disambi)
        fmt += "Orthographies: %s" % str(self.orths)
        return fmt


class Lexicon(object):
    """
    This class represents the dimlex lexicon for connectives
    """
    def __init__(self):
        self.connectives = {}  # Dict with canonical spelling as key, LexConnective as values
        self.orthography_variants = {}  # Maps orthographical variants to canonical. Discontinous "entweder_oder"

    def disambi(self, word):
        """ Helper to tell whether a word disambiguously represents connective

        :type word: basestring
        :rtype: bool
        """
        return self.connectives[self.orthography_variants[word]].disambi


