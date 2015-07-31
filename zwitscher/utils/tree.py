#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module includes methods to deal with parse trees
"""
from bs4 import BeautifulSoup  # Python's XML parser
# import networkx as nx
# import matplotlib.pyplot as plt

__author__ = 'arkadi'


class ConstituencyTree():
    """
    A tree object that is created from a TigerXML format.
    """

    def __init__(self, tiger_xml):
        """

        :param tiger_xml: The xml string of one sentence
        :type tiger_xml: basestring
        """
        self.id_str = None
        self.id_dict = {}  # Mapping IDs to nodes
        self.root_id = None
        self.root = None
        self.terminals = []
        self.nodes = []
        self._parse_tiger_xml(tiger_xml)

    def _parse_tiger_xml(self, tiger_xml):
        sent_soup = BeautifulSoup(tiger_xml)
        self.id_str = sent_soup.find('s')['id']
        self.root_id = sent_soup.find('graph')['root']
        root_soup = sent_soup.find(id=self.root_id)
        self.root = Node(node_soup=root_soup, sent_soup=sent_soup, parent=None, tree=self)
        # Some terminals might not be part of the tree! Sort them for any chance, each has the form s1234_x
        # But x is increasing, but it might skip some numbers
        terminal_ids = sorted([t['id'] for t in sent_soup.find_all('t')],
                              key=lambda x: int(x.split('_')[-1]))
        for t_id in terminal_ids:
            if t_id in self.id_dict:
                self.terminals.append(self.id_dict[t_id])
            else:
                self.terminals.append(Node(node_soup=sent_soup.find(id=t_id),
                                           sent_soup=sent_soup, parent=None, tree=self))

    def iter_nodes(self, include_terminals=False):
        """ Iterator over the nodes in the tree

        :param include_terminals: Iterate over non-terminal nodes as well?
        :type include_terminals: bool
        """
        for node in self.nodes:
            if not node.terminal:
                yield node

    def path_to_root(self, terminal_index):
        """
        Gives the categories of the path to the root
        :param terminal_index: index of the terminal position in the sentence
        :type terminal_index: integer
        :return:
        :rtype: list
        """
        terminal = self.terminals[terminal_index]
        ancestor_cats = []
        node = terminal
        while node.parent is not None:
            ancestor_cats.append(node.parent.cat)
            node = node.parent
        return ancestor_cats

    # def draw(self):
    #     G = nx.DiGraph()
    #
    #     for id, node in self.id_dict.iteritems():
    #         if node.terminal:
    #             G.add_node(id, POS=node.pos)
    #         else:
    #             G.add_node(id, cat=node.cat)


class Node():

    def __init__(self, node_soup=None, sent_soup=None, parent=None, tree=None):
        self.id_str = node_soup['id']
        self.parent = parent
        self.tree = tree
        self.tree.id_dict[self.id_str] = self
        self.tree.nodes.append(self)
        self.arg0 = False  # From the gold standard
        self.arg1 = False
        self.arg1_proba = 0.0  # From the classifier
        self.arg0_proba = 0.0
        self.label = ''  # Any label to put
        self.children = []
        self.terminal = False
        self.cat = 'None'
        if node_soup.name == 'nt':
            self.cat = node_soup['cat']
            #try:
            self.children = [Node(node_soup=sent_soup.find(id=edge['idref']),
                                  sent_soup=sent_soup, parent=self, tree=tree)
                             for edge in node_soup.find_all('edge')]
            #except:
            #    import ipdb; ipdb.set_trace()
        if node_soup.name == 't':
            self.terminal = True
            self.word = node_soup['word']
            self.pos = node_soup['pos']

    def __unicode__(self):
        if self.terminal:
            if self.parent is not None:
                fmt = u'<Terminal node %s in sentence %s with parent %s: "%s"-%s>' % (self.id_str,
                                                                                      self.tree.id_str,
                                                                                      self.parent.id_str,
                                                                                      self.word,
                                                                                      self.pos)
            else:
                fmt = u'<Terminal node %s in sentence %s with no parent: "%s"-%s>' % (self.id_str, self.tree.id_str,
                                                                                      self.word, self.pos)
        else:
            if self.parent is not None:
                fmt = u'<Non-terminal node %s in sentence %s with parent %s and children %s: %s>' %\
                      (self.id_str, self.tree.id_str, self.parent.id_str, str([child.id_str for child in self.children]), self.cat)
            else:
                fmt = u'<Root node %s in sentence %s with children %s: %s>' % \
                      (self.id_str, self.tree.id_str, str([child.id_str for child in self.children]), self.cat)
        return fmt

    def path_to_root(self):
        """ Constructs shortest path from this node to the root

        :return: categories of the intermediate nodes
        :rtype: list
        """
        cat_path = []
        node = self
        while node.parent is not None:
            cat_path.append(node.parent.cat)
            node = node.parent
        return cat_path

    def path_from_root(self):
        path = self.path_to_root()
        path.reverse()
        return path

    def path_to_other(self, node):
        """ Constructs shortest path from a terminal to this node

        :param node: other node (should be from the same tree)
        :type node: Node
        :return: categories of the intermediate nodes
        :rtype: list
        """
        my_path = self.path_to_root()
        other_path = node.path_to_root()
        while len(my_path) > 0 and len(other_path) > 0:
            if my_path[-1] == other_path[-1]:
                my_path.pop(-1)
                other_path.pop(-1)
            else:
                break
        other_path.reverse()
        my_path.extend(other_path)
        return my_path

    def path_from_other(self, node):
        path = self.path_to_other(node)
        path.reverse()
        return path

    def position_in_sentence(self):
        terminal_indices = [ter.id_str for ter in self.tree.terminals]
        return terminal_indices.index(self.id_str)

    def terminals(self):
        """ Gets all the terminal descendants of this node

        :return: a list of terminal nodes
        :rtype: list
        """
        terminals = []
        children = self.children
        for child in children:
            if child.terminal:
                terminals.append(child)
            else:
                children.extend(child.children)
        return terminals

    def terminal_indices(self):
        """ Gets the indices of the terminal descendants of this node

        :return: The indices in the sentence
        :rtype: list
        """
        return [self.tree.terminals.index(ter) for ter in self.terminals()]

