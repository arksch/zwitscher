#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
__author__ = 'arkadi'


class TreeTestCase(unittest.TestCase):

    def setUp(self):
        import os.path
        from bs4 import BeautifulSoup
        from tree import ConstituencyTree

        with open(os.path.join(os.path.dirname(__file__),
                               '../data/test_pcc/syntax/maz-00001.xml')) as f:
            xml_file = f.read()
        soup = BeautifulSoup(xml_file)
        sentences = soup.find_all('s')
        self.trees = [ConstituencyTree(str(sent)) for sent in sentences]

        with open(os.path.join(os.path.dirname(__file__),
                               '../data/test_pcc/syntax/maz-00001.xml')) as f:

            xml_file = f.read()
        soup = BeautifulSoup(xml_file)
        sentences = soup.find_all('s')
        self.other_trees = [ConstituencyTree(str(sent)) for sent in sentences]

    def test_fields_tree(self):
        tree = self.trees[4]

        assert tree.id_str == 's2169'

        assert unicode(tree.root) == "<Root node s2169_502 in sentence s2169 with children ['s2169_5', " +\
                                     "'s2169_6', 's2169_7', 's2169_501']: S>"

    def test_terminals(self):
        tree = self.trees[4]
        assert [t.word for t in tree.terminals] == ['Der', u'Rückzieher',
                                                    'der', 'Finanzministerin',
                                                    'ist',
                                                    'aber', u'verständlich',
                                                    '.']

        assert [t.pos for t in tree.terminals] == ['ART', 'NN', 'ART', 'NN',
                                                   'VAFIN', 'ADV', 'ADJD', '$.']

        assert [unicode(t) for t in tree.terminals] == \
               [
                   u'<Terminal node s2169_1 in sentence s2169 with parent s2169_501: "Der"-ART>',

                   u'<Terminal node s2169_2 in sentence s2169 with parent s2169_501: "R\xfcckzieher"-NN>',
                   u'<Terminal node s2169_3 in sentence s2169 with parent s2169_500: "der"-ART>',
                   u'<Terminal node s2169_4 in sentence s2169 with parent s2169_500: "Finanzministerin"-NN>',
                   u'<Terminal node s2169_5 in sentence s2169 with parent s2169_502: "ist"-VAFIN>',
                   u'<Terminal node s2169_6 in sentence s2169 with parent s2169_502: "aber"-ADV>',
                   u'<Terminal node s2169_7 in sentence s2169 with parent s2169_502: "verst\xe4ndlich"-ADJD>',
                   u'<Terminal node s2169_8 in sentence s2169 with no parent: "."-$.>']

    def test_path_to_root(self):
        tree = self.trees[4]
        assert tree.path_to_root(2) == ['NP', 'NP', 'S']

    def test_iter_nodes(self):
        assert len(list(self.trees[3].iter_nodes())) == 7
        assert len(list(self.trees[1].iter_nodes(include_terminals=True))) == 10

    def test_nodes(self):

        big_tree = self.trees[5]
        #print 'Terminals:'
        #print [unicode(t) for t in big_tree.terminals]
        node1 = big_tree.terminals[0]
        node2 = big_tree.terminals[13].parent

        assert big_tree.id_str == 's2170'

        assert node1.path_to_root() == ['NP', 'S']
        assert node2.path_from_root() == ['S', 'NP', 'S', 'VP', 'VP']

        assert node1.path_to_other(node2) == ['S', 'VP', 'VP']

        assert node2.terminal_indices() == [12, 13, 14]


if __name__ == '__main__':
    unittest.main(verbosity=2, buffer=False)