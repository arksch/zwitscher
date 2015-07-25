#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'arkadi'

def test_constituency_tree():
    import os.path
    from bs4 import BeautifulSoup
    from tree import ConstituencyTree
    with open(os.path.join(os.path.dirname(__file__), '../data/test_pcc/syntax/maz-00001.xml')) as f:
        xml_file = f.read()
    soup = BeautifulSoup(xml_file)
    sentence = str(soup.find_all('s')[4])
    tree = ConstituencyTree(sentence)
    assert [t.word for t in tree.terminals] == ['Der', u'Rückzieher', 'der', 'Finanzministerin', 'ist',
                                                'aber', u'verständlich', '.']
    assert [t.pos for t in tree.terminals] == ['ART', 'NN', 'ART', 'NN', 'VAFIN', 'ADV', 'ADJD', '$.']
    assert tree.path_to_root(2) == ['NP', 'NP', 'S']
    assert tree.id == 's2169'
    assert [unicode(t) for t in tree.terminals] ==\
           [u'<Terminal node s2169_1 in sentence s2169 with parent s2169_501: "Der"-ART>',
            u'<Terminal node s2169_2 in sentence s2169 with parent s2169_501: "R\xfcckzieher"-NN>',
            u'<Terminal node s2169_3 in sentence s2169 with parent s2169_500: "der"-ART>',
            u'<Terminal node s2169_4 in sentence s2169 with parent s2169_500: "Finanzministerin"-NN>',
            u'<Terminal node s2169_5 in sentence s2169 with parent s2169_502: "ist"-VAFIN>',
            u'<Terminal node s2169_6 in sentence s2169 with parent s2169_502: "aber"-ADV>',
            u'<Terminal node s2169_7 in sentence s2169 with parent s2169_502: "verst\xe4ndlich"-ADJD>',
            u'<Terminal node s2169_8 in sentence s2169 with no parent: "."-$.>']
    assert unicode(tree.root) == "<Root node s2169_502 in sentence s2169 with children ['s2169_5', " +\
                                 "'s2169_6', 's2169_7', 's2169_501']: S>"


if __name__ == '__main__':
    test_constituency_tree()