"""
Scripts to test time performance of methods
"""
import timeit
import os.path

from bs4 import BeautifulSoup
from utils.tree import ConstituencyTree

from features import siblings

__author__ = 'arkadi'


with open(os.path.join(os.path.dirname(__file__),
                       'data/test_pcc/syntax/maz-00001.xml')) as f:
    xml_file = f.read()
soup = BeautifulSoup(xml_file)
sentences = soup.find_all('s')
trees = [ConstituencyTree(str(sent)) for sent in sentences]
terminals = []
for tree in trees:
    terminals.extend(tree.terminals)

no_orphans = [ter for ter in terminals if ter.parent is not None]
nodes = [list(tree.iter_nodes()) for tree in trees]

if __name__ == '__main__':
    # Finding with IDs is about 10 times faster
    print 'node.position in tree with ID'
    print timeit.timeit('for terminal in terminals: [ter.id_str for ter in terminal.tree.terminals].index(terminal.id_str)',
                  setup='from __main__ import terminals', number=1)
    print 'node.position in tree with node'
    print timeit.timeit('for terminal in terminals: terminal.tree.terminals.index(terminal)',
                  setup='from __main__ import terminals', number=1)

    print 'finding siblings'
    print timeit.timeit(
        'for terminal in no_orphans: len(terminal.parent.terminals())',
        setup='from __main__ import no_orphans', number=2)

    # print 'finding siblings positions'
    # print timeit.timeit(
    #     'for terminal in no_orphans: [node.position_in_sentence() for node in terminal.parent.terminals()]',
    #     setup='from __main__ import no_orphans', number=2)
    #
    # print 'counting siblings'
    # print timeit.timeit(
    #     'for terminal in terminals: siblings(terminal)',
    #     setup='from __main__ import terminals, siblings', number=2)

    print 'relative pos'
    print timeit.timeit(
        'for node in nodes: len(terminal.parent.terminals())',
        setup='from __main__ import nodes', number=2)