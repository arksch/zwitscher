"""
Unit tests for the analysis methods
"""
from utils.connectives import DiscConnective
from analysis import analyse_argument_positions

__author__ = 'arkadi'


def test_analyse_argument_positions():
    sentences = [(0, 10), (10, 20), (20, 30)]
    # A connective with arg1 before arg0 and in another sentence
    test_conn1 = DiscConnective()
    test_conn1.positions = [10]
    test_conn1.arg0 = range(10, 20)
    test_conn1.arg1 = range(0, 10)
    token_dist, sent_dist, involved_sents = analyse_argument_positions(sentences, test_conn1)
    assert (token_dist, sent_dist, involved_sents) == (-1, -1, -2)
    # A connective with arg1 after arg0 in the same sentence and token distance 1
    test_conn2 = DiscConnective()
    test_conn2.positions = [10]
    test_conn2.arg0 = range(10, 15)
    test_conn2.arg1 = range(16, 20)
    token_dist, sent_dist, involved_sents = analyse_argument_positions(sentences, test_conn2)
    assert (token_dist, sent_dist, involved_sents) == (2, 0, 1)
    # A connective with arg1 wrapping around arg0
    test_conn3 = DiscConnective()
    test_conn3.positions = [12]
    test_conn3.arg0 = range(12, 15)
    test_conn3.arg1 = range(15, 20)
    test_conn3.arg1.extend([10,11])
    token_dist, sent_dist, involved_sents = analyse_argument_positions(sentences, test_conn3)
    assert (token_dist, sent_dist, involved_sents) == (0, 0, 1)  # (1,0,1)

