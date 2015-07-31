__author__ = 'arkadi'
import unittest


def test_flat_index_to_nested():
    from gold_standard import flat_index_to_nested
    flat_sents = [(0, 3), (3, 6)]
    sent_to_index_dict = dict([(i, sent) for sent, i in enumerate(flat_sents)])
    flat_indices = [2, 5]
    nested_indices = flat_index_to_nested(flat_sents, flat_indices, sent_to_index_dict=sent_to_index_dict)
    assert nested_indices == [(0, 2), (1, 2)]
    flat_indices = [2, 3]
    nested_indices = flat_index_to_nested(flat_sents, flat_indices, sent_to_index_dict=sent_to_index_dict)
    assert nested_indices == [(0, 2), (1, 0)]

def test_pcc_without_syntax_to_gold():
    import os
    from utils.PCC import load_connectors as load_pcc
    from gold_standard import pcc_to_gold

    test_pcc = load_pcc(connector_folder=os.path.join(os.path.dirname(__file__), 'data/test_pcc/connectors'),
                        syntax_folder='',
                        pickle_folder='')
    test_gold_pcc, _ = pcc_to_gold(test_pcc)
    assert test_gold_pcc.iloc[1]['sentences'][0][3] == 'Dagmar'
    assert test_gold_pcc.iloc[1]['connective_positions'] == [(10, 0)]


def test_subset():
    from gold_standard import subset
    assert not subset([1, 2, 3], [1, 2, 4])
    assert subset([1, 2], [0, 3, 2, 1])


class SyntaxGoldTestCase(unittest.TestCase):

    def setUp(self):
        import os.path
        from utils.PCC import load_connectors as load_pcc

        self.test_pcc = load_pcc(
            connector_folder=os.path.join(os.path.dirname(__file__),
                                          'data/test_pcc/connectors'),
            syntax_folder=os.path.join(os.path.dirname(__file__),
                                       'data/test_pcc/syntax'),
            pickle_folder='')

    def test_pcc_to_gold(self):
        from gold_standard import pcc_to_gold
        from gold_standard import label_arg_node

        test_gold_pcc, syntax_dict = pcc_to_gold(self.test_pcc)
        assert test_gold_pcc.iloc[1]['sentences'][1][0] == 'Dagmar'
        assert test_gold_pcc.iloc[1]['connective_positions'] == [(11, 0)]


        # print test_gold_pcc.iloc[8]['sentences'][8]

        tree_id = test_gold_pcc.iloc[8]['syntax_ids'][8]
        tree = syntax_dict[tree_id]
        nested_conn = test_gold_pcc.iloc[8]['connective_positions']
        nested_arg0 = test_gold_pcc.iloc[8]['arg0']
        nested_arg1 = test_gold_pcc.iloc[8]['arg1']
        insent_arg0 = [tok for (sent, tok) in nested_arg0]
        insent_arg1 = [tok for (sent, tok) in nested_arg1]
        insent_conn = [tok for (sent, tok) in nested_conn]
        assert insent_arg0 == [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        assert insent_arg1 == [11, 12, 13]
        assert insent_conn == [14]

        label_arg_node(insent_arg0, tree, label=0)
        label_arg_node(insent_arg1, tree, label=1)

        for node in tree.iter_nodes():
            if node.arg1:
                assert node.id == 's2173_507'
            if node.arg0:
                assert node.id == 's2173_506'





if __name__ == '__main__':
    test_pcc_without_syntax_to_gold()
    unittest.main(verbosity=1, buffer=False)
