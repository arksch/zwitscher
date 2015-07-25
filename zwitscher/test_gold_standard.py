__author__ = 'arkadi'

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
    test_gold_pcc = pcc_to_gold(test_pcc)
    assert test_gold_pcc.iloc[1]['sentences'][0][3] == 'Dagmar'
    assert test_gold_pcc.iloc[1]['connective_positions'] == [(10, 0)]

def test_pcc_with_syntax_to_gold():
    import os
    from utils.PCC import load_connectors as load_pcc
    from gold_standard import pcc_to_gold

    test_pcc = load_pcc(connector_folder=os.path.join(os.path.dirname(__file__), 'data/test_pcc/connectors'),
                        syntax_folder=os.path.join(os.path.dirname(__file__), 'data/test_pcc/syntax'),
                        pickle_folder='')
    test_gold_pcc = pcc_to_gold(test_pcc)
    assert test_gold_pcc.iloc[1]['sentences'][1][0] == 'Dagmar'
    assert test_gold_pcc.iloc[1]['connective_positions'] == [(11, 0)]


if __name__ == '__main__':
    test_pcc_with_syntax_to_gold()
    test_pcc_without_syntax_to_gold()
