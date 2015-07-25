#!/usr/bin/env python
# -*- coding: utf-8 -*-

def test_pcc_parser():
    from PCC import Parser
    import os.path
    test_pcc_path = os.path.join(os.path.dirname(__file__), '../data/test_pcc/')
    with open(os.path.join(test_pcc_path, 'connectors/maz-00001.xml'), 'r') as f:
        discourse_xml = f.read()
    with open(os.path.join(test_pcc_path, 'syntax/maz-00001.xml'), 'r') as f:
        syntax_xml = f.read()
    pcc_parser = Parser()
    discourse = pcc_parser.parse_discourse(discourse_xml, syntax_xml=syntax_xml)

    assert len(discourse.connectives) == 9, 'Parsed all connectives'
    assert len(discourse.syntax) == 15, 'Parsed all syntax trees'
    assert discourse.tokens[discourse.sentences[-2][0]: discourse.sentences[-2][1]] == \
           [t.word for t in discourse.syntax[-2].terminals], 'Syntax and discourse tokens match'


if __name__ == '__main__':
    test_pcc_parser()