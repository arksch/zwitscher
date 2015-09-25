"""
A script to get the connective positions from the PCC
"""
import json

import click

from gold_standard import flat_index_to_nested
from utils.PCC import Parser


__author__ = 'arkadi'


@click.command(help='A script to create nested or unnested indices of connectives '
                    'in the sentences of a discourse from PCC files')
@click.option('--discourse_path', '-d',
              help='Path to the discourse xml file. '
                   'Defaults to data/test_pcc/connectors/maz-00001.xml',
              default='data/test_pcc/connectors/maz-00001.xml')
@click.option('--syntax_path', '-s',
              help='Path to the syntax xml file. '
                   'Defaults to data/test_pcc/syntax/maz-00001.xml',
              default='data/test_pcc/syntax/maz-00001.xml')
@click.option('--nested', '-n',
              is_flag=True,
              help='Output the nested or unnested positions',
              default=True)
def discourse_positions_from_pcc(discourse_path='data/test_pcc/connectors/maz-00001.xml',
                                 syntax_path='data/test_pcc/syntax/maz-00001.xml',
                                 nested=True):
    parser = Parser()
    with open(discourse_path, 'r') as f:
        discourse_xml = unicode(f.read(), 'utf-8')
    with open(syntax_path, 'r') as f:
        syntax_xml = unicode(f.read(), 'utf-8')
    discourse = parser.parse_discourse(discourse_xml, syntax_xml=syntax_xml)
    flat_indices = [conn.positions for conn in discourse]
    if not nested:
        print json.dumps(flat_indices)
        return flat_indices
    else:
        flat_sents = discourse.sentences
        nested_indices = [flat_index_to_nested(flat_sents, i) for i in flat_indices]
        print json.dumps(nested_indices)
        return nested_indices


if __name__ == '__main__':
    discourse_positions_from_pcc()
