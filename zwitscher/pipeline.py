"""
The pipeline to predict argument spans only from a syntax tree
"""
import os
import json
import pickle

import click

from utils.PCC import parse_syntax
from utils.connectives import Discourse, DiscConnective
from predict import predict_sentence_dist, predict_argspans
from gold_standard import nested_indices_to_flat, pcc_to_gold,\
    pcc_to_arg_node_gold, clean_data, same_sentence, different_sentence

__author__ = 'arkadi'


def label_arguments_in_different_sentences(different_sentence_connectives,
                                           connective_dict,
                                           flat_sentences):
    """ Creates argument spans in different sentences

    Dumb approach: Just label the full sentence
    :param different_sentence_connectives: dataframe with connective data in
    the rows, and ['connective_positions', 'sentences'] as columns.
    :type different_sentence_connectives: pd.DataFrame
    :param connective_dict: {flat_conn_positions: connective}
    :type connective_dict: dict
    :param flat_sentences: [(0, 3), (3, 10)] sentence boundaries
    :type flat_sentences: list
    """
    for i in different_sentence_connectives.index:
        row = different_sentence_connectives.ix[i]
        conn_positions = row['connective_positions']
        sentences = row['sentences']
        if len(conn_positions) == 1:
            arg0_sent = conn_positions[0][0]
            # Note, that we could take sentence_dist instead of -1
            arg1_sent = conn_positions[0][0] - 1
        elif len(conn_positions) > 1:
            arg0_sent = conn_positions[-1][0]
            arg1_sent = conn_positions[0][0]
        else:
            raise ValueError('Positions have not been classified')
        arg0 = [(arg0_sent, tok) for tok in
                range(len(sentences[arg0_sent]))]

        if arg1_sent >= 0:
            arg1 = [(arg1_sent, tok) for tok in
                    range(len(sentences[arg1_sent]))]
        else:
            arg1 = []

        flat_connective_positions = nested_indices_to_flat([conn_positions], flat_sentences)[0]
        disc_connective = connective_dict[tuple(flat_connective_positions)]
        disc_connective.arg0 = nested_indices_to_flat([arg0], flat_sentences)[0]
        disc_connective.arg1 = nested_indices_to_flat([arg1], flat_sentences)[0]


filepath = os.path.dirname(__file__)
datapath = os.path.join(filepath, 'data')
default_syntax_path = os.path.join(datapath,'test_pcc/syntax/maz-00001.xml')
default_connective_positions_path = os.path.join(datapath,'test_pcc/maz-00001_nested_conn_indices.json')
default_sent_dist_path = os.path.join(datapath, 'classifiers/default_sent_dist_classification_dict.pickle')
default_argspan_path = os.path.join(datapath, 'classifiers/default_argspan_classification_dict.pickle')
@click.command(help='Running a pipeline to label argument spans in a syntax '
                    'annotated discourse with known positions of connectives')
@click.option('--syntax_trees_xml_path', '-s',
              help='Path to a file with syntactic information in TigerXML format',
              default=default_syntax_path)
@click.option('--connective_positions_path', '-c',
              help="Path to a file with nested connective positions in json format\n"
                   "E.g. '[[[14, 0], [14, 5]], [[11, 0]], [[2, 0], [2, 1]]]'",
              default=default_connective_positions_path)
@click.option('--sent_dist_classification_path', '-sc',
              help='Path to the sentence distance classifier trained with the '
                   'learn_sentdist.py script',
              default=default_sent_dist_path)
@click.option('--argspan_classification_path', '-as',
              help='Path to the argspan classifier trained with the '
                   'learn_argspan.py script',
              default=default_argspan_path)
def pipeline(syntax_trees_xml_path,
             connective_positions_path,
             sent_dist_classification_path,
             argspan_classification_path):
    """ Running a pipeline to label argument spans in a syntax annotated discourse
    with known positions of connectives

    :param syntax_trees_xml_path: Path to TigerXML of the syntax to analyse
    :type syntax_trees_xml_path: str
    :param connective_positions_path: nested positions (sent, tok) of the connectives
    :type connective_positions_path: str
    :param sent_dist_classification_path: path to the sentence distance
    classifier trained with the learn_sentdist.py script
    :type sent_dist_classification_path: str
    :param argspan_classification_path: Path to the argspan classifier trained with the learn_argspan.py script
    :type argspan_classification_path: str
    :return: prints the discourse to standard out
    :rtype:
    """
    # Open all files
    with open(syntax_trees_xml_path, 'r') as f:
        syntax_xml = f.read()
    with open(connective_positions_path, 'r') as f:
        connective_positions = json.load(f)
    with open(sent_dist_classification_path, 'rb') as f:
        sent_dist_classification_dict = pickle.load(f)
    with open(argspan_classification_path, 'rb') as f:
        arg_span_classification_dict = pickle.load(f)

    #connective_positions = [[[11, 0]], [[2, 0], [2, 1]]]
    # Load the trees into one discourse object and get the tokens from the trees
    syntax_trees = parse_syntax(syntax_xml)
    discourse = Discourse()
    discourse.load_tiger_syntax(syntax_trees, load_tokens=True)
    flat_connective_positions = nested_indices_to_flat(connective_positions, discourse.sentences)
    for positions in flat_connective_positions:
        connective = DiscConnective()
        connective.positions = positions
        discourse.connectives.append(connective)

    connective_dict = {tuple(connective.positions): connective
                       for connective in discourse.connectives}
    syntax_id_to_sent_index = dict([(y, x) for (x, y) in
                                    enumerate([synt.id_str for synt in discourse.syntax])])

    pcc_df, syntax_dict = pcc_to_gold([discourse])
    clean_pcc = clean_data(pcc_df)

    #### Predict sentence distance

    sent_dist_clf = sent_dist_classification_dict['sent_dist_classifier']
    sent_dist_feature_list = sent_dist_classification_dict['feature_list']
    sent_dist_label_features = sent_dist_classification_dict['label_features']
    sent_dist_label_encoder = sent_dist_classification_dict['label_encoder']
    sent_dists = predict_sentence_dist(clean_pcc,
                                       sent_dist_clf,
                                       sent_dist_feature_list,
                                       sent_dist_label_features,
                                       sent_dist_label_encoder)
    clean_pcc['sentence_dist'] = sent_dists
    different_sentence_connective_df = different_sentence(clean_pcc)
    label_arguments_in_different_sentences(different_sentence_connective_df,
                                           connective_dict,
                                           discourse.sentences)

    ### Predict argument spans

    same_sentence_connectives = same_sentence(clean_pcc)
    node_df, node_dict = pcc_to_arg_node_gold(same_sentence_connectives,
                                              syntax_dict)

    logit_arg0_clf = arg_span_classification_dict['logit_arg0_clf']
    logit_arg1_clf = arg_span_classification_dict['logit_arg1_clf']
    arg_span_feature_list = arg_span_classification_dict['feature_list']
    arg_span_label_features = arg_span_classification_dict['label_features']
    arg_span_label_encoder = arg_span_classification_dict['label_encoder']
    arg_span_binary_encoder = arg_span_classification_dict['binary_encoder']

    arg0_spans, arg1_spans = predict_argspans(node_df,
                                              syntax_dict,
                                              logit_arg0_clf,
                                              logit_arg1_clf,
                                              arg_span_feature_list,
                                              arg_span_label_features,
                                              arg_span_label_encoder,
                                              arg_span_binary_encoder)

    for conn_id, arg0_span in arg0_spans.items():
        syntax_id, insent_conn_pos = conn_id
        sent_index = syntax_id_to_sent_index[syntax_id]
        nested_conn_index = [(sent_index, tok) for tok in insent_conn_pos]
        flat_conn_index = nested_indices_to_flat([nested_conn_index], discourse.sentences)[0]
        nested_arg0_index = [(sent_index, tok) for tok in arg0_span]
        flat_arg0_span = nested_indices_to_flat([nested_arg0_index], discourse.sentences)[0]
        connective = connective_dict[tuple(flat_conn_index)]
        connective.arg0 = flat_arg0_span

    for conn_id, arg1_span in arg1_spans.items():
        syntax_id, insent_conn_pos = conn_id
        sent_index = syntax_id_to_sent_index[syntax_id]
        nested_conn_index = [(sent_index, tok) for tok in insent_conn_pos]
        flat_conn_index = nested_indices_to_flat([nested_conn_index], discourse.sentences)[0]
        nested_arg1_index = [(sent_index, tok) for tok in arg1_span]
        flat_arg1_span = nested_indices_to_flat([nested_arg1_index], discourse.sentences)[0]
        connective = connective_dict[tuple(flat_conn_index)]
        connective.arg1 = flat_arg1_span

    print discourse.__str__()
    return discourse



if __name__ == '__main__':

    pipeline()
