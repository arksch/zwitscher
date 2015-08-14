"""
The pipeline to predict argument spans only from a syntax tree
"""
import os
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
    for i in range(len(different_sentence_connectives)):
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
        row['arg0'] = arg0
        arg1 = [(arg1_sent, tok) for tok in
                range(len(sentences[arg1_sent]))]
        row['arg1'] = arg1

        flat_connective_positions = nested_indices_to_flat([conn_positions], flat_sentences)[0]
        disc_connective = connective_dict[tuple(flat_connective_positions)]
        disc_connective.arg0 = arg0
        disc_connective.arg1 = arg1


def pipeline(syntax_trees_xml_path,
             connective_positions,
             sent_dist_classification_pickle,
             argspan_classification_pickle):
    """

    :param syntax_trees_xml_path:
    :type syntax_trees_xml_path:
    :param connective_positions: nested positions (sent, tok) of the connectives
    :type connective_positions: list
    :param sent_dist_classification_pickle:
    :type sent_dist_classification_pickle:
    :param argspan_classification_pickle:
    :type argspan_classification_pickle:
    :return:
    :rtype:
    """
    # Open all files
    with open(syntax_trees_xml_path, 'r') as f:
        syntax_xml = f.read()
    with open(sent_dist_classification_pickle, 'rb') as f:
        sent_dist_classification_dict = pickle.load(f)
    with open(argspan_classification_pickle, 'rb') as f:
        arg_span_classification_dict = pickle.load(f)

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


if __name__ == '__main__':
    syntax_path = 'data/test_pcc/syntax/maz-00001.xml'
    connective_positions = [[(2, 0), (2, 1)]]
    sent_dist_classification_pickle = 'data/classifiers/d7eaf2f0713f41b3b1d9ed1ad3acd194_sent_dist_classification_dict.pickle'
    argspan_classification_pickle = 'data/classifiers/7ad155392ef74f428430765e55df4200_argspan_classification_dict.pickle'
    pipeline(syntax_path,
             connective_positions,
             sent_dist_classification_pickle,
             argspan_classification_pickle)