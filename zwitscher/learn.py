#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A script to learn classification
"""

__author__ = 'arkadi'
import pandas as pd


from sklearn.preprocessing import LabelEncoder

from sklearn.cross_validation import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from utils.PCC import load as load_pcc
from utils.dimlex import load as load_dimlex
from gold_standard import pcc_to_gold
from features import discourse_connective_text_featurizer as featurizer

# ToDo: load data
# ToDo: featurize data
# ToDo: clean data
# ToDo: encode labels
# ToDo: split data into X and y, train and test
# ToDo: train random forest
# ToDo: evaluate with cross val
# ToDo: train on whole data set
# ToDo: return classifier and label encoder

def load_data(connector_folder='/media/arkadi/arkadis_ext/NLP_data/ger_twitter/' +
                               'potsdam-commentary-corpus-2.0.0/connectors'):
    # Load PCC data
    pcc = load_pcc(connector_folder)
    # creating a pd.DataFrame
    return pcc_to_gold(pcc)

def clean_data(dataframe):
    # Fixme: go on here! Kick out everything that has None in sentences, connective_positions or 'sentence_dist'
    pass

def featurize_data(dataframe):
    features = dataframe.apply(lambda row: pd.Series(featurizer(row['sentences'], row['connective_positions'])),
                               axis=1, reduce=False)
    return features





def learn_random_forest():
    connectives = load_data()  # le.fit(data[:,0]), data[:, 0] = le.transform(data[:, 0]) und zurück mit le.inverse_transform
    clf = RandomForestClassifier(min_samples_leaf=5, n_jobs=-1, verbose=2)
    scores = cross_val_score(clf, connectives.data, connectives.target, cv=5)  # numpy.array

def main(connector_folder='/media/arkadi/arkadis_ext/NLP_data/ger_twitter/' +
                          'potsdam-commentary-corpus-2.0.0/connectors'):
    pcc_df = load_data(connector_folder)  # Loaded X and y into one dataframe
    clean_pcc = clean_data(pcc_df)
    features = featurize_data(clean_pcc)  # Got features of X


if __name__ == '__main__':
    main()