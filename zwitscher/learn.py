#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A script to learn classification
"""

__author__ = 'arkadi'

from sklearn.preprocessing import LabelEncoder

from sklearn.cross_validation import cross_val_score

from sklearn.ensemble import RandomForestClassifier



connectives = load_data()  # le.fit(data[:,0]), data[:, 0] = le.transform(data[:, 0]) und zurück mit le.inverse_transform
clf = RandomForestClassifier(min_samples_leaf=5, n_jobs=-1, verbose=2)
scores = cross_val_score(clf, connectives.data, connectives.target, cv=5)  # numpy.array