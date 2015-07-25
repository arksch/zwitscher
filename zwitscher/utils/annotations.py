#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file holds methods to parse the annotation file
"""

__author__ = 'arkadi'


def parse(txt):
    split = txt.split('\n\n')
    sent_dict = {}
    for sent in split:
        if sent.startswith('nodes of sentence'):
            sent_dict.update(eval('{%s}' % sent.replace('\n', '')[18:]))
    return sent_dict
