# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 11:39:38 2019

@author: famato1
"""
import itertools 

def countX(lst, x):
    '''A function counting the number of time an element is present in a list'''
    return lst.count(x) 

def list_combs(x):
    '''A function returning all the possible combination between n numbers'''
    return [c for i in range(len(x)+1) for c in itertools.combinations(x,i)]


def combs(x):
    return itertools.chain.from_iterable(
        itertools.combinations(x, i + 1)
        for i in range(len(x)+1))

def recursive_combination_excluder(lst, r, excludes):
    return (comb for comb in itertools.combinations(lst, r)
                 if any(e.issubset(comb) for e in excludes))
