# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 22:27:50 2022

@author: Shahir, Faraz, Pratyush
"""

def make_pair_shape(t):
    return t if isinstance(t, tuple) else (t, t)
