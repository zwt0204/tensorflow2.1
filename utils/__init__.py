# -*- encoding: utf-8 -*-
"""
@File    : __init__.py.py
@Time    : 2020/5/6 16:30
@Author  : zwt
@git   : 
@Software: PyCharm
"""
from .radam import RAdam
from .lamb import LAMB
from .lookhead import Lookahead

__all__ = ['RAdam', 'LAMB', 'Lookahead', 'get_optimizer']


def get_optimizer(optimizer, **kwargs):
    assert optimizer.lower() in ['radam', 'lamb'], ValueError('not a supported optimizer name')
    if optimizer.lower() == 'radam':
        return RAdam(**kwargs)
    if optimizer.lower() == 'lamb':
        return LAMB(**kwargs)