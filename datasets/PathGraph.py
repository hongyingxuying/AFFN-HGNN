# -*- coding: utf-8 -*-
"""
@author: HanJinZhe
@project: GNNforIFD
@file: PathGraph.py
@time: 2022/11/28 10:49
@description： 
"""

from datasets.Generator import gen_graph


def pathGraph(interval, data, label, task):
    a, b = 0, interval
    graph_list = []
    while b <= len(data):
        graph_list.append(data[a:b])
        a += interval
        b += interval
    graphset = gen_graph('PathGraph', graph_list, label, task)
    return graphset
