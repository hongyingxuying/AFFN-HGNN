import torch
from math import sqrt
import numpy as np
from datasets.Generator import gen_graph


def RadiusGraph(interval,data,label,task):
    a, b = 0, interval
    graph_list = []
    while b <= len(data):
        graph_list.append(data[a:b])
        a += interval
        b += interval
    graphset = gen_graph("RadiusGraph", graph_list, label, task)
    return graphset


