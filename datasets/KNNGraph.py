from datasets.Generator import gen_graph



def KNNGraph(interval,data,label,task):
    a, b = 0, interval
    graph_list = []
    while b <= len(data):
        graph_list.append(data[a:b])
        a += interval
        b += interval
    graphset = gen_graph('KNNGraph', graph_list, label, task)
    return graphset