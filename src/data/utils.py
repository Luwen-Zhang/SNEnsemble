import numpy as np
from typing import List


def get_corr_sets(where_corr: np.ndarray, names: List):
    where_corr = [[names[x] for x in y] for y in where_corr]
    corr_chain = {}

    def add_edge(x, y):
        if x not in corr_chain.keys():
            corr_chain[x] = [y]
        elif y not in corr_chain[x]:
            corr_chain[x].append(y)

    for x, y in zip(*where_corr):
        if x != y:
            add_edge(x, y)
            add_edge(y, x)
    corr_feature = list(corr_chain.keys())
    for x in np.setdiff1d(names, corr_feature):
        corr_chain[x] = []

    def dfs(visited, graph, node, ls):
        if node not in visited:
            ls.append(node)
            visited.add(node)
            for neighbour in graph[node]:
                ls = dfs(visited, graph, neighbour, ls)
        return ls

    corr_sets = []
    for x in corr_feature[::-1]:
        if len(corr_sets) != 0:
            for sets in corr_sets:
                if x in sets:
                    break
            else:
                corr_sets.append(dfs(set(), corr_chain, x, []))
        else:
            corr_sets.append(dfs(set(), corr_chain, x, []))

    corr_sets = [[x for x in y] for y in corr_sets]
    return corr_feature, corr_sets
