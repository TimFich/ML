import networkx as nx
import random
import matplotlib.pyplot as plt
import os

NUMBER_OF_POINTS = 10
NUMBER_OF_CLUSTERS = 3


def make_clusters(edges):
    result_edges = []
    result_indexes = []

    for i in range(NUMBER_OF_CLUSTERS - 1):
        max_weight = 0
        result_index = 0
        for j in range(len(edges)):
            if edges[j][2] > max_weight and j not in result_indexes:
                max_weight = edges[j][2]
                result_index = j
        result_indexes.append(result_index)

    for i in range(len(edges)):
        if i not in result_indexes:
            result_edges.append(edges[i])

    return result_edges


def find_shortest_path(nodes, edges):
    result_edges = []
    visited_nodes = set()

    current_node = 0
    is_back = False
    is_first_back = True
    back_index = 0
    while len(visited_nodes) != len(nodes):
        shortest_path = float('inf')
        result_edge = ()
        for edge in edges:
            if is_back:
                if edge[0] == current_node:
                    if edge[1] in visited_nodes:
                        continue
                elif edge[1] == current_node:
                    if edge[0] in visited_nodes:
                        continue
            else:
                if edge[1] in visited_nodes or edge[0] in visited_nodes:
                    continue

            if edge in result_edges:
                continue

            if edge[0] == current_node or edge[1] == current_node:
                if edge[2] < shortest_path:
                    shortest_path = edge[2]
                    result_edge = edge

        if current_node not in visited_nodes:
            visited_nodes.add(current_node)

        if len(result_edge) == 0:
            is_back = True
            if is_first_back:
                is_first_back = False
                back_index = len(result_edges) - 1
            else:
                back_index -= 1

            if result_edges[back_index][0] == current_node:
                current_node = result_edges[back_index][1]
            else:
                current_node = result_edges[back_index][0]
        else:
            is_back = False
            back_index += 1
            result_edges.append(result_edge)
            if result_edge[0] == current_node:
                current_node = result_edge[1]
            else:
                current_node = result_edge[0]

    return result_edges


def make_algorithm(nodes, edges, labels):
    path_script = os.getcwd()
    dir_name = 'photos'
    os.mkdir(os.path.join(path_script, dir_name))

    G = nx.Graph()
    n = 0
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(edges)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.savefig(os.path.join(path_script, dir_name) + f'/{n}.png')
    plt.close()

    edges = find_shortest_path(nodes, edges)
    labels = {}
    for edge in edges:
        labels[(edge[0], edge[1])] = edge[2]

    n += 1
    G.clear()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(edges)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.savefig(os.path.join(path_script, dir_name) + f'/{n}.png')
    plt.close()

    edges = make_clusters(edges)
    labels = {}
    for edge in edges:
        labels[(edge[0], edge[1])] = edge[2]
    n += 1
    G.clear()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(edges)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.savefig(os.path.join(path_script, dir_name) + f'/{n}.png')
    plt.close()


if __name__ == '__main__':
    nodes = range(NUMBER_OF_POINTS)
    edges = []
    labels = {}
    for i in range(NUMBER_OF_POINTS):
        for j in range(i + 1, NUMBER_OF_POINTS):
            if random.randint(0, 1) == 1:
                random_weight = random.randint(1, 50)
                edges.append((i, j, random_weight))
                labels[(i, j)] = random_weight

    make_algorithm(nodes, edges, labels)
