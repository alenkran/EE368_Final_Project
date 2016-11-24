import networkx as nx
import numpy as np

def greedy_reconstruct_frame_order(iso_result):
    num_frames, num_components = iso_result.shape

    # Compute full (duplicated distance matrix)
    distances = np.zeros((num_frames, num_frames))    
    for fr1 in range(num_frames):
        for fr2 in range(num_frames):
            distances[fr1, fr2] = np.linalg.norm(iso_result[fr1,] - iso_result[fr2,])
    distances[distances==0] = np.nan
    overall_std = np.nanstd(distances)
    overall_mean = np.nanmean(distances)

    # Purge frames who's average distance to every other frame is too large
    distances_copy = distances
    for fr in range(num_frames):
        frame_mean = np.nanmean(distances[fr,])
        if abs(frame_mean - overall_mean)/overall_std > 1:
            distances[fr,] = np.nan
            distances[:,fr] = np.nan

    pair1, pair2 = np.unravel_index(np.nanargmin(distances), distances.shape)
    distances[pair1, pair2] = np.nan
    distances[pair2, pair1] = np.nan
    final_order = np.array([pair1, pair2])   

    while True:
        min1 = -1
        min2 = -1
        if np.sum(np.isnan(distances[pair1,])) == num_frames:
            min1 = np.inf
        else:
            min1 = np.nanmin(distances[pair1,])
        if np.sum(np.isnan(distances[pair2,])) == num_frames:
            min2 = np.inf
        else:           
            min2 = np.nanmin(distances[pair2,])
        if min1 == np.inf and min2 == np.inf:
            break
        if min1 < min2:
            minidx = np.nanargmin(distances[pair1,])
            final_order = np.insert(final_order, 0, minidx)
            distances[pair1,] = np.nan
            distances[:,pair1] = np.nan
            pair1 = minidx
        else:
            minidx = np.nanargmin(distances[pair2,])
            final_order = np.append(final_order, minidx)
            distances[pair2,] = np.nan
            distances[:,pair2] = np.nan
            pair2 = minidx
        distances[pair1, pair2] = np.nan
        distances[pair2, pair1] = np.nan
    return final_order

def graph_reconstruct_frame_order(iso_result):
    num_frames, num_components = iso_result.shape

    # Compute full (duplicated distance matrix)
    distances = np.zeros((num_frames, num_frames))    
    for fr1 in range(num_frames):
        for fr2 in range(num_frames):
            distances[fr1, fr2] = np.linalg.norm(iso_result[fr1,] - iso_result[fr2,])

    G = nx.from_numpy_matrix(distances)
    #G = nx.minimum_spanning_tree(G)
    return G

def hamilton(G):
    F = [(G,[G.nodes()[0]])]
    n = G.number_of_nodes()
    while F:
        graph,path = F.pop()
        confs = []
        for node in graph.neighbors(path[-1]):
            conf_p = path[:]
            conf_p.append(node)
            conf_g = nx.Graph(graph)
            conf_g.remove_node(path[-1])
            confs.append((conf_g,conf_p))
        for g,p in confs:
            if len(p)==n:
                return p
            else:
                F.append((g,p))
    return F