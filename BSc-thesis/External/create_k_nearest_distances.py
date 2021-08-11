from numpy.random import uniform
from numpy.linalg import norm
from scipy.spatial import Delaunay
from tqdm.notebook import tqdm

from sklearn.neighbors import kneighbors_graph
# from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances
from scipy.sparse import csc_matrix
from copy import deepcopy

from Stable_Tree.utils.utils import break_triangle_inequality
from Stable_Tree.graph_generation.graph_generation import Complete_graph, k_nearest_neighbor_graph

import numpy as np
import networkit as nk
import networkx as nx
import time

# def patch_together(G, position, bridges):
    
#     G = deepcopy(G)
    
#     list_of_all_nodes = np.array(range(G.numberOfNodes()))
    
#     cc = nk.components.ConnectedComponents(G)
#     cc.run()
#     components = cc.getComponents()
#     number_of_components = cc.numberOfComponents()
    
#     while number_of_components != 1:
        
#         for component in tqdm(components, desc="Patching the components together"):

#             compA = np.array(component)
#             compB = np.delete(list_of_all_nodes, compA)


#             pointsA = position[compA]
#             pointsB = position[compB]
#             diff = pairwise_distances(pointsA, pointsB, n_jobs=-1)
#             top_bridges = np.argpartition(diff.flatten(), bridges)[0:bridges]
#             indices = np.unravel_index(top_bridges, diff.shape)
#             nodesA = compA[indices[0]]
#             nodesB = compB[indices[1]]

#             for i in range(bridges):
#                 try:
#                     nodeA = nodesA[i]
#                     nodeB = nodesB[i]
#                 except:
#                     print(nodesA, nodesB, indices, top_bridges, compA, compB)
#                     raise RuntimeError("Bug somewhere")
#                 distance = norm(position[nodeA] - position[nodeB])
#                 if not G.hasEdge(nodeA, nodeB):
#                     G.addEdge(nodeA, nodeB, distance)
                    
                    
#         cc = nk.components.ConnectedComponents(G)
#         cc.run()
#         components = cc.getComponents()
#         number_of_components = cc.numberOfComponents()
        
#     return G


def k_nearest_d(k, distances):
    ind_1 = []
    ind_2 = []
    
    for i, row in enumerate(distances):
        result = np.argpartition(row, k)[0:k]
        for j in range(k):
            ind_1.append(i)
            ind_2.append(result[j])
            
    knn = np.zeros(distances.shape)
    knn[ind_1, ind_2] = distances[ind_1, ind_2]
    
    knn = csc_matrix(knn)

    G = nx.from_scipy_sparse_matrix(knn)
    G = nk.nxadapter.nx2nk(G, weightAttr="weight")
            
    return G          
        

def find_k_d(distances):
    k = 8 # min k = 8
    G = k_nearest_d(k, distances)
    
    if not is_connected(G):
        
        print("The k-nearest graph is not connected for the guess k=6")
        
        minimum = k
        while True:
                print("try the new new upper limit:", k)
                
                k = 2*k
                
                G = k_nearest_d(k, distances)
                if is_connected(G):
                    maximum = k
                    k = minimum + (maximum - minimum) // 2
                    break
                else:
                    minimum = k
        
        while True:
            
            G = k_nearest_d(k, distances)
            if is_connected(G):
                maximum = k
            else:
                minimum = k
                
            if maximum - minimum == 1:
                k = maximum
                break
            
            k = minimum + (maximum - minimum) // 2
            
            print()
            print("minimal k, actual k, maximal k")
            print(minimum, k, maximum)
    
    if k < 100:
        k = int(1.33*k)+1
    
    print("Final k:", k)
    
    return k
        

def is_connected(G):
    
    cc = nk.components.ConnectedComponents(G)
    cc.run()
    return cc.numberOfComponents() == 1



def create_k_nearest_graph(distances, bridges, k=None):
    
    if k is None:
        k = find_k_d(distances)

    G = k_nearest_d(k, distances)

    if not is_connected(G):
        patch_together_d(G, position, bridges)
        
    return G, k

def create_k_nearest_graph_d(distances, bridges, k=None):
    
    if k is None:
        k = find_k_d(distances)

    G = k_nearest_d(k, distances)

    if not is_connected(G):
        raise RuntimeError("Not implemented")
#         patch_together_d(G, position, bridges)
        
    return G, k