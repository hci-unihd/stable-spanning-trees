from numpy.random import uniform
from numpy.linalg import norm
from scipy.spatial import Delaunay
from tqdm.notebook import tqdm

from sklearn.neighbors import kneighbors_graph
# from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances

from Stable_Tree.utils.utils import break_triangle_inequality
from Stable_Tree.graph_generation.graph_generation import Complete_graph, k_nearest_neighbor_graph

import numpy as np
import networkit as nk
import time

from External.create_k_nearest import create_k_nearest_graph



def create_graph(number_of_nodes, mode="Delaunay", position=None, gamma=1, k=None, beta=None, delta=2, bridges=5):
    """
    Creates the graph for the experiments.
    
    Parameters
    ----------
    number_of_nodes : int
        the number of nodes in the graph
    mode : str
        The type of generating algorithm (Delaunay, Full, Full+Exp-Triangle, K_Nearest, K_Nearest+Triangle, K_Nearest+Density)
    position=None : ndarray
        Optional; The initial position of the Graph nodes. Will be sampled uniformly if not given
    gamma=1 : number
        Optional; Parameter for the triangle break
    k=None : int
        Optional; Number of the nearest neighbors if this alg is chosen. If not given the lowest k for a connected graph is
        calculated and this value times 1.33 is chosen
    beta=None : float
        Optional; default value is k; On average, how many points should be considered in the density
    delta=2 : float
        Optional; How strong should the density be weighted. 0 -> not considered, 1 -> linearly, 2 -> squared, etc...
        
    Returns
    -------
    ndarray
        array with the projected cluster centers
    """
    
    # Position of the nodes
    if position is None:
        position = uniform(0,1,(number_of_nodes,2))
        position[:,0] *= 4
    else:
        assert number_of_nodes == len(position)
            
    if mode == "Delaunay":
        if len(position[0]) != 2:
            raise RuntimeError("Delaunay is only suited for 2D")
        # create the graph
        G = nk.graph.Graph(weighted = True)
        G.addNodes(number_of_nodes)
        number_of_edges = 0
        tri = Delaunay(position)

        for triangle in tri.simplices:
            for i in range(3):
                u, v = i, (i+1) % 3
                if not G.hasEdge(triangle[u], triangle[v]):
                    G.addEdge(triangle[u], triangle[v], norm(position[triangle[u]] - position[triangle[v]]))

                    # upper bound
                    number_of_edges += 1

        return G, position
    
    elif mode == "Full":
        
        distances = pairwise_distances(position, position, n_jobs=-1)
        
        
        G=nk.Graph(n=len(position),weighted=True)
        for u in tqdm(range(len(position)), desc="Create the full graph"):
            for v in range(u):
                G.addEdge(u,v,distances[u, v])

        return G, position
    
    elif mode == "Full+Exp-Triangle":
        
        distances = pairwise_distances(position, position, n_jobs=-1)
        
        exp_distances = np.exp(gamma*distances) - 1
        
        
        G=nk.Graph(n=len(position),weighted=True)
        for u in tqdm(range(len(position)), desc="Create the full graph"):
            for v in range(u):
                G.addEdge(u,v,exp_distances[u, v])

        return G, position
    
    elif mode == "K_Nearest":
        
        G, _ = create_k_nearest_graph(position, bridges, k)
            
                
        return G, position
    
    elif mode == "K_Nearest+Triangle":
        
        G, _ = create_k_nearest_graph(position, bridges, k)
        
        # Apply Enric's triangle break
        break_triangle_inequality(G, function='piecewise_exp_squared', gamma=gamma)
        
        return G, position
    
    elif mode == "K_Nearest+Density":
        
        G, k = create_k_nearest_graph(position, bridges, k)
        
        if beta is None:
            beta = k
         
        # use (beta) times the average distance between two connected nodes as the radius for the density
        radius = beta * G.totalEdgeWeight()/(G.numberOfEdges())

        density = np.zeros(G.upperNodeIdBound())
        
        distances = pairwise_distances(position, position, n_jobs=-1)
        
        for u in tqdm(G.iterNodes(), desc="calculating the densities for the density criterion",
                      total=G.numberOfNodes()):
            dist = distances[u]
            numbers = np.sum(dist<=radius)
            density[u] = numbers**delta
        for u, v, w in tqdm(G.iterEdgesWeights(), desc="updating the weights", total=G.numberOfEdges()):
            w_new = w / (density[u]*density[v])
            G.setWeight(u, v, w_new)
        return G, position
    
    elif mode == "K_Nearest+Density2":
        
        G, k = create_k_nearest_graph(position, bridges, k)
        
        if beta is None:
            beta = k
         
        # use (beta) times the average distance between two connected nodes as the radius for the density
        radius = beta * G.totalEdgeWeight()/(G.numberOfEdges())
        
        distances = pairwise_distances(position, position, n_jobs=-1)
        
        neighbors = [[] for _ in range(G.upperNodeIdBound())]
        
        for u in tqdm(G.iterNodes(), desc="calculating the densities for the density criterion",
                      total=G.numberOfNodes()):
            
            dist = distances[u]
            neighbors_mask = (dist<=radius)
            neighbors[u] = np.argwhere(neighbors_mask)

        for u, v, w in tqdm(G.iterEdgesWeights(), desc="updating the weights", total=G.numberOfEdges()):
            
            common = len(np.intersect1d(neighbors[u], neighbors[v], assume_unique=True))
            u_diff = len(neighbors[u]) - common
            v_diff = len(neighbors[v]) - common
            
            # prevent division by zero
            common += 1
            
            # prevent zero weights
            u_diff += 1
            v_diff += 1
            
#             print(common, u_diff, v_diff)
            
            w_new = w * ((u_diff * v_diff) / (common * common))**delta
            G.setWeight(u, v, w_new)
        return G, position
    
    else:
        raise KeyError("Not implemented")


def arr_to_dict(arr):
    dct = {}
    for idx, elem in tqdm(enumerate(arr), desc="Create a dict from the position array"):
        dct[idx] = elem
    return dct




                