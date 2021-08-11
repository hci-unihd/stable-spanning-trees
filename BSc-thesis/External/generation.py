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
from External.create_k_nearest_distances import create_k_nearest_graph_d
from External.k_means_pp import k_means_pp

import matplotlib.pyplot as plt
import random


def create_graph(number_of_nodes, mode="Delaunay", position=None, gamma=1, k=None, beta=None, delta=2, bridges=5, n_clusters=None, min_scale=0.3, max_scale=1.1, return_distance_array=False, return_clusters = False):
    """
    Creates the graph for the experiments.
    
    Parameters
    ----------
    number_of_nodes : int
        the number of nodes in the graph
    mode : str
        The type of generating algorithm (Delaunay, Full, Full+Exp-Triangle, K_Nearest, K_Nearest+Triangle, K_Nearest+Density, K_Nearest+Density2,
        K_Nearest+Density2+Recompute)
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
    
    elif mode == "K_Nearest+Density+Recompute":
        
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
            distances[u] /= numbers**delta
            distances[:,u] /= numbers**delta
            
        G, _ = create_k_nearest_graph_d(distances, bridges, k)
        
        if return_distance_array == True:
            return G, position, distances
        
        return G, position
    
    
    elif mode == "K_Nearest+Density2+Recompute":
        
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

        for u, row in tqdm(enumerate(distances), desc="updating the distance array", total=len(distances)):
            for v, distance in enumerate(row):
                # symmetric density array!
                # v == u -> distance is and stays 0
                if v <= u:
                    continue
                    
                if distance > 2 * radius:
                    # We dont need to compute the intersection; It is empty!
                    x = ( (len(neighbors[u]) + 1) * (len(neighbors[v]) + 1) )**delta
                    distances[u, v] *= x
                    distances[v, u] *= x
                    continue
            
            
                common = len(np.intersect1d(neighbors[u], neighbors[v], assume_unique=True))
                u_diff = len(neighbors[u]) - common
                v_diff = len(neighbors[v]) - common

                # prevent division by zero
                common += 1

                # prevent zero weights
                u_diff += 1
                v_diff += 1


                x = ((u_diff * v_diff) / (common * common))**delta
                distances[u, v] *= x
                distances[v, u] *= x
            
        G, _ = create_k_nearest_graph_d(distances, bridges, k)
        
        if return_distance_array == True:
            return G, position, distances
        
        return G, position
    
    elif mode == "K_Nearest+Cluster+Recompute":
        
        distances = pairwise_distances(position, position, n_jobs=-1)
        
        cluster_centers, cluster_labels = k_means_pp(n_clusters, position, metric="euclidean", return_labels=True)
        
        cluster_labels = np.array(cluster_labels)
        unique_labels = np.unique(cluster_labels)

        all_nodes = list(range(len(position)))


        # Calculate the maximum distance in the clusters to their centers (=radius of the cluster)
        max_dist_from_center = np.zeros(len(cluster_centers))
        for node in all_nodes:
            label = cluster_labels[node]
            if max_dist_from_center[label] < np.linalg.norm(position[cluster_centers[label]] - position[node]):
                max_dist_from_center[label] = np.linalg.norm(position[cluster_centers[label]] - position[node])


        # Make the weights near the centers of the clusters lower
        node_weights =  [None for _ in range(len(all_nodes))]

        # extrapolate between min_scale and max_scale
        for node in tqdm(all_nodes, desc="Updating the distance array by using k means clusters"):
            label = cluster_labels[node]      
            if max_dist_from_center[cluster_labels[node]] == 0:
                node_weights[node] = 1
            else:
                node_weights[node] = (np.linalg.norm(position[cluster_centers[label]] -
                                                     position[node]) / max_dist_from_center[label]) * (max_scale-min_scale) + min_scale
            
            distances[node] *= node_weights[node]**delta
            distances[:,node] *= node_weights[node]**delta
            
        
        G, _ = create_k_nearest_graph_d(distances, bridges, k)
            
        if return_clusters:
            return G, position, np.array(cluster_labels)
            
        return G, position
         
            
    
    
    else:
        raise KeyError("Not implemented")


def arr_to_dict(arr):
    dct = {}
    for idx, elem in tqdm(enumerate(arr), desc="Create a dict from the position array"):
        dct[idx] = elem
    return dct


def load_image(filename):
    a=plt.imread(filename+'.png')
    rgb_weights = [0.2989, 0.5870, 0.1140]
    
    grayscale_image = np.dot(a[...,:3], rgb_weights)>0
    
    return grayscale_image
    
    
def sample_points_from_image(n,img,Random=True):
    if not Random:
        random.seed(42)
    non_zero = np.where(img!=0)
    # non_zero=np.vstack((non_zero[0],non_zero[1])).T
    
    
    idx = random.sample(range(len(non_zero[0])),n)
    
    x_coord = non_zero[0][idx]
    y_coord = non_zero[1][idx]
    return x_coord,y_coord


from copy import deepcopy
import operator

def find_backbone(old_ICT, number_of_vertices=0):
    
    #TODO for many vertices min_max_priority queue better...
    
    high_vertices = [[0, None, None]]
    backbone_nodes = set()
    
    ICT = deepcopy(old_ICT)
    
    initials = [0,0]

    Backbone = nk.Graph(n=ICT.numberOfNodes(), weighted=True)

    bc = nk.centrality.Betweenness(ICT, normalized=True, computeEdgeCentrality=True)
    bc.run()
    ICT_centralities = bc.edgeScores()

    seed = np.argmax(ICT_centralities)
    seed

    for u, v in ICT.iterEdges():
        eid = ICT.edgeId(u, v)
        ICT.setWeight(u, v, ICT_centralities[eid])
        if eid == seed:
            initials[0] = u
            initials[1] = v
            Backbone.addEdge(u, v, ICT_centralities[eid])
            backbone_nodes.add(u)
            backbone_nodes.add(v)

    # Go the tree up and down
    for i in range(2):
        node = initials[i]
        last = initials[(i+1)%2]
        while True:
            max_w = 0

            neighbors = list(ICT.iterNeighbors(node))

            # find the edge that belongs to the backbone as well
            for neighbor in neighbors:
                w = ICT.weight(node, neighbor)
                if w > max_w and neighbor != last:
                    max_w = w
                    keep = neighbor

            Backbone.addEdge(node, keep, max_w)
            backbone_nodes.add(keep)

            # Mark the largest vertices
            for neighbor in neighbors:
                if (neighbor != last) and (neighbor != keep):
                    vertex_weight = ICT.weight(node, neighbor)
                    if number_of_vertices > 0 and vertex_weight > high_vertices[-1][0]:
                        high_vertices.append([vertex_weight, node, neighbor])
                        high_vertices.sort(key = operator.itemgetter(0), reverse=True)
                    if len(high_vertices) > number_of_vertices:
                        high_vertices.pop()

            last = node
            node = keep

            if ICT.degree(node)==1:
                break
            if ICT.degree(last)==1:
                break
         
        
        
    while(number_of_vertices > 0):
        number_of_vertices -= 1
        max_w, last, node = high_vertices.pop(0)
        Backbone.addEdge(last, node, max_w)
        backbone_nodes.add(node)
        
        
        while True:
            max_w = 0

            neighbors = list(ICT.iterNeighbors(node))

            # find the edge that belongs to the backbone as well
            for neighbor in neighbors:
                w = ICT.weight(node, neighbor)
                if w > max_w and neighbor != last:
                    max_w = w
                    keep = neighbor

            Backbone.addEdge(node, keep, max_w)
            backbone_nodes.add(keep)

            # Mark the largest vertices
            for neighbor in neighbors:
                if (neighbor != last) and (neighbor != keep):
                    vertex_weight = ICT.weight(node, neighbor)
                    if  number_of_vertices > 0 and vertex_weight > high_vertices[-1][0]:
                        high_vertices.append([vertex_weight, node, neighbor])
                        high_vertices.sort(key = operator.itemgetter(0), reverse=True)
                    if len(high_vertices) > number_of_vertices:
                        high_vertices.pop()

            last = node
            node = keep

            if ICT.degree(node)==1:
                break
            if ICT.degree(last)==1:
                break
    

    return Backbone, np.sort(list(backbone_nodes))