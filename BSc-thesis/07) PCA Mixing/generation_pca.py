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

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



def create_graph(number_of_nodes, mode, position, gamma=1, k=None, beta=2, delta=0.5):
    
    assert number_of_nodes == len(position)
            
    if mode == "PCA-k-nearest-only":
        
        position = StandardScaler().fit_transform(position)
        
        pca = PCA(n_components=100).fit_transform(position)
        
        G, k = create_k_nearest_graph(pca, k)

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
            

            w_real_space = distances[u,v]
            
            w_new = w_real_space * ((u_diff * v_diff) / (common * common))**delta
            G.setWeight(u, v, w_new)
        return G, position
    
    elif mode == "PCA-distance-only":
        
        position = StandardScaler().fit_transform(position)
        
        pca = PCA(n_components=100).fit_transform(position)
        
        G, k = create_k_nearest_graph(position, k)
         
        # use (beta) times the average distance between two connected nodes as the radius for the density
        radius = beta * G.totalEdgeWeight()/(G.numberOfEdges())
        
        distances = pairwise_distances(pca, pca, n_jobs=-1)
        
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
            
            
            w_pca = distances[u,v]
            w_new = w_pca * ((u_diff * v_diff) / (common * common))**delta
            G.setWeight(u, v, w_new)
        return G, position
    
    elif mode == "PCA-density-only":
        
        position = StandardScaler().fit_transform(position)
        
        pca = PCA(n_components=100).fit_transform(position)
        
        G, k = create_k_nearest_graph(position, k)
         
        # use (beta) times the average distance between two connected nodes as the radius for the density
        radius = beta * G.totalEdgeWeight()/(G.numberOfEdges())
        
        distances = pairwise_distances(pca, pca, n_jobs=-1)
        
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

            w_new = w * ((u_diff * v_diff) / (common * common))**delta
            G.setWeight(u, v, w_new)
        return G, position
    
    elif mode == "PCA":
        
        position = StandardScaler().fit_transform(position)
        
        pca = PCA(n_components=100).fit_transform(position)
        
        G, k = create_k_nearest_graph(pca, k)
         
        # use (beta) times the average distance between two connected nodes as the radius for the density
        radius = beta * G.totalEdgeWeight()/(G.numberOfEdges())
        
        distances = pairwise_distances(pca, pca, n_jobs=-1)
        
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
            
            
            w_pca = distances[u,v]
            w_new = w_pca * ((u_diff * v_diff) / (common * common))**delta
            G.setWeight(u, v, w_new)
        return G, position
    
    
    else:
        raise KeyError("Not implemented")




                