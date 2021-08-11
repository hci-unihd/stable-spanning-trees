import networkit as nk
import networkx as nx
import numpy as np
from copy import deepcopy
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph

def reweight_boundaries(position, G, cluster_centers, cluster_labels, distance_threshold, min_scale = 0.3, max_scale=1.1):
    
    cluster_labels = np.array(cluster_labels)
    unique_labels = np.unique(cluster_labels)
    
    structured_components = []
    
    for label in np.sort(unique_labels):
        structured_components.append(np.argwhere(cluster_labels == label).T[0])
    
    all_nodes = list(range(len(position)))
    
    real_distance_threshold = distance_threshold * G.totalEdgeWeight()/(G.numberOfEdges())
    
    
    
    # Apply a distance based filtering to receive the boundaries
    second_filter = []
    second_filter_label = []
    for component in structured_components:
        
        if len(component) == 0:
            continue
        
        rest = np.delete(all_nodes, component)
        
        distances = pairwise_distances(position[component], position[rest])
        
        min_dist = np.min(distances, axis=1)
        min_dist_idx = np.argmin(distances, axis=1)
        
        for idx, node in enumerate(component):
            if min_dist[idx] <= real_distance_threshold:
                
                second_filter.append(node)
                second_filter_label.append(cluster_labels[rest[min_dist_idx[idx]]])
                
    
    # Mark the boundaries with the labels of the to siding clusters
    boundaries = dict()
    for idx, node in enumerate(second_filter):
        i = cluster_labels[node]
        j = second_filter_label[idx]
        
        if i > j:
            key = str((j,i))
        else:
            key = str((i,j))
            
        if key in boundaries:
            boundaries[key].append(node)
        else:
            boundaries[key] = []
            boundaries[key].append(node)
    
    boundary_labels = [-1 for _ in range(len(all_nodes))]
    boundary_centers = list(range(len(boundaries)))
    max_dist_from_center = np.zeros(len(boundaries))
    
    
    # Calculate the distance in the boundaries to their centers
    for boundary_label, key in enumerate(boundaries):
        
        boundary_centers[boundary_label] = np.mean(position[boundaries[key]], axis=0)
        
        for node in boundaries[key]:
            boundary_labels[node] = boundary_label
            
            if max_dist_from_center[boundary_label] < np.linalg.norm(boundary_centers[boundary_label] - position[node]):
                max_dist_from_center[boundary_label] = np.linalg.norm(boundary_centers[boundary_label] - position[node])
            
            
    boundary_labels = np.array(boundary_labels)
        
    
    # Make the weights near the centers of the glue point clusters lower
    
    node_weights =  [None for _ in range(len(all_nodes))]
    
    for boundary_label, key in enumerate(boundaries):
        
        if max_dist_from_center[boundary_label] == 0:
            for node in boundaries[key]:
                node_weights[node] = 1
            continue
        
        for node in boundaries[key]:
            
            
            # extrapolate between 0.5 and 1.1 * old weight
            node_weights[node] = (np.linalg.norm(boundary_centers[boundary_label] - position[node]) / max_dist_from_center[boundary_label]) * (max_scale-min_scale) + min_scale

            
    for u, v, w in G.iterEdgesWeights():
        
        if (boundary_labels[u] == -1) or (boundary_labels[v] == -1):
            continue
            
        elif boundary_labels[u] == boundary_labels[v]:
            G.setWeight(u, v, w*node_weights[u]*node_weights[v])
    
                
    return G, np.array(second_filter), np.array(boundary_centers), np.array(boundary_labels)


def reweight_clusters(position, G, cluster_centers, cluster_labels, min_scale = 0.3, max_scale=1.1):
    
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
    for node in all_nodes:
        label = cluster_labels[node]      
        node_weights[node] = (np.linalg.norm(position[cluster_centers[label]] - position[node]) / max_dist_from_center[label]) * (max_scale-min_scale) + min_scale

            
    for u, v, w in G.iterEdgesWeights():
        
        if cluster_labels[u] == cluster_labels[v]:
            G.setWeight(u, v, w*node_weights[u]*node_weights[v])
    
                
    return G