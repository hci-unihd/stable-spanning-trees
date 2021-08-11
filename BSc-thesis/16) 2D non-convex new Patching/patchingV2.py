import networkit as nk
import networkx as nx
import numpy as np
from copy import deepcopy
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph

def patch_together(ICT_forest, position, centrality_threshold=1.5, distance_threshold=1, k=5):
    
    real_distance_threshold = distance_threshold * ICT_forest.totalEdgeWeight()/(ICT_forest.numberOfEdges())
    
    all_nodes = list(ICT_forest.iterNodes())
    
    bc = nk.centrality.Betweenness(ICT_forest, False, True)
    bc.run()  
    score = np.array(bc.edgeScores()) / len(all_nodes)

    
    # Apply a first, centality based filtering
    first_filter = set()
    for u, v in ICT_forest.iterEdges():
        if score[ICT_forest.edgeId(u, v)] <= centrality_threshold:
            first_filter.add(u)
            first_filter.add(v)
    if len(first_filter) == 0:
        return [], []
    
    
    # Extract the components from the ICT forest
    first_filter = np.sort(list(first_filter)).astype(int)
    cc = nk.components.ConnectedComponents(ICT_forest)
    cc.run()
    components = cc.getComponents()
    labels = [None for _ in range(len(all_nodes))]
    for idx, component in enumerate(components):
        for u in component:
            labels[u] = idx
    
    labels = np.array(labels)
    
    
    # Reduce the components to the points in the first filter
    filtered_components = []
    for idx in range(len(components)):
        filtered_components.append(first_filter[np.argwhere(labels[first_filter] == idx).T[0]])
    
    
    
    # Apply a second, distance based filtering
    second_filter = []
    second_filter_label = []
    for i, component in enumerate(filtered_components):
        
        if len(component) == 0:
            continue
        
        rest = np.delete(all_nodes, components[i])
        
        distances = pairwise_distances(position[component], position[rest])
        
        min_dist = np.min(distances, axis=1)
        min_dist_idx = np.argmin(distances, axis=1)
        
        for idx, node in enumerate(component):
            if min_dist[idx] <= real_distance_threshold:
                
                second_filter.append(node)
                second_filter_label.append(labels[rest[min_dist_idx[idx]]])
                
    
    boundaries = dict()
    for idx, node in enumerate(second_filter):
        i = labels[node]
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
    
    for boundary_label, key in enumerate(boundaries):
        
        boundary_centers[boundary_label] = np.mean(position[boundaries[key]], axis=0)
        
        for node in boundaries[key]:
            boundary_labels[node] = boundary_label
            
            if max_dist_from_center[boundary_label] < np.linalg.norm(boundary_centers[boundary_label] - position[node]):
                max_dist_from_center[boundary_label] = np.linalg.norm(boundary_centers[boundary_label] - position[node])
            
            
    boundary_labels = np.array(boundary_labels)
        
                
    # create knn graph on the "glue" points            
    kn_matrix = kneighbors_graph(position[second_filter], n_neighbors=k, mode='distance')
    kn = nx.from_scipy_sparse_matrix(kn_matrix)
    kn = nk.nxadapter.nx2nk(kn, weightAttr="weight")
    G = nk.Graph(n=len(position), weighted=True)

    
    # Make sure that the knn graph does not connect multiple boundaries and make the weights near the centers of the glue point clusters lower
    
    node_weights =  [None for _ in range(len(all_nodes))]
    
    for boundary_label, key in enumerate(boundaries):
        
        for node in boundaries[key]:
            
            
            # extrapolate between 0.6 and 1.4 * old weight
            node_weights[node] = (np.linalg.norm(boundary_centers[boundary_label] - position[node]) / max_dist_from_center[boundary_label]) * 0.8 + 0.6

            
    
    for u, v, w in kn.iterEdgesWeights():
        if boundary_labels[second_filter[u]] == boundary_labels[second_filter[v]]:
            kn.setWeight(u, v, w*node_weights[second_filter[u]]*node_weights[second_filter[v]])
            
            
        else:
            kn.removeEdge(u, v)
    
    
    
    
    # needed for the creation of the good edges array
    forest2 = deepcopy(ICT_forest)
    for u in kn.iterNodes():
        for v in kn.iterNodes():
            if forest2.hasEdge(second_filter[u], second_filter[v]):
                forest2.removeEdge(second_filter[u], second_filter[v])
    
    good_edges = []
    for u, v in forest2.iterEdges():
        good_edges.append([u,v])
    
    
    # Add the knn edges to the final graph
    for u, v, w in kn.iterEdgesWeights():
        G.addEdge(second_filter[u], second_filter[v], w)
    
    
    # Add the ICT forest edges to the final graph
    for u, v, w in ICT_forest.iterEdgesWeights():
        if not G.hasEdge(u, v):
            G.addEdge(u, v, w)
                
    return G, first_filter, np.array(second_filter), good_edges, np.array(boundary_labels)
        
        
        
        