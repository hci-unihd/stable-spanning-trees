from External.generation import find_backbone
import numpy as np
import warnings
from sklearn.metrics import pairwise_distances
from copy import deepcopy

def ICT_clusters(old_ICT, position, avg_cluster_len = 15, log=False):
    """
    Clusters are computed using the structure of the given ICT
    
    The size of the intersection of the backbone and the clusters is between avg_cluster_len and 2*avg_cluster_len if no warning occures
    
    """
    
    old_ICT = deepcopy(old_ICT)
    position = deepcopy(position)
    
    
    backbone, backbone_nodes = find_backbone(old_ICT, number_of_vertices=1)
    
    for node in backbone.iterNodes():
        if backbone.degree(node) == 1:
            endpoint = node
            break
            
    all_nodes = list(range(len(position)))
    done_vertex_neighbors = [[] for _ in range(len(position))]
    vertices = []
    branch_boundaries = []
    
    
    node = list(backbone.iterNeighbors(endpoint))[0]
    last = endpoint
    branch_boundaries.append([endpoint])
    len_counter = 1
    
    # find the pure branches
    if log:
        print("Find the branches")
    while True:
        
        neighbors = list(backbone.iterNeighbors(node))
        
        if len(neighbors) == 2:
            
            found_neighbor = False
            for neighbor in neighbors:
                if neighbor != last:
                    found_neighbor = True
                    break
                    
            assert found_neighbor
                    
            if len_counter == 0:
                branch_boundaries.append([node])
                
            last = node
            node = neighbor
            len_counter += 1
            
            continue
        
        
        if len(neighbors) == 1:
            if log:
                print("Found endpoint")
            
            branch_boundaries[-1].append(last)
            branch_boundaries[-1].append(node)
            branch_boundaries[-1].append(len_counter)
            len_counter = 0
            
            # Continue from another uncompleted vertex
            if len(vertices) != 0:
                vertex = vertices[0]
                if log:
                    print(f"Continue Search at node {vertex}")
                    vertex_neighbors = list(backbone.iterNeighbors(vertex))
                    print(f"with the neighbors {vertex_neighbors}. The neighbors {done_vertex_neighbors[vertex]} are allready finished.")
                
            else:
                break
            
            neighbors = list(backbone.iterNeighbors(vertex))
            
            i = 0
            
            for neighbor in neighbors:
                if neighbor not in done_vertex_neighbors[vertex]:
                    good_neighbor = neighbor
                    i += 1
                    
            assert i > 0
            if i == 1:
                vertices.pop(0)
                    
            done_vertex_neighbors[vertex].append(good_neighbor)
            last = vertex
            node = good_neighbor
            
            continue
            
            
        if len(neighbors) > 2:
            
            if log:
                print("Found vertex")
            
            if len_counter == 0:
                branch_boundaries.append([node, None, node, 1])
            
            branch_boundaries[-1].append(last)
            branch_boundaries[-1].append(node)
            branch_boundaries[-1].append(len_counter)
            len_counter = 0
            
            vertices.append(node)
            done_vertex_neighbors[node].append(last)
            
            found_neighbor = False
            for neighbor in neighbors:
                if neighbor != last:
                    found_neighbor = True
                    break
                    
            assert found_neighbor
            
            done_vertex_neighbors[node].append(neighbor)
            
            last = node
            node = neighbor
            
    cluster_boundaries = []
    
    if log:
        print("Create the cluster boundaries")
    
    for end, second, start, size in branch_boundaries:


        # I really do not expect this cases to happen...
        if size <= avg_cluster_len:
            warnings.warn("Small clustersizes occure. Results might not be as expected!" , RuntimeWarning)
            cluster_boundaries.append([start, end])
            continue

        if size < 4:
            raise RuntimeError("At lest one of the clusters is to small due to unfortunate branching! This is a problem of the ICT_cluster algorithm")

        
        rest = size % avg_cluster_len
        
        number_of_clusters = size // avg_cluster_len
        
        branch_cluster_sizes = [avg_cluster_len for i in range(number_of_clusters)]
        
        # Increase a random cluster by 1 element while assuring balancing
        permutation = np.random.permutation(number_of_clusters)
        
        index = 0
        while rest > 0:
            
            index = index % number_of_clusters
            
            branch_cluster_sizes[index] += 1
            
            index += 1
            rest -= 1
            
        branch_cluster_sizes = np.array(branch_cluster_sizes)[permutation]
        
        
        # add the cluster boundaries
        node = second
        last = start
        cluster_len = 1
        cluster_boundaries.append([start])
        
        cluster_index = 0
        
        while node != end:
            
            if cluster_len == 0:
                cluster_boundaries.append([node])
            
            neighbors = list(backbone.iterNeighbors(node))
            
            found_neighbor = False
            for neighbor in neighbors:
                if neighbor != last:
                    found_neighbor = True
                    break
                    
            assert found_neighbor
            
            cluster_len += 1
            
            if cluster_len == branch_cluster_sizes[cluster_index]:
                cluster_boundaries[-1].append(node)
                cluster_index += 1
                cluster_len = 0
                
            last = node
            node = neighbor
    
    # Assign each node to the cluster that contains the nearest boundary point:
    if log:
        print("Build the clusters")
        
        
    clusters = list(cluster_boundaries)
    if log:
        print(f"cluster boundaries: {cluster_boundaries}")
    flattenend_boundaries = np.array(cluster_boundaries).astype(int).flatten()
    
    rest = np.delete(all_nodes, flattenend_boundaries)
    
    distances = pairwise_distances(position[flattenend_boundaries], position[rest])
    
    # "//2" because of flattening
    indices = np.argmin(distances, axis=0) // 2

    # Create the clusters array
    for index, node in zip(indices, rest):
        clusters[index].append(node)
        
    # Create the labels and the centers arrays
    cluster_labels = np.zeros(len(position))
    centers = []
    
    for label, cluster in enumerate(clusters):
        
        centers.append(np.mean(position[cluster], axis=0))
        
        for node in cluster:
            cluster_labels[node] = label

        
        
    
    
        
        
    true_centers = []
    for center in centers:
        true_centers.append(np.argmin(np.linalg.norm(position-center, axis=1)))
        
    return np.array(true_centers), cluster_labels