import networkit as nk
import numpy as np
from numpy.random import randint
from sklearn.cluster import KMeans
from numpy.linalg import norm
from IPython.display import clear_output
from copy import deepcopy
from time import time 
from External.generation import find_backbone
import warnings
from sklearn.metrics import pairwise_distances



def centers(pos):
    """
    Convenience function for the standard of k_means_pp
    """
    number_of_nodes = len(pos)
    ε, δ = 0.03, 0.1
    r = 1 / (ε**2) * (int(np.log2(number_of_nodes - 1)) + 1 + np.log(1/δ))
    k = int(np.sqrt(r))
    k = np.min((k, number_of_nodes))
    
    return k_means_pp(k, pos)


def k_means_pp(k, pos, G=None, metric="euclidean", return_labels=False, log=False):
    '''
    Calculate k means ++ clustering. For the shortest path distance an APSP problem has to be solved -> slow
    
    Parameters
    ----------
    k : int
        The number of clusters
    pos : list or similar
        The positions of the data points. Needed for the distance computation
    metric : str
        The type of the distance measure (euclidean, euclidean_max or SP)
        
    Returns
    -------
    ndarray
        array with the projected cluster centers
    '''
    if metric == "euclidean":
        start = time()
        means = KMeans(n_clusters=k, init="k-means++").fit(pos)
        centers = means.cluster_centers_
        
        if log:
            print("sklearn is done:", time()-start)
        checkpoint = time()
        
        true_centers = []
        for center in centers:
            true_centers.append(np.argmin(norm(pos-center, axis=1)))
            
        if log:
            print("My own part is done:", time()-checkpoint)
            
        if return_labels:
            return np.array(true_centers), means.labels_
        
        return np.array(true_centers)

    elif metric == "SP":
        
        if return_labels:
            raise KeyError("Not implemented")
            
        
        def choose_k_initials(G, k, dist):

            def chances(G, dist, centers):
                
                probs = np.zeros(G.numberOfNodes())
                for node in G.iterNodes():
                    if node in centers:
                        continue
                    prob = np.inf
                    for center in centers:
                        prob = np.min((prob, dist.getDistance(node, center)))
                    probs[node] = prob
                return probs / np.sum(probs)
                    
            
            centers = []
            centers.append(randint(0, G.numberOfNodes()))
            nodes = list(range(G.numberOfNodes()))
            for i in range(k-1):
                clear_output(wait=True)
                print(i)
                prob = chances(G, dist, centers)
                new_center = np.random.choice(nodes, p=prob)
                centers.append(new_center)
            return np.array(centers)
            
            
        dist = nk.distance.APSP(G)
        dist.run()
        
        centers = choose_k_initials(G, k, dist)
        old_centers = np.zeros(k)
        
        clusters = [[] for _ in range(k)]
        iterations = 0
        while (centers != old_centers).any() and iterations < 10:
            iterations += 1
            for node in G.iterNodes():
                distance = np.inf
                center_idx = None
                for i, center in enumerate(centers):
                    d = dist.getDistance(node, center)
                    if distance > d:
                        cluster_idx = i
                        distance = d
                clusters[cluster_idx].append(node)
                
            for idx, cluster in enumerate(clusters):
                optimal_node = None
                min_maximal_distance = np.inf
                
                for node in cluster:
                    maximal_distance = 0
                    
                    for target in cluster:
                        maximal_distance = np.max((maximal_distance, dist.getDistance(node, target)))
                                                
                    if min_maximal_distance > maximal_distance:
                        optimal_node = node
                        min_maximal_distance = maximal_distance
                
                old_centers = deepcopy(centers)
                centers[idx] = optimal_node

        if iterations == 10:
            print("iteration limit reached")

        return np.array(centers)

    elif metric == "euclidean_max":

        means = KMeans(n_clusters=k, init="k-means++").fit(pos)
        centers = means.cluster_centers_
        labels = means.labels_

        clusters = [[] for _ in range(k)]
        for node in range(len(pos)):
            clusters[labels[node]].append(node)
        new_centers = []
        for cluster in clusters:
            max_distance = 0
            new_center = None
            for node in cluster:
                distance_to_all = []
                for center in centers:
                    distance_to_all.append(norm(center-pos[node]))
                    
                distance_to_all = np.sum(distance_to_all)
                
                if distance_to_all > max_distance:
                    max_distance = distance_to_all
                    new_center = node
            new_centers.append(new_center)

        if return_labels:
            return np.array(new_centers), labels
        
        return np.array(new_centers)

    else:
        raise KeyError("Not implemented")

def ICT_clusters(old_ICT, position, number_of_vertices, avg_cluster_len = 15, log=False):
    """
    Clusters are computed using the structure of the given ICT
    
    The size of the intersection of the backbone and the clusters is between avg_cluster_len and 2*avg_cluster_len if no warning occures
    
    """
    
    old_ICT = deepcopy(old_ICT)
    position = deepcopy(position)
    
    
    backbone, backbone_nodes = find_backbone(old_ICT, number_of_vertices=number_of_vertices)
    
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
            else:
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


        if size <= 2:
            warnings.warn("At least one ICT clusters was discarded!" , RuntimeWarning)
            # By doing nothing those points will be assigned to the nearest other cluster. A len 2 cluster does not seem useful!
            continue
            
        if size <= avg_cluster_len:
            warnings.warn("Small clustersizes occure. Results might not be as expected!" , RuntimeWarning)
            cluster_boundaries.append([start, end])
            continue

        
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

def cfilter(centers, labels, t=5, position_likes=None, labels_likes=None):
    """
    This function returns the clusters together with node labels. If t>0, all clusters with size len(cluster) < t will be reoved from the dataset an the position_like arrays
    will be adjusted accordingly. The labels likes will not be adjusted, but only cutted!
    """
    if position_likes is None:
        position_likes = []
        
    if labels_likes is None:
        labels_likes = []
    
    # backup the labels and the clusters
    centers, labels = deepcopy(centers), deepcopy(labels)
    unique_labels = np.unique(labels)
    
    # Create lists for each cluster contaning the related nodes
    components = []
    node_labels = [None for _ in range(len(labels))]
    
    for label in np.sort(unique_labels):
        component = np.argwhere(label == labels).T
        assert len(component) == 1

        for node in component[0]:
            node_labels[node] = label

        components.append(component[0].tolist())
    
    # Find the cluster indices of the to small clusters
    removal = []
    for idx, component in enumerate(components):
        if len(component) < t:
            removal.append(idx)
    
    # delete the too small centers
    new_centers = np.delete(centers, removal)
    
    # remove the old labels from the labels array
    position_removal = []
    labels_np = np.array(labels)
    
    for label in removal:
        position_removal += np.argwhere(label==labels_np).T[0].tolist()
    
    new_labels = np.delete(labels, position_removal)
    
    
    # adjust the position arrays
    new_position_likes = []
    for pos_like in position_likes:
        new_position_likes.append(np.delete(pos_like, position_removal, axis=0))
        
    # adjust the labels arrays
    new_labels_likes = []
    for lab_like in labels_likes:
        new_labels_likes.append(np.delete(lab_like, position_removal))
    
    
    # remove te gaps in the labels array (They sholud be the indices of the clusters array...):
    difference = [0 for i in range(len(unique_labels))]
    last_difference = 0
    
    for label in np.sort(unique_labels):
        if label in removal:
            last_difference += 1
            
        difference[label] = last_difference
        
    for idx, label in enumerate(new_labels):
        new_labels[idx] -= difference[label]
        
    
    # adjust the cluster centers to the fact, that the position array changed (they are the indices of it...):    
    pos_difference = [0 for i in range(len(labels))]
    last_difference = 0
      
    for idx in range(len(labels)):
        if idx in position_removal:
            last_difference += 1
            
        
        pos_difference[idx] = last_difference
        
    for idx, center in enumerate(new_centers):
        new_centers[idx] -= pos_difference[center]
        
        
        
    # Adjust the output to the input
    if len(new_position_likes) != 0:
        if len(new_labels_likes) != 0:
            return new_centers, new_labels, new_position_likes, new_labels_likes
        
        else:
            return new_centers, new_labels, new_position_likes
    
    elif len(new_labels_likes) != 0:
        return new_centers, new_labels, new_labels_likes
    
    return new_centers, new_labels


def cassign(centers, labels, position, t):
    
    number_of_nodes = len(labels)
    unique_labels = np.unique(labels)
    
    # Create the lists containing the nodes of each component
    components = []
    node_labels = [None for _ in range(number_of_nodes)]
    for label in np.sort(unique_labels):
        component = np.argwhere(label == labels).T
        assert len(component) == 1

        for node in component[0]:
            node_labels[node] = label

        components.append(component[0].tolist())
    
    node_labels = np.array(node_labels).astype(int)
    
    distances = pairwise_distances(position, position)

    cleaned_components = [False for _ in range(len(unique_labels))]

    
    while True:

        for idx, component in enumerate(components):

            if cleaned_components[idx] == True:
                continue

            if len(component) <= t:

                for node in component:
                    nearests = np.argsort(distances[node])
                    for elem in nearests:
                        if elem not in component:
                            nearest = elem
                            break
                    node_labels[node] = node_labels[nearest]
                    components[node_labels[nearest]].append(node)
                cleaned_components[idx] = True
                continue

        break
        
    new_components = []
    new_node_labels = [None for _ in range(number_of_nodes)]
    
    
    shift = 0
    for idx, component in enumerate(components):
        if cleaned_components[idx] == False:
            new_components.append(component)
            for node in component:
                new_node_labels[node] = idx - shift
        else:
            shift += 1
           
    new_centers = []
    
    for component in new_components:
        center = np.mean(position[component], axis=0)
        new_centers.append(np.argmin(np.linalg.norm(position-center, axis=1)))
    
    return new_centers, new_node_labels