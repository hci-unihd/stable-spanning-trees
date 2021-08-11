import networkit as nk
import numpy as np
from numpy.random import randint
from sklearn.cluster import KMeans
from numpy.linalg import norm
from IPython.display import clear_output
from copy import deepcopy
from time import time


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


def k_means_pp(k, pos, G=None, metric="euclidean", return_labels=False):
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
        
        print("sklearn is done:", time()-start)
        checkpoint = time()
        
        true_centers = []
        for center in centers:
            true_centers.append(np.argmin(norm(pos-center, axis=1)))
            
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
