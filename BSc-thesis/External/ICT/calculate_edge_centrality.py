from collections import deque
from numpy.random import randint
import networkit as nk
import numpy as np

from Stable_Tree.methods.Iterative.ICT_exact import Iterative_centrality_tree_adjacency0


def calculate_centrality_cluster_py(G, cluster_centers):
    '''
    Claculate the approximate edge centrality for the edges between the cluster centers.
    The other edge centralities are approximated as 0
    
    Parameters
    ----------
    G : networkit graph
        The corresponding graph 
    cluster_centers : list or similar
        a array containing the coordinates of the cluster centers
        must be a subset of the nodes of G

    Returns
    -------
    list
        a list of floats containing the computed centralities of the edges
    '''
    
    score = np.zeros(G.upperEdgeIdBound())
    sssp = nk.distance.Dijkstra(G, 0, True, False)
    for source in cluster_centers:
        sssp.setSource(source)
        sssp.run()
        for target in cluster_centers:
            if source==target:
                continue
            q = deque()
            if sssp.numberOfPaths(target)>0:
                for pred in sssp.getPredecessors(target):
                    q.append(pred)
                    score[G.edgeId(target, pred)]+=1
                while len(q) != 0:
                    node = q.pop()
                    for pred in sssp.getPredecessors(node):
                        q.append(pred)
                        score[G.edgeId(node, pred)]+=1
    return score


def calculate_centrality_cluster_c(G, cluster_centers):
    '''
    Claculate the approximate edge centrality for the edges between the cluster centers.
    The other edge centralities are approximated as 0. C++ implementation
    
    Parameters
    ----------
    G : networkit graph
        The corresponding graph 
    cluster_centers : list or similar
        a array containing the coordinates of the cluster centers
        must be a subset of the nodes of G

    Returns
    -------
    list
        a list of floats containing the computed centralities of the edges
    '''
    bc = nk.centrality.BA(G, cluster_centers)
    bc.run()
    return np.array(bc.edgeScores())


def calculate_centrality_cluster_all_py(G, cluster_centers):
    '''
    Claculate the approximate edge centrality for the edges between the cluster centers.
    The other edge centralities are approximated as 0
    
    Parameters
    ----------
    G : networkit graph
        The corresponding graph 
    cluster_centers : list or similar
        a array containing the coordinates of the cluster centers
        must be a subset of the nodes of G

    Returns
    -------
    list
        a list of floats containing the computed centralities of the edges
    '''
    
    score = np.zeros(G.upperEdgeIdBound())
    sssp = nk.distance.Dijkstra(G, 0, True, False)
    # for source in G.iterNodes():
    for source in cluster_centers:
        for target in G.iterNodes():
        # for target in cluster_centers:
            if source==target:
                continue
            sssp.setSource(source)
            sssp.setTarget(target)
            sssp.run()
            q = deque()
            if sssp.numberOfPaths(target)>0:
                for pred in sssp.getPredecessors(target):
                    q.append(pred)
                    score[G.edgeId(target, pred)]+=1
                while len(q) != 0:
                    node = q.pop()
                    for pred in sssp.getPredecessors(node):
                        q.append(pred)
                        score[G.edgeId(node, pred)]+=1
    return score


def calculate_centrality_cluster_all_c(G, cluster_centers):
    '''
    Claculate the approximate edge centrality for the edges between the cluster centers.
    The other edge centralities are approximated as 0. C++ implementation
    
    Parameters
    ----------
    G : networkit graph
        The corresponding graph 
    cluster_centers : list or similar
        a array containing the coordinates of the cluster centers
        must be a subset of the nodes of G

    Returns
    -------
    list
        a list of floats containing the computed centralities of the edges
    '''
    bc = nk.centrality.ClusterToAll(G, cluster_centers)
    bc.run()
    return np.array(bc.edgeScores())

def calculate_centrality_cluster_all2_c(G, cluster_centers, node_weights):
    '''
    Claculate the approximate edge centrality for the edges between the cluster centers.
    The other edge centralities are approximated as 0. C++ implementation
    
    Parameters
    ----------
    G : networkit graph
        The corresponding graph 
    cluster_centers : list or similar
        a array containing the coordinates of the cluster centers
        must be a subset of the nodes of G

    Returns
    -------
    list
        a list of floats containing the computed centralities of the edges
    '''
    bc = nk.centrality.ClusterToAll2(G, cluster_centers, node_weights)
    bc.run()
    return np.array(bc.edgeScores())



def calculate_centrality_exact(G):
    '''
    Claculate the exact edge centrality for all edges
    Parameters
    ----------
    G : networkit graph
        The corresponding graph

    Returns
    -------
    list
        a list of floats containing the computed centralities of the edges
    '''
    bc = nk.centrality.Betweenness(G, True, True)
    bc.run()
    return np.array(bc.edgeScores())


def calculate_centrality_RK_py(G, ε, δ):
    '''
    Calculate the centrality using the RK sampling strategy. VD is approximated by the number of nodes in the graph
    
    Parameters
    ----------
    G : networkit graph
        The corresponding graph 
    ε : float
        the ε value for the RK algorithm
    δ : float:
        the δ value for the RK algorithm

    Returns
    -------
    list
        a list of floats containing the computed centralities of the edges
    '''
    vd = G.numberOfNodes()
    r = int(1 / (ε**2) * (int(np.log2(vd - 1)) + 1 + np.log(1/δ)))

    score = np.zeros(G.numberOfEdges())
    
    sssp = nk.distance.Dijkstra(G, 0, True, False)
    
    for i in range(r):
            target = randint(0, G.numberOfNodes())
            source = randint(0, G.numberOfNodes())
            sssp.setSource(source)
            sssp.setTarget(target)
            sssp.run()
            t = target
            while t != source:
                decision = np.random.random()
                for z in sssp.getPredecessors(t):
                    weight = sssp.numberOfPaths(z) / sssp.numberOfPaths(t)
                    if weight >= decision:
                        break
                    decision -= weight
                edgeId = G.edgeId(z, t)
                score[edgeId] += 1 / r
                t = z
    return score


def calculate_centrality_RK_c(G, ε, δ):
    '''
    Calculate the centrality using the RK sampling strategy. VD is approximated by the number of nodes in the graph.
    Implemented in C++
    
    Parameters
    ----------
    G : networkit graph
        The corresponding graph 
    ε : float
        the ε value for the RK algorithm
    δ : float:
        the δ value for the RK algorithm

    Returns
    -------
    list
        a list of floats containing the computed centralities of the edges
    '''
    bc = nk.centrality.ApproxBetweennessEdges(G, ε, δ, 1.0, True)
    bc.run()
    return np.array(bc.edgeScores())

