import networkit as nk
import numpy as np

from IPython.display import clear_output
from copy import deepcopy

from Stable_Tree.methods.Iterative.ICT_exact import Iterative_centrality_tree_adjacency0
from External.ICT.calculate_edge_centrality import calculate_centrality_cluster_py, calculate_centrality_cluster_c, calculate_centrality_exact, calculate_centrality_RK_py, calculate_centrality_RK_c
from External.ICT.calculate_edge_centrality import calculate_centrality_cluster_all_py, calculate_centrality_cluster_all_c
import matplotlib.pyplot as plt

from itertools import chain
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from numpy.linalg import norm

from tqdm.notebook import tqdm

def create_ICT(good_edges, w, G):
    '''
    Create the ICT from a list of good edges. The original graph is needed for the number of Nodes only
    
    Parameters
    ----------
    good_edges : iterable
        List with the edges of the ICT
        
    w : float
        weights of the edges of the ICT
        
    G : networkit graph
        the original graph
        
    Returns
    -------
    networkit.graph
        The ICT constructed by using the List good_edges
    '''
    
    G_tree = nk.graph.Graph(weighted = True)
    G_tree.addNodes(G.numberOfNodes())
    for edge in good_edges:
        G_tree.addEdge(*edge, w)

    return G_tree


def compute_widths(G):
    '''
    compute the width of the edges for the plotting algorithm using the betweenness centrality of the edges
    
    Parameters
    ----------
    G : networkit graph
        The graph for which the width should be computed
        
    Returns
    -------
    python list
        a list containing the widths
    '''
    bc = nk.centrality.Betweenness(G, False, True)
    bc.run()  
    score = np.array(bc.edgeScores())
    score /= np.max(score)
    widths = []
    for edge in G.iterEdges():
        widths.append(np.sqrt(score[G.edgeId(*edge)])*12)
    return widths


def find_best_edge(G, w, edgeId_to_edge, algorithm_type, cluster_centers=None, ε=None, δ=None):
    
    '''
    Find the egde with the largest centrality. If there is a tie choose the shortest one.
    The weight of the chosen edge is forced to be larger than w
    
    Parameters
    ----------
    G : networkit graph
        The corresponding graph 
    w : float
        the value to which the corresponding edges weight should be set
    edgeId_to_edge : list or similar
        an array containing the transforamtion from edge ID to (u,v) notation
    algorithm_type : str
        The type of the centrality algorithm (exact_own, RK, RK_py, cluster, cluster_py)
    cluster_centers : list or similar
        optional, an array containing the coordinates of the cluster centers
        must be a subset of the nodes of G    
    ε : float
        optional, the ε value for the RK algorithm
    δ : float
        optional, the δ value for the RK algorithm
        
    Returns
    -------
    int
        the id of the edge, whichs weight is larger than epsilon, with the maximum centrality
    list
        a list of floats containing the computed centralities of the edges
    '''
   
    # calculate the betweeness of all edges
    if algorithm_type == "exact_own":
        scores = calculate_centrality_exact(G)
    if algorithm_type == "cluster_py":
        scores = calculate_centrality_cluster_py(G, cluster_centers)
    if algorithm_type == "cluster":
        scores = calculate_centrality_cluster_c(G, cluster_centers)
    if algorithm_type == "cluster_all_py":
        scores = calculate_centrality_cluster_all_py(G, cluster_centers)
    if algorithm_type == "cluster_all":
        scores = calculate_centrality_cluster_all_c(G, cluster_centers)
    if algorithm_type == "RK":
        scores = calculate_centrality_RK_c(G, ε, δ)
    if algorithm_type == "RK_py":
        scores = calculate_centrality_RK_py(G, ε, δ)
    edge_id_sorted = np.array(scores).argsort()
    

    # find the best edge
    best = None
    min_score = 0

    for edge_id in edge_id_sorted[::-1]:
        if G.weight(*edgeId_to_edge[edge_id]) > w:
            score = scores[edge_id]
            if score < min_score:
                break
            min_score = score
            if (best is None):
                best = edge_id
            if G.weight(*edgeId_to_edge[edge_id]) < G.weight(*edgeId_to_edge[best]):
                best = edge_id

    if best is not None:
        edge_id = best
               
    
    return int(edge_id), scores


def calculate_edgeId_to_edge(G):
    '''
    Calculate the edgeId_to_edge array
    
    Parameters
    ----------
    G : networkit.graph
        The corresponding graph
    
    Returns
    -------
    ndarray
        the required array
    '''

    edgeId_to_edge = np.zeros((G.upperEdgeIdBound(),2))
    for edge in tqdm(G.iterEdges(), desc="create edgeId array", total=G.numberOfEdges()):
        edgeId_to_edge[G.edgeId(*edge)] = edge

    return edgeId_to_edge


def calculate_w(G):
    '''
    Calculate the lower weight for the BC steps
    
    Parameters
    ----------
    G : networkit.graph
        The corresponding graph
    
    Returns
    -------
    float
        The queried lower weight
    '''
    
    w_min = np.inf
    for u,v,w in tqdm(G.iterEdgesWeights(), desc="Calculate the lower bound for the weights", total=G.numberOfEdges()):
        if w<w_min:
            w_min = w

    return (w_min/G.numberOfNodes())/4


def calculate_ICT(G, algorithm_type, cluster_centers=None, ε=0.03, δ=0.1, zeros_stay_zeros=False,
                  update_G=10, create_snaps=False, position=None, folder_name=None):

    '''
    Calculate the ICT using the cluster algorithm
    
    Parameters
    ----------
    G : networkit graph
        The corresponding graph 
    algorithm_type : str
        The type of the centrality algorithm (exact_own, RK, RK_py, cluster, cluster_py)
    cluster_centers=None : list or similar
        optional, an array containing the coordinates of the cluster centers
        must be a subset of the nodes of G
    ε=0.03 : float
        optional, the ε value for the RK algorithm
    δ=0.1 : float:
        optional, the δ value for the RK algorithm
    zeros_stay_zeros=False: bool
        optional, is it possible to remove zero centrality edges? Depends on the algorithm_type (is it deterministic?)
    update_G=10 : number
        optional, the factor that decidest, when the working-G and the edgeId_to_edge arr will be recomputed (=1 for always).
        Must not be smaller than 1
    create_snaps=False : bool
        optional, should the algorithm save a image of the graph after every iteration
    position=None : ndarray
        optional, The position of the node of the graph. Needed is create_snaps=True!
    folder_name=None : str
        optional, the path where potential snapshots should be saved
        
        
    Returns
    -------
    networkit.graph()
        the ICT
    '''

    def create_new_graph(G):
        G_new = nk.graph.Graph(weighted = True)
        G_new.addNodes(G.numberOfNodes())
        for u,v,w in G.iterEdgesWeights():
            G_new.addEdge(u,v,w)
        G_new.indexEdges()
        return G_new

    def add_sure_edges(G, good_edges, w):
        found_edges = 0
        for u in G.iterNodes():
            if G.degree(u) == 1:
                # only one neighbor
                for v in G.iterNeighbors(u):
                    if G.weight(u, v) > w:
                        good_edges.append([u, v])
                        G.setWeight(u, v, w)
                        found_edges += 1
        if found_edges > 1:
            print(f"added {found_edges} edges early")
        return found_edges
    
    def add_remaining_edges(G, good_edges, w):
        '''
        Add all edges from G to the ICT
        '''
        found_edges = 0
        for u in G.iterNodes():
            for v in G.iterNeighbors(u):
                if G.weight(u, v) > w:
                    good_edges.append([u, v])
                    G.setWeight(u, v, w)
                    found_edges += 1
        if found_edges > 1:
            print(f"added {found_edges} edges early")
        return found_edges

    def calculate_edgeId_to_edge_no_bar(G):

        edgeId_to_edge = np.zeros((G.upperEdgeIdBound(),2))
        for edge in G.iterEdges():
            edgeId_to_edge[G.edgeId(*edge)] = edge

        return edgeId_to_edge

    
    
    # Exact: use enrics implementation, faster
    if algorithm_type == "exact":
        return Iterative_centrality_tree_adjacency0(G,Added_edges_per_iteration=1)[1]

    good_edges = []
    skipped = 0

    # Translation array
    edgeId_to_edge = calculate_edgeId_to_edge(G)

    # Weight of the "highway"
    w = calculate_w(G)

    G_backup = deepcopy(G)

    # iterate over all nodes
    for i in tqdm(range(G_backup.numberOfNodes()-1), desc="Iteration over all nodes"):
        
        # To keep the tqdm progress bar correct
        if skipped > 0:
            skipped -= 1
            continue
        
        # calculate centrality
        edge_id, scores = find_best_edge(G_backup, w, edgeId_to_edge, algorithm_type, cluster_centers, ε, δ)
        if zeros_stay_zeros:
            
           
            
            # remove edges with zero centrality. They cannot be part of the ICT and are not relevant
            zeros = np.argwhere(scores == 0)
            zeros_idx = edgeId_to_edge[zeros].astype(int)
            for idx in zeros_idx:
                if G_backup.hasEdge(*idx[0]):
                    G_backup.removeEdge(*idx[0])

            # Nothing will happen anymore, we can exit early
            if scores[edge_id]==0:
                break
                

        # set the weight of the best edge to a minimal value
        G_backup.setWeight(*edgeId_to_edge[edge_id], w)

        # Append the good edges later 
        good_edges.append(edgeId_to_edge[edge_id].tolist())


        if create_snaps:
            if folder_name is None:
                folder_name = algorithm_type

            if position is None:
                print("Position array is not given. Cannot create snaps.")
                continue
            ICT = create_ICT(good_edges, w, G_backup)
            ICT.indexEdges()
            widths_ICT = compute_widths(ICT)
            widths_G = np.array(compute_widths(ICT))/10

            fig, ax = plt.subplots(1, figsize=(24,6))
            nk.viztasks.drawGraph(ICT, pos=position, ax=ax, width=widths_ICT, edge_color = "blue")
            nk.viztasks.drawGraph(G_backup, pos=position, ax=ax, width=1, edge_color = "red")
            if (algorithm_type == "cluster") or (algorithm_type=="cluster_py"):
                ax.plot(*position[cluster_centers].T, marker=".", color = "Red")
            ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
            ax.set_axis_on()
            ax.set_title(f"Creation of the ICT using the algorithm: " + algorithm_type)
            ax.axis("equal")
            name = str(i)
            plt.savefig(f"./Output/Videos/{folder_name}/{name.zfill(5)}.png")
            plt.close()
            
        if zeros_stay_zeros:
            if (len(edgeId_to_edge) > update_G*G_backup.numberOfEdges()) or (G_backup.numberOfEdges() < G_backup.numberOfNodes()+10):
                print(f"update Arr {len(edgeId_to_edge)} -> {G_backup.numberOfEdges()}")
                G_backup = create_new_graph(G_backup)
                edgeId_to_edge = calculate_edgeId_to_edge_no_bar(G_backup)
                
                skipped = add_sure_edges(G_backup, good_edges, w)
                
            # All remaining edges can be added
            if (G_backup.numberOfEdges() == G_backup.numberOfNodes()-1):
                skipped += add_remaining_edges(G_backup, good_edges, w)
                





    # add the edges with w weight to the spanning tree
    ICT = create_ICT(good_edges, w, G_backup)

    return ICT



def calculate_ICT_update(previous_ICT, G, position, cluster_centers,
                        used_components=19, reduction_factor=100):

    ICT = deepcopy(previous_ICT)
    
    
    weights = []
    edges = []
    
    for u, v, w in ICT.iterEdgesWeights():
        weights.append(G.weight(u, v))
        edges.append([u, v])
        
        
    # TODO: remove later
    plt.plot(sorted(weights)[::-1])
    plt.plot(used_components, sorted(weights)[::-1][used_components], marker="o", color="red")
    plt.show()
    
    
    largest_edges_indices = np.argpartition(weights, len(weights)-used_components-1)[-used_components:]
    
    for edge_idx in largest_edges_indices:
        ICT.removeEdge(*edges[edge_idx])
        
    

    cc = nk.components.ConnectedComponents(ICT)
    cc.run()
    components = cc.getComponents()
    
    node_to_component = np.zeros(G.numberOfNodes()).astype(int)
    for i, component in enumerate(components):
        for node in component:
            node_to_component[node] = i
    
    G_new = deepcopy(G)

    for u, v, w in G_new.iterEdgesWeights():
        if node_to_component[u] == node_to_component[v]:
            G_new.setWeight(u, v, w/reduction_factor)
            
            
    ICT_update = calculate_ICT(G_new, algorithm_type="cluster_all", cluster_centers=cluster_centers,
                                zeros_stay_zeros=True, update_G=1.1)
    ICT_update.indexEdges()
        
    return ICT_update, components, node_to_component