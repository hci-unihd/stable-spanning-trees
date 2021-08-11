import networkx as nx
from numpy import inf
from Stable_Tree.methods.Iterative.iterative_utils import add_edge_tree_by_edge_centrality_dict,filter_edges_not_used
from Stable_Tree.utils.utils import order_tuple,min_edge_weight,is_networkit_graph,copy_nkGraph

from copy import deepcopy
from heapq import heappush, heappop
from itertools import count
import numpy as np
from IPython.display import clear_output
from tqdm.notebook import tqdm
try:
    import networkit as nk
    networkit_imported=True
except ImportError:
    networkit_imported=False

#%%

def Iterative_centrality_tree_adjacency0(G,centrality_measure='sp_centrality',Added_edges_per_iteration=1,nx=False
                                         ,return_seq_centralities=False):
    '''
    Computes the ICT tree
    Parameters
    ----------
    G : networkx.Graph() or networkit.Graph()
        DESCRIPTION.
    Added_edges_per_iteration : int, optional
        parameter that determines how many edges are added in each iteration to
        the ICT. The default is 1.
    return_seq_centralities : bool, optional
        If True the edge centralities in each iteration are returned. The default is False.

    Returns
    -------
     Edges_T: list
        edges of the ICT
    T: networkx.Graph()
        ICT tree

    '''
    if networkit_imported and not nx:
        if not is_networkit_graph(G):
            G=nk.nxadapter.nx2nk(G,weightAttr='weight')
        return Iterative_centrality_tree_adjacency0_nk(G,Added_edges_per_iteration=Added_edges_per_iteration,return_seq_centralities=return_seq_centralities)
    else:
        return Iterative_centrality_tree_adjacency0_nx(G,centrality_measure=centrality_measure,Added_edges_per_iteration=Added_edges_per_iteration)
        


def Iterative_centrality_tree_adjacency0_nx(G,centrality_measure='sp_centrality',Added_edges_per_iteration=1):
    '''
    

    Computes the ICT tree
    Parameters
    ----------
    G : networkx.Graph()
        DESCRIPTION.
    Added_edges_per_iteration : int, optional
        parameter that determines how many edges are added in each iteration to
        the ICT. The default is 1.
    return_seq_centralities : bool, optional
        If True the edge centralities in each iteration are returned. The default is False.

    Returns
    -------
     Edges_T: list
        edges of the ICT
    T: networkx.Graph()
        ICT tree

    '''
    G_updated=G.copy()
    T=nx.Graph()
    Edges_cycle=[]
    w_min=min_edge_weight(G)
            
    num_nodes=G.number_of_nodes()
    Edges_T=[]
    node_classes_T={}
    new_weight=w_min/(G.number_of_edges()+1)
    for _ in range(0,num_nodes-1,Added_edges_per_iteration):
        if centrality_measure=='sp_centrality':
            E_centrality_dict=nx.edge_betweenness_centrality(G_updated,weight='weight',normalized=False)
        elif centrality_measure=='flow_betweeness':
            E_centrality_dict=nx.edge_current_flow_betweenness_centrality(G_updated,weight='weight')
        
        length_edges_T=len(Edges_T)
        
        Edges_T,Edges_cycle,node_classes_T=add_edge_tree_by_edge_centrality_dict(E_centrality_dict,Edges_T,Edges_cycle,node_classes_T,G
                                     ,Added_edges_per_iteration)
        for counter in range(length_edges_T,len(Edges_T)):
            e=Edges_T[counter]
            T.add_weighted_edges_from([(*e,G.get_edge_data(e[0],e[1])['weight'])])
            G_updated[e[1]][e[0]]['weight']=new_weight
        filter_edges_not_used(G_updated,E_centrality_dict)
        
    return Edges_T,T


def Iterative_centrality_tree_adjacency0_nk(G,Added_edges_per_iteration=1,return_seq_centralities=False):
    '''
    
    Computes the ICT tree
    Parameters
    ----------
    G : networkit.Graph()
        DESCRIPTION.
    Added_edges_per_iteration : int, optional
        parameter that determines how many edges are added in each iteration to
        the ICT. The default is 1.
    return_seq_centralities : bool, optional
        If True the edge centralities in each iteration are returned. The default is False.

    Returns
    -------
     Edges_T: list
        edges of the ICT
    T: networkx.Graph()
        ICT tree

    '''
    G_updated=copy_nkGraph(G)
    G_updated.indexEdges()
    num_nodes=G.numberOfNodes()

    T=nk.Graph(n=num_nodes,weighted=True)
    Edges_cycle=[]
    w_min=min_edge_weight(G)

            
    Edges_T=[]
    node_classes_T={}
    new_weight=w_min/(num_nodes+1)
    #TODO
    #remove seq_dicts
    if return_seq_centralities:
        seq_dicts=[]
    for i in tqdm(range(0,num_nodes-1,Added_edges_per_iteration), desc="Iteration over all nodes"):
        
#         clear_output(wait=True)
#         print(f"iteration {i+1} / {num_nodes-1}")
    
        G_updated.indexEdges()

        EC=nk.centrality.Betweenness(G_updated,normalized=False,computeEdgeCentrality=True).run().edgeScores()
        if return_seq_centralities:
            EC=15*np.array(EC)/max(EC)
            
        #EC to dict 
        E_centrality_dict={}
        for e in G_updated.iterEdges():
            u,v=tuple(sorted(e))
            idx=G_updated.edgeId(u,v)
            E_centrality_dict[(u,v)]=EC[idx]
            # if EC[idx]==0:
            #     G_updated.removeEdge(*e)
        length_edges_T=len(Edges_T)
        
        Edges_T,Edges_cycle,node_classes_T=add_edge_tree_by_edge_centrality_dict(E_centrality_dict,Edges_T,Edges_cycle,node_classes_T,G
                                     ,Added_edges_per_iteration)
        
        # if len(node_classes_T)>1:
        #     print('more than two clusters')
        for counter in range(length_edges_T,len(Edges_T)):
            e=Edges_T[counter]
            T.addEdge(e[0],e[1],G.weight(e[0],e[1]))
            G_updated.setWeight(e[0],e[1],new_weight)
            if return_seq_centralities:
                seq_dicts.append(E_centrality_dict)
        if len(Edges_T)==num_nodes-1:
            break
        #Remove edges with centrality equal to 0
        filter_edges_not_used(G_updated,E_centrality_dict)
        # print(G_updated.numberOfEdges()/G.numberOfEdges())
        # quit
    if return_seq_centralities:
        return Edges_T,T,seq_dicts
    return Edges_T,T


#%%

def Iterative_centrality_tree_merge(G,centrality_measure='sp_centrality',nx=False):
    if networkit_imported and not nx:
        if not is_networkit_graph(G):
            G=nk.nxadapter.nx2nk(G,weightAttr='weight')
        return Iterative_centrality_tree_merge_nk(G)
    else:
        return Iterative_centrality_tree_merge_nx(G,centrality_measure='sp_centrality')
        


def Iterative_centrality_tree_merge_nx(G,centrality_measure='sp_centrality'):
    '''
    Merge scenario nodes.
    In contrast to the previous algorithm this one counts less shortest paths corresponding to the paths formed by one edge
    with length 0, since these edges are contracted and they do not exist.
    
    '''
    G_updated=G.copy()
    T=nx.Graph()
    Edges_T=[]
    merges={}
    for _ in range(G.number_of_nodes()-1):
        if centrality_measure=='sp_centrality':
            E_centrality_dict=nx.edge_betweenness_centrality(G_updated,weight='weight')
        elif centrality_measure=='flow_betweeness':
            E_centrality_dict=nx.edge_current_flow_betweenness_centrality(G_updated,weight='weight')
        e=max(E_centrality_dict, key=E_centrality_dict.get)
        e=tuple(sorted(e))
        
        
        
        w_min=inf
        if len(merges)==0:
            e_min=e
        
        if e[0] not in merges.keys():
            merges[e[0]]=set([e[0]])
            
        if e[1] not in merges.keys():
            for node in merges[e[0]]:
                if e[1] in G.neighbors(node):
                    if G[node][e[1]]['weight']<w_min:
                        w_min=G[node][e[1]]['weight']
                        e_min=(node,e[1])
        else:
            for node1 in merges[e[0]]:
                for node2 in merges[e[1]]:
                    if node1 in G.neighbors(node2):
                        if G[node1][node2]['weight']<w_min:
                            w_min=G[node1][node2]['weight']
                            e_min=(node1,node2)
        
        e_min=tuple(sorted(e_min))
        Edges_T.append(e_min)

        T.add_weighted_edges_from([(*e_min,G.get_edge_data(e_min[0],e_min[1])['weight'])])
        
        if e[1] not in merges.keys():
            merges[e[0]]=merges[e[0]].union(set(e))
        else:
            merges[e[0]]=merges[e[0]].union(merges[e[1]])
            del merges[e[1]]
        
        
        G_updated=nx.contracted_edge(G_updated,e,self_loops=False)
        for node in G_updated.neighbors(e[0]):
            w_min1=inf
            for node1 in merges[e[0]]:
                if node in G.neighbors(node1):
                    if G[node1][node]['weight']<w_min1:
                        w_min1=G[node][node1]['weight']
            
            G_updated[node][e[0]]['weight']=w_min1
        
       
    return Edges_T,T




def Iterative_centrality_tree_merge_nk(G,centrality_measure='sp_centrality'):
    '''
    Merge scenario nodes.
    In contrast to the previous algorithm this one counts less shortest paths corresponding to the paths formed by one edge
    with length 0, since these edges are contracted and they do not exist.
    
    '''
    G_updated=copy_nkGraph(G)
    G_updated.indexEdges()
    num_nodes=G.numberOfNodes()
    Edges_T=[]
    merges={}
    T=nk.Graph(n=num_nodes,weighted=True)
    for _ in range(num_nodes-1):
        EC=nk.centrality.Betweenness(G_updated,normalized=True,computeEdgeCentrality=True).run().edgeScores()
        E_centrality_dict={}

        for e in G_updated.iterEdges():
            u,v=tuple(sorted(e))
            idx=G_updated.edgeId(u,v)
            E_centrality_dict[(u,v)]=EC[idx]
        
        e=max(E_centrality_dict, key=E_centrality_dict.get)
        e=tuple(sorted(e))
        
        
        
        w_min=inf
        if len(merges)==0:
            e_min=e
        
        if e[0] not in merges.keys():
            merges[e[0]]=set([e[0]])
            
        if e[1] not in merges.keys():
            for node in merges[e[0]]:
                if e[1] in G.iterNeighbors(node):
                    if G.weight(node,e[1])<w_min:
                        w_min=G.weight(node,e[1])
                        e_min=(node,e[1])
        else:
            for node1 in merges[e[0]]:
                for node2 in merges[e[1]]:
                    if node1 in G.iterNeighbors(node2):
                        if G.weight(node1,node2)<w_min:
                            w_min=G.weight(node1,node2)
                            e_min=(node1,node2)
        
        e_min=tuple(sorted(e_min))
        Edges_T.append(e_min)

        T.addEdge(e_min[0],e_min[1],G.weight(e_min[0],e_min[1]))
        
        if e[1] not in merges.keys():
            merges[e[0]]=merges[e[0]].union(set(e))
        else:
            merges[e[0]]=merges[e[0]].union(merges[e[1]])
            del merges[e[1]]
        
        
        
        for node in G_updated.iterNeighbors(e[1]):
            w1=G_updated.weight(node,e[1])
            if node!=e[0]:
                if G_updated.hasEdge(node,e[0]):
                    w0=G_updated.weight(node,e[0])
                    G_updated.setWeight(node,e[0],min(w1,w0))
                else:
                    G_updated.addEdge(node,e[0],w1)
                
        G_updated.removeNode(e[1])
       
    return Edges_T,T


#%%


def Find_Affected_Sources(G,updated_edge,distances):
    '''


    Parameters
    ----------
    G : TYPE
        DESCRIPTION.
    updated_edge : TYPE
        DESCRIPTION.
    distances : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    u,v,w=updated_edge
    affected_sources=[]
    for node in G.nodes:
        tuple1=order_tuple(u,node)
        tuple2=order_tuple(v,node)
        d1=distances[tuple1[0]][tuple1[1]]
        d2=distances[tuple2[0]][tuple2[1]]
            
        if w+d1<=d2:
            affected_sources.append(node)
            
    return affected_sources
    
def augmented_ASPS_update(G,updated_edge,distances,num_paths):
    '''
    Implementation Bergamini et al. "Faster Betweenness Centrality Updates in Evolving Networks" 2017.

    Parameters
    ----------
    G : TYPE
        DESCRIPTION.
    updated_edge : TYPE
        DESCRIPTION.
    distances : TYPE
        DESCRIPTION.
    num_paths : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    #init variables
    vis=[]#visited nodes
    u,v,w=updated_edge
    S={}#affected sources
    Q=[]#
    P={} #predecesors
    
    new_distances=deepcopy(distances)
    new_num_paths=deepcopy(num_paths)
    
    S[v]=Find_Affected_Sources(G,updated_edge,distances)
    Q.append(v)
    P[v]=v
    vis.append(v)
    
    while len(Q)>0:
        t=Q.pop(0)
        for s in S[P[v]]:
            tuple0=order_tuple(s,t)
            tuple1=order_tuple(s,u)
            tuple2=order_tuple(v,t)
            
            d0=distances[tuple0[0]][tuple0[1]]
            d1=distances[tuple1[0]][tuple1[1]]
            d2=distances[tuple2[0]][tuple2[1]]
            
            
            if d0>=d1+w+d2:
                if d0>d1+w+d2:
                    new_distances[tuple0[0]][tuple0[1]]=d1+w+d2
                    new_num_paths[tuple0[0]][tuple0[1]]=0
                new_num_paths[tuple0[0]][tuple0[1]]=new_num_paths[tuple0[0]][tuple0[1]]+num_paths[tuple1[0]][tuple1[1]]*num_paths[tuple2[0]][tuple2[1]]
                if t!=v:
                    try:
                        S[t].append(s)
                    except:
                        S[t]=[s]
        
        for z in G.neighbors(t):
            tuple1=order_tuple(u,z)
            tuple2=order_tuple(v,z)
                       
            d1=distances[tuple1[0]][tuple1[1]]
            d2=distances[tuple2[0]][tuple2[1]]
            if z not in vis and d1>=w+d2:
                Q.append(z)
                vis.append(z)
                P[z]=t
            
    return new_distances,new_num_paths,S
#
#def augment_affected_sources(list_nodes,S):
#    keys=[*S]
#    for node in list_nodes:
#        if node not in keys :
#            for s in keys:
#                if node in S[s]:
#                    try:
#                        S[node].append(s)
#                    except:
#                        S[node]=[s]
#    return S

def approx_comparison(d1,d2,w,tol=1e-10):
    return abs(d1-d2-w)<tol


def contribution_nodes(G,old_distances,old_num_paths,new_distances,new_num_paths,S,E_centrality,updated_edge,old_weight):
    
    for s in S:
        Ts_old_dict= {}
        Ts_list=[]
        Ts_new_dict= {}
        
        old_contribution_s = dict.fromkeys(G, 0.0)
        old_contribution_s.update(dict.fromkeys(G.edges(), 0.0))
        new_contribution_s = dict.fromkeys(G, 0.0)
        new_contribution_s.update(dict.fromkeys(G.edges(), 0.0))
        
        for k in S[s]:
            tuple0=order_tuple(s,k)
            Ts_old_dict[k]=old_distances[tuple0[0]][tuple0[1]]
            Ts_new_dict[k]=new_distances[tuple0[0]][tuple0[1]]
            Ts_list.append(k)
        
        while len(Ts_old_dict)!=0:
            z=max(Ts_old_dict,key=Ts_old_dict.get) 
            del Ts_old_dict[z]
            for y in G.neighbors(z):
#                try:
#                    E_centrality[(y,z)]-=2*old_contribution_s[(y,z)]
#                except:
#                    E_centrality[(z,y)]-=2*old_contribution_s[(z,y)]
#                
                tuple1=order_tuple(s,z)
                tuple2=order_tuple(s,y)
                d1=old_distances[tuple1[0]][tuple1[1]]
                d2=old_distances[tuple2[0]][tuple2[1]]
                if (z==updated_edge[0] and y== updated_edge[1]) or (z==updated_edge[1] and y== updated_edge[0]):
                    w=old_weight
                else:
                    w=G.get_edge_data(z,y)['weight']
                if approx_comparison(d1,d2,w):# and y!=s:
                    if z in Ts_list:
                        c=old_num_paths[tuple2[0]][tuple2[1]]/old_num_paths[tuple1[0]][tuple1[1]]*(1+old_contribution_s[z])
                    else:
                        c=old_num_paths[tuple2[0]][tuple2[1]]/old_num_paths[tuple1[0]][tuple1[1]]*old_contribution_s[z]
                        
                    if y not in Ts_old_dict.keys() and y!=s:
                        Ts_old_dict[y]=d2
                        
                    old_contribution_s[y]+=c
                    if (z,y) in old_contribution_s.keys():
                        old_contribution_s[(z,y)]=c
                    else:
                        old_contribution_s[(y,z)]=c
                    try:
                        E_centrality[(y,z)]-=2*old_contribution_s[(y,z)]
                    except:
                        E_centrality[(z,y)]-=2*old_contribution_s[(z,y)]
                    
        while len(Ts_new_dict)!=0:
            z=max(Ts_new_dict,key=Ts_new_dict.get) 
            del Ts_new_dict[z]
            for y in G.neighbors(z):
#                try:
#                    E_centrality[(y,z)]+=2*new_contribution_s[z]
#                except:
#                    E_centrality[(z,y)]+=2*new_contribution_s[z]
                
                tuple1=order_tuple(s,z)
                tuple2=order_tuple(s,y)
                d1=new_distances[tuple1[0]][tuple1[1]]
                d2=new_distances[tuple2[0]][tuple2[1]]
                w=G.get_edge_data(z,y)['weight']
                if approx_comparison(d1,d2,w):# and y!=s:
                    if z in Ts_list:
                        c=new_num_paths[tuple2[0]][tuple2[1]]/new_num_paths[tuple1[0]][tuple1[1]]*(1+new_contribution_s[z])
                    else:
                        c=new_num_paths[tuple2[0]][tuple2[1]]/new_num_paths[tuple1[0]][tuple1[1]]*new_contribution_s[z]
                        
                    if y not in Ts_new_dict.keys() and y!=s:
                        Ts_new_dict[y]=d2
                    
                    new_contribution_s[y]+=c
                    if (z,y) in new_contribution_s.keys():
                        new_contribution_s[(z,y)]=c
                    else:
                        new_contribution_s[(y,z)]=c
                    try:
                        E_centrality[(y,z)]+=2*new_contribution_s[(y,z)]
                    except:
                        E_centrality[(z,y)]+=2*new_contribution_s[(z,y)]
                    
    return E_centrality
                
def iterative_dynamic_centrality_tree(G):
    
    '''
    Implementation algorithm
    @misc{bergamini2017faster,
      title={Faster Betweenness Centrality Updates in Evolving Networks}, 
      author={Elisabetta Bergamini and Henning Meyerhenke and Mark Ortmann and Arie Slobbe},
      year={2017},
    '''
    #init
    G_updated=G.copy()
    w_min=min_edge_weight(G)
    num_nodes=G.number_of_nodes()
    Edges_T=[]
    Edges_cycle=[]
    T=nx.Graph()
    node_classes_T={}

    # list_nodes=list(G.nodes())
    new_weight=w_min/(G.number_of_edges()+1)
    E_centrality_dict,old_distances,old_num_paths=edge_betweenness_centrality(G)
    e=max(E_centrality_dict, key=E_centrality_dict.get)
    T.add_weighted_edges_from([(*e,G.get_edge_data(e[0],e[1])['weight'])])
    Edges_T.append(e)
    old_weight=G_updated[e[1]][e[0]]['weight']
    G_updated[e[1]][e[0]]['weight']=new_weight
    for _ in range(1,num_nodes-1):
        updated_edge=[e[0],e[1],new_weight]
        
        # breakpoint()
        new_distances,new_num_paths,affected_sources=augmented_ASPS_update(G_updated,updated_edge,old_distances,old_num_paths)
        
#        affected_sources=augment_affected_sources(list_nodes,affected_sources)
        
        E_centrality_dict=contribution_nodes(G_updated,old_distances,old_num_paths,new_distances,new_num_paths,affected_sources,E_centrality_dict,updated_edge,old_weight)
        Edges_T,Edges_cycle,node_classes_T=add_edge_tree_by_edge_centrality_dict(E_centrality_dict,Edges_T,Edges_cycle,node_classes_T,G)
        e=Edges_T[-1]
        T.add_weighted_edges_from([(*e,G.get_edge_data(e[0],e[1])['weight'])])
        old_weight=G_updated[e[1]][e[0]]['weight']
        G_updated[e[1]][e[0]]['weight']=new_weight
        old_distances=deepcopy(new_distances)
        old_num_paths=deepcopy(new_num_paths)
#        print(T.edges(data=True))
    


    return Edges_T,T

#%%
#Implementation networkx modified to get distances and number of paths
# @py_random_state(4)
def edge_betweenness_centrality(G, k=None, normalized=True,seed=None):
    r"""Compute betweenness centrality for edges.

    Betweenness centrality of an edge $e$ is the sum of the
    fraction of all-pairs shortest paths that pass through $e$

    .. math::

       c_B(e) =\sum_{s,t \in V} \frac{\sigma(s, t|e)}{\sigma(s, t)}

    where $V$ is the set of nodes, $\sigma(s, t)$ is the number of
    shortest $(s, t)$-paths, and $\sigma(s, t|e)$ is the number of
    those paths passing through edge $e$ [2]_.

    Parameters
    ----------
    G : graph
      A NetworkX graph.

    k : int, optional (default=None)
      If k is not None use k node samples to estimate betweenness.
      The value of k <= n where n is the number of nodes in the graph.
      Higher values give better approximation.

    normalized : bool, optional
      If True the betweenness values are normalized by $2/(n(n-1))$
      for graphs, and $1/(n(n-1))$ for directed graphs where $n$
      is the number of nodes in G.

    weight : None or string, optional (default=None)
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
        Note that this is only used if k is not None.

    Returns
    -------
    edges : dictionary
       Dictionary of edges with betweenness centrality as the value.

    See Also
    --------
    betweenness_centrality
    edge_load

    Notes
    -----
    The algorithm is from Ulrik Brandes [1]_.

    For weighted graphs the edge weights must be greater than zero.
    Zero edge weights can produce an infinite number of equal length
    paths between pairs of nodes.

    References
    ----------
    .. [1]  A Faster Algorithm for Betweenness Centrality. Ulrik Brandes,
       Journal of Mathematical Sociology 25(2):163-177, 2001.
       http://www.inf.uni-konstanz.de/algo/publications/b-fabc-01.pdf
    .. [2] Ulrik Brandes: On Variants of Shortest-Path Betweenness
       Centrality and their Generic Computation.
       Social Networks 30(2):136-145, 2008.
       http://www.inf.uni-konstanz.de/algo/publications/b-vspbc-08.pdf
    """
    betweenness = dict.fromkeys(G, 0.0)  # b[v]=0 for v in G
    # b[e]=0 for e in G.edges()
    betweenness.update(dict.fromkeys(G.edges(), 0.0))
    D = dict.fromkeys(G, {})
    sigma = dict.fromkeys(G, {})
    if k is None:
        nodes = G
    else:
        nodes = seed.sample(G.nodes(), k)
    for s in nodes:
        # use Dijkstra's algorithm
        S, P, sigma[s],D[s] = _single_source_dijkstra_path_basic(G, s)
        # accumulation
        betweenness = _accumulate_edges(betweenness, S, P, sigma[s], s)
        
    # rescaling
    for n in G:  # remove nodes to only return edges
        del betweenness[n]
    # betweenness = _rescale_e(
    #     betweenness, len(G), normalized=normalized, directed=G.is_directed()
    # )
    return betweenness,D,sigma

def _single_source_dijkstra_path_basic(G, s):
    # modified from Eppstein
    S = []
    P = {}
    for v in G:
        P[v] = []
    sigma = dict.fromkeys(G, 0.0)  # sigma[v]=0 for v in G
    D = {}
    sigma[s] = 1.0
    push = heappush
    pop = heappop
    seen = {s: 0}
    c = count()
    Q = []  # use Q as heap with (distance,node id) tuples
    push(Q, (0, next(c), s, s))
    while Q:
        (dist, _, pred, v) = pop(Q)
        if v in D:
            continue  # already searched this node.
        sigma[v] += sigma[pred]  # count paths
        S.append(v)
        D[v] = dist
        for w, edgedata in G[v].items():
            vw_dist = dist + edgedata.get('weight', 1)
            if w not in D and (w not in seen or vw_dist < seen[w]):
                seen[w] = vw_dist
                push(Q, (vw_dist, next(c), v, w))
                sigma[w] = 0.0
                P[w] = [v]
            elif vw_dist == seen[w]:  # handle equal paths
                sigma[w] += sigma[v]
                P[w].append(v)
    
    #avoid double counting path
    for key in sigma.keys():
        sigma[key]/=2
    return S, P, sigma,D


def _accumulate_edges(betweenness, S, P, sigma, s):
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        coeff = (1 + delta[w]) / sigma[w]
        for v in P[w]:
            c = sigma[v] * coeff
            if (v, w) not in betweenness:
                betweenness[(w, v)] += c
            else:
                betweenness[(v, w)] += c
            delta[v] += c
        if w != s:
            betweenness[w] += delta[w]
    return betweenness

