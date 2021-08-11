#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 17:25:26 2021

@author: enfita
"""
from Stable_Tree.methods.Iterative.iterative_utils import add_edge_tree_by_edge_centrality_dict,filter_edges_not_used
from Stable_Tree.utils.utils import order_tuple,min_edge_weight

import networkx as nx
from numpy import inf 
from random import choices,sample
from random import seed as randomseed
from heapq import heappush, heappop
from itertools import count
from math import floor,log2,log
import operator
from Stable_Tree.methods.Iterative.ICT_exact import Iterative_centrality_tree_adjacency0
import numpy as np

randomseed(42)
#%%
# def UpdateSSSP_Bergamini(G,d_s,num_path_s,updated_edges):
#     # d_s=deepcopy(old_d_s)
#     # num_path_s=deepcopy(old_num_path_s)
#     Q = {}
#     for e in updated_edges:
#         u,v,w=e

#         if d_s[u]>d_s[v]:
#             v,u=u,v
#         d=min(d_s[u]+w,d_s[v])
#         Q[v]=d
        
#     while len(Q)!=0:
#         v=min(Q,key=Q.get)
#         p_v=Q.pop(v)
#         con_v=inf
#         for u in G.neighbors(v):
#             w=G.get_edge_data(u,v)['weight']
#             con_v=min(con_v,d_s[u]+w)
            
#         if con_v==p_v:
#             d_s[v]=p_v
#             num_path_s[v]=0
#             for z in G.neighbors(v):
#                 w=G.get_edge_data(z,v)['weight']
#                 if d_s[v]==d_s[z]+w:
#                     num_path_s[v]+=num_path_s[z]
#                 if d_s[z]>=d_s[v]+w:
#                     if z in Q.keys():
#                         if d_s[v]+w<=Q[z]:
#                             Q[z]=d_s[v]+w
#                     else:
#                         Q[z]=d_s[v]+w
#     return d_s,num_path_s


def UpdateSSSP(G,d_s,num_path_s,updated_edges):
    '''
    Updates shortest paths distances with starting node equal s. Only valid if
    the edges that have been updated have decreased their weight.

    Parameters
    ----------
    G : networkx.Graph()
    d_s : dict
        current shortest path distances from node s. key=node, value=distance
    num_path_s : dict 
         key=node, value=number of shortest path starting at s and 
         ending at node. s is always lower than node.
    updated_edges : dict
         key=edge, value=updated weight of the edge.

    Returns
    -------
    d_s : dict
        updated shortest path distances from node s. key=node, value=distance
    num_path_s : dict
        Updated ->key=edge, value=number of shortest path crossing the edge

    '''
    # d_s=deepcopy(old_d_s)
    # num_path_s=deepcopy(old_num_path_s)
    d_old=d_s.copy()
    Q = {}
    
    for e in updated_edges:
        
        u,v,w=e

        if d_s[u]>d_s[v]:
            v,u=u,v
        if d_s[u]+w<=d_s[v]:
            Q[v]=d_s[u]+w

    while len(Q)!=0:
        v=min(Q,key=Q.get)
        p_v=Q.pop(v)
        if p_v<=d_old[v]:
            d_s[v]=p_v
            num_path_s[v]=0
            for z in G.neighbors(v):
                w=G.get_edge_data(z,v)['weight']
                if d_s[v]==d_s[z]+w:
                    num_path_s[v]+=num_path_s[z]
                if d_s[z]>=d_s[v]+w:
                    if z in Q.keys():
                        if d_s[v]+w<=Q[z]:
                            Q[z]=d_s[v]+w
                    else:
                        Q[z]=d_s[v]+w
                        
    return d_s,num_path_s

def edge_centrality_approx_update(G,edge_centrality,distances,num_paths,paths,updated_edges):
    '''
    Implementation update edge centrality from
    @misc{bergamini2017faster,
      title={Faster Betweenness Centrality Updates in Evolving Networks}, 
      author={Elisabetta Bergamini and Henning Meyerhenke and Mark Ortmann and Arie Slobbe},
      year={2017},}

    Parameters
    ----------
    G : TYPE
        DESCRIPTION.
    edge_centrality : dict
        key=edge, value=edge centrality
    distances : dict of dicts
         key1=s, key2=t, value=shortest path between s and 
         t. s is always lower than t.
    num_paths :  dict of dicts
         key1=s, key2=t, value=number of shortest path starting at s and 
         ending at t. s is always lower than t.
    paths : list
        sampled paths.
    updated_edges : dict
         key=edge, value=updated weight of the edge.

    Returns
    -------
    edge_centrality : dict
        Updated edge centralities: key=edge, value=edge centrality
    distances_new : dict of dicts
         Updated distances: key1=s, key2=t, value=shortest path between s and 
         t. s is always lower than t.
    num_paths_new : dict of dicts
         updated num_paths: key1=s, key2=t, value=number of shortest path starting at s and 
         ending at t. s is always lower than t.
    paths : list
        updated sampled paths.

    '''
    r=len(paths)
    distances_new={}
    num_paths_new={}
    for path_idx in range(r):
        path=paths[path_idx]
        s=path[0]#source
        t=path[-1]#target
        s,t=order_tuple(s,t)#sets as source the node with lower index
        assert(s<t)
        d_st_old=distances[s][t]
        num_paths_st_old=num_paths[s][t]
        # update distances
        if s not in distances_new.keys():
            distances_new[s],num_paths_new[s]=UpdateSSSP(G,distances[s].copy(),num_paths[s].copy(),updated_edges)
        #update edge centrality
        if distances_new[s][t]<d_st_old or num_paths_new[s][t]!=num_paths_st_old:
            # breakpoint()
            for i in range(len(path)-1):
                u=path[i]
                v=path[i+1]
                if (u, v) not in edge_centrality:
                    edge_centrality[(v,u)]-=1/r
                else:
                    edge_centrality[(u,v)]-=1/r
                
            assert(v==t)
            
            
            edge_centrality,path=sample_path(G,s,t,distances_new[s],num_paths_new[s],edge_centrality,r)
            paths[path_idx]=path
                    
    return edge_centrality,distances_new,num_paths_new,paths
#%%
#TODO
#be careful with edges with weight 0 to not be crossed more than once.
def sample_path(G,s,t,d_s,num_paths_s,edge_centrality,r):
    v=t
    path=[t]
    while True:
        nodes=[]
        probs=[]
        for u in G.neighbors(v):
            w=G.get_edge_data(u,v)['weight']
            if d_s[v]==d_s[u]+w:
                nodes.append(u)
                probs.append(num_paths_s[u]/num_paths_s[v])
        
        z=choices(nodes,probs)[0]
        
        
        
        if (z, v) not in edge_centrality:

            edge_centrality[(v,z)]+=1/r
        else:
            edge_centrality[(z,v)]+=1/r
        path.append(z)
        if z==s:
            break
        else:
            v=z
    path.reverse()
    return edge_centrality,path



def computeExtendedSSSP(G, s,weight='weight' ):
    '''
    Implementation from networkx edge centrality
    function _single_source_dijkstra_path_basic
    Parameters
    ----------
    G : TYPE
        DESCRIPTION.
    s : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    be careful with edges with weight 0 to not be crossed more than once.
    '''
    
    
    # modified from Eppstein
    S = []
    # P = {}
    # for v in G:
    #     P[v] = []
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
            vw_dist = dist + edgedata.get(weight, 1)
            if w not in D and (w not in seen or vw_dist < seen[w]):
                seen[w] = vw_dist
                push(Q, (vw_dist, next(c), v, w))
                sigma[w] = 0.0
                # P[w] = [v]
            elif vw_dist == seen[w]:  # handle equal paths
                sigma[w] += sigma[v]
                # P[w].append(v)
    return D, sigma


def sample_source_target(sources,non_sampled_targets):
    s=sample(sources,1)[0]
    # print('s=',s)
    t=sample(non_sampled_targets[s],1)[0]
    # print('t=',t)
    non_sampled_targets[s].remove(t)
    if len(non_sampled_targets[s])==0:
        del non_sampled_targets[s]
        sources.remove(s)
    return s,t,sources,non_sampled_targets
    
    
def edge_centrality_RK(G,eps=0.1,delta=0.1,c=0.5):
    '''
    Implementation approximation edge centrality algorithm 1 in
    
    @article{article,
    author = {Bergamini, Elisabetta and Meyerhenke, Henning},
    year = {2015},
    month = {10},
    pages = {},
    title = {Approximating Betweenness Centrality in Fully Dynamic Networks},
    journal = {Internet Mathematics},
    doi = {10.1080/15427951.2016.1177802}
    }

    Parameters
    ----------
    G : TYPE
        DESCRIPTION.
    eps : TYPE, optional
        DESCRIPTION. The default is 0.1.
    delta : TYPE, optional
        DESCRIPTION. The default is 0.1.
    c : TYPE, optional
        DESCRIPTION. The default is 0.5.

    Returns
    -------
    edge_centrality : TYPE
        DESCRIPTION.
    d : TYPE
        DESCRIPTION.
    num_paths : TYPE
        DESCRIPTION.
    paths : TYPE
        DESCRIPTION.

    '''
    #init

    edge_centrality=dict.fromkeys(G.edges(), 0.0)
    d={}
    num_paths={}
    paths=[]
    
    #sample source and target
    sources=list(G.nodes())
    sources.remove(G.number_of_nodes()-1)
    non_sampled_targets={}
    
    # %%timeit
    # non_sampled_targets={}
    # for i in sources:
    #     non_sampled_targets[i]=list(range(i+1,G.number_of_nodes()))
    # %%timeit
    non_sampled_targets={i:list(range(i+1,G.number_of_nodes())) for i in sources}

    s,t,sources,non_sampled_targets=sample_source_target(sources,non_sampled_targets)
    #computation distances and number shortest paths from the source s
    d[s],num_paths[s]=computeExtendedSSSP(G, s )
   

    #Approximation vertex diameter    
    w_min= min_edge_weight(G)
    d_=d[s].copy()
    key_max=max(d_.items(), key=operator.itemgetter(1))[0]
    du=d_[key_max]
    del d_[key_max]
    dv=max(d_.values()) 
    VD=min(1+(du+dv)/w_min,G.number_of_nodes())
    print('Vertex Diameter=%i'%VD)
    
    
    r=floor(min((c/eps**2)*(floor(log2(VD-1))+1+log(1/delta)),G.number_of_nodes()*(G.number_of_nodes()-1)/2))
    print('number paths=%i'%r)
    #Sample path an update edge_centrality
    edge_centrality,path=sample_path(G,s,t,d[s],num_paths[s],edge_centrality,r)
    paths.append(path)
    for i in range(1,r):
        s,t,sources,non_sampled_targets=sample_source_target(sources,non_sampled_targets)
        if s not in d.keys():
            d[s],num_paths[s]=computeExtendedSSSP(G, s )
        edge_centrality,path=sample_path(G,s,t,d[s],num_paths[s],edge_centrality,r)
        paths.append(path)
        
    return edge_centrality,d,num_paths,paths


def iterative_approx_centrality_tree(G,eps,delta,New_weight=None,Exact=False
                                     ,Added_edges_per_iteration=1):
    
    '''
    computes an approximation of the ICT tree by computing the edge centralities
    using the algorithm in 
    @article{article,
    author = {Bergamini, Elisabetta and Meyerhenke, Henning},
    year = {2015},
    month = {10},
    pages = {},
    title = {Approximating Betweenness Centrality in Fully Dynamic Networks},
    journal = {Internet Mathematics},
    doi = {10.1080/15427951.2016.1177802}
    }

    Parameters
    ----------
    G : networkx.graph()
        Graph.
    eps : TYPE
        DESCRIPTION.
    delta : TYPE
        DESCRIPTION.
    new_weight : TYPE, optional
        It sets the weight of the updated edge. The default is None.
        If None: the new weight is set to the minimum weightof G divided by the 
        number of edges
        If =='average':sets the newweight equal to the average of the weights of the edges, which have
        a lower weight than the current edge weight being updated
        if isinstance(new_weight,float): new_weight is the float
    Added_edges_per_iteration: int
        parameter that determines how many edges are added in each iteration to
        the ICT. Default=1
    Returns
    -------
    E_T: list
        edges of the ICT
    T: networkx.Graph()
        approximation ICT tree

    '''
    
    if Exact or G.number_of_nodes()<=150:
        return Iterative_centrality_tree_adjacency0(G,Added_edges_per_iteration=Added_edges_per_iteration)
    #init
    G_updated=G.copy()
    w_min=min_edge_weight(G)
    num_nodes=G.number_of_nodes()
    Edges_T=[]
    Edges_cycle=[]
    T=nx.Graph()
    node_classes_T={}
    
    if New_weight is None:
        new_weight=w_min/((G.number_of_nodes()-1)*G.number_of_nodes())
    elif isinstance(New_weight,float):
        new_weight=New_weight
    elif New_weight=='average':
        edge_list=list(G.edges())
        all_weights=np.array([G[e[0]][e[1]]['weight'] for e in G.edges()])
    
    #init edge centralities
    E_centrality_dict,d,num_paths,paths=edge_centrality_RK(G,eps,delta,c=0.5)
    e=max(E_centrality_dict, key=E_centrality_dict.get)
    node_classes_T[min(*e)]=set(e)
    T.add_edge(*e)
    T[e[0]][e[1]]['weight'] = G.get_edge_data(e[0],e[1])['weight']
    Edges_T.append(e)
    if New_weight=='average':
        current_weight=G.get_edge_data(e[0],e[1])['weight']
        new_weight=np.mean(all_weights[np.where(all_weights<=current_weight)])
        all_weights[edge_list.index(e)]=new_weight
    G_updated[e[1]][e[0]]['weight']=new_weight
    updated_edges=[[e[0],e[1],new_weight]]
    for _ in range(1,num_nodes-1,Added_edges_per_iteration):
        
        # breakpoint()
        E_centrality_dict,d,num_paths,paths=edge_centrality_approx_update(G_updated,E_centrality_dict,d,num_paths,paths,updated_edges)
        
        length_edges_T=len(Edges_T)
        Edges_T,Edges_cycle,node_classes_T=add_edge_tree_by_edge_centrality_dict(E_centrality_dict,Edges_T,Edges_cycle,node_classes_T,G
                                     ,Added_edges_per_iteration)
        updated_edges=[]
        for counter in range(length_edges_T,len(Edges_T)):
            e=Edges_T[counter]
            T.add_weighted_edges_from([(*e,G.get_edge_data(e[0],e[1])['weight'])])

            if New_weight=='average':
                current_weight=G.get_edge_data(e[0],e[1])['weight']
                new_weight=np.mean(all_weights[np.where(all_weights<=current_weight)])
                all_weights[edge_list.index(e)]=new_weight
            G_updated[e[1]][e[0]]['weight']=new_weight
            updated_edges.append([e[0],e[1],new_weight])
            
        # filter_edges_not_used(G_updated,E_centrality_dict,update_Ec=True)
        # print(G.number_of_edges()-G_updated.number_of_edges())
    return Edges_T,T