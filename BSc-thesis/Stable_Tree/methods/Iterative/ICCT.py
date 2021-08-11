#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 16:03:06 2021

@author: enfita
"""
import networkx as nx
import numpy as np
from Stable_Tree.utils.utils import order_tuple,min_edge_weight,is_networkit_graph, imported_nk
if imported_nk:
    from networkit import nxadapter

def RW_prob_matrix(W):
    D=np.diag(W.sum(1).reshape(-1)**(-1))
    return D@W



def current_centroids(G,D=None):
    sorted_list_nodes=sorted(G.nodes())
    if D is None:
        D=nx.floyd_warshall_numpy(G,nodelist=sorted_list_nodes,weight='weight')
    D_mean=np.mean(D,axis=1)
    sorted_D_mean=D_mean.argsort()[:2]
    centroids=[sorted_list_nodes[sorted_D_mean[i]] for i in [0,1]]
    return centroids,D


def argsort_centroids(G,D=None):
    sorted_list_nodes=sorted(G.nodes())
    if D is None:
        D=nx.floyd_warshall_numpy(G,nodelist=sorted_list_nodes,weight='weight')
        # print('Floyd time =%f'%(time.time()-start))
    D_mean=np.mean(D,axis=1)
    return D_mean.argsort(),D_mean,D




def update_D(D,updated_edge):
    '''
    https://cs.stackexchange.com/questions/76846/updating-the-distances-matrix-after-single-edge-was-decreased

    Parameters
    ----------
    D : TYPE
        DESCRIPTION.
    updated_edge : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''

    u,v,w=updated_edge
    
    D_new=D.copy()
    n=D.shape[0]

    for s in range(n):
        D_new[s,:]=np.minimum(D[s,:],np.minimum(D[s,u]+w+D[v,:],D[s,v]+w+D[u,:]))
    return D_new

def iterative_addition_centroids_distances(G,theta=1,proportion=1):
    '''
    Adds edge which connects the most central node not yet in the tree to the most
    central node belonging to the tree.

    Parameters
    ----------
    G : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    if imported_nk and is_networkit_graph(G):
        G=nxadapter.nk2nx(G)
    G_complete=G.number_of_edges()*2==(G.number_of_nodes()*(G.number_of_nodes()-1))
    G_updated=G.copy()
    new_weight=min_edge_weight(G)*proportion
        
    T=nx.Graph()
    Edges_T=[]
    
    
    C=nx.to_numpy_array(G_updated,nodelist=sorted(G.nodes()),weight='weight')
    argsort_centroids_list,D_mean,D=argsort_centroids(G_updated)
    node1=argsort_centroids_list[0]
    T.add_node(node1)
    D_T=np.array([[0]])
    for _ in range(1,G.number_of_nodes()):
        
        if _>1:
            D=update_D(D,(*e,new_weight))

        argsort_centroids_list,D_mean,D=argsort_centroids(G_updated,D)

        for node1 in argsort_centroids_list:
            if node1 not in T.nodes():
                if not G_complete:
                    neighbors_node1=list(set(G.neighbors(node1)).intersection(set(T.nodes())))
                    if neighbors_node1:
                        break
                else:
                    break
        
        # wmin=np.infty
        # for node in T.nodes():
        #     # w=G[node1][node]['weight']
        #     w=D[node1,node]
        #     if wmin>w:
        #         node2=node
        #         wmin=w
        if _==1:
            T_nodes=list(T.nodes())
            node2=T_nodes[0]
        else:
            if G_complete:
                T_nodes=list(T.nodes())
                Prob_constrained=np.exp(-theta*C[node1,T_nodes])#/np.sum(np.exp(-theta*C[node1,T_nodes]))
                argnode=np.argmax(Prob_constrained/np.mean(D_T,1))
                node2=T_nodes[argnode]
            else:
                T_nodes=list(T.nodes())
    
                # neighbors_node1=list(set(G.neighbors(node1)).intersection(set(T.nodes())))
                assert(T_nodes)
                
                Prob_constrained=np.exp(-theta*C[node1,neighbors_node1])#/np.sum(np.exp(-theta*C[node1,T_nodes]))
                
                indices=[T_nodes.index(v) for v in neighbors_node1]
                argnode=np.argmax(Prob_constrained/np.mean(D_T[indices],1))
    
                node2=neighbors_node1[argnode]
            
        
        
        
        e=(node1,node2)
        e=order_tuple(*e)
            
        Edges_T.append(e)
        T.add_edge(*e)
        T[e[0]][e[1]]['weight'] = G.get_edge_data(e[0],e[1])['weight']
        
        D_T=update_DT(D_T,T_nodes.index(node2),G[node1][node2]['weight'])
        
        # C=nx.to_numpy_array(G_updated,weight='weight')
        # new_weight=np.mean(C[np.where((C<=T[e[0]][e[1]]['weight']) & (C>0))])
        G_updated[e[0]][e[1]]['weight']=new_weight
        
    return Edges_T,T


def update_DT(D_T,v,w):
    d_new_node=np.expand_dims(D_T[v,:]+w,0)
    D_T=np.vstack([D_T,d_new_node])
    D_T=np.hstack([D_T,np.hstack([d_new_node,np.array([[0]])]).T])
    return D_T