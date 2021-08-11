#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 11:49:02 2021

@author: enric
"""



import os,sys
sys.path.append(os.path.realpath('../'))
import numpy as np
import networkx as nx
import random


#%%

#CONTRACTION TREES
def Sample_tree_by_contractions(G,mu=10):
    
    
    #initialization 
    Edges_class={}
    Edges_class_weights={}
    Edges_weights={}
    
    for e in G.edges():
        e=tuple(sorted(e))
        Edges_class[e]=set({e})
        Edges_class_weights[e]=np.exp(-mu*G.get_edge_data(*e)['weight'])
        Edges_weights[e]=np.exp(-mu*G.get_edge_data(*e)['weight'])
    T=nx.Graph()
    
    G_contracted=G.copy()
    
    for _ in range(G.number_of_nodes()-1):
        #sample edge
        contracted_edge,contracted_edge_class=Sample_edge(Edges_weights,Edges_class,
                                                          Edges_class_weights)

        #add sampled edge to the tree
        T.add_edge(*contracted_edge,weight=G.get_edge_data(contracted_edge[0],contracted_edge[1])['weight'])
        
        #Contract sampled edge and update weights and classes
        G_contracted,Edges_class,Edges_class_weights=Update_contracted_graph(G_contracted,
                                                                             contracted_edge_class,
                                                                             Edges_class,
                                                                             Edges_class_weights)
    
    assert(nx.is_tree(T))
    return T

def Sample_edge(Edges_weights,Edges_class,Edges_class_weights):
    #sample edge contracted graph
    contracted_edge_class=random.choices(list(Edges_class_weights.keys()), 
                                   weights = list(Edges_class_weights.values()),
                                   k = 1)[0]
    
    
    #sample edge from the class of the contracted edge in the contracted graph
    weigths_class_contracted_edge=[Edges_weights[e] for e in Edges_class[contracted_edge_class]]
    contracted_edge=random.choices(list(Edges_class[contracted_edge_class]), 
                                   weights = weigths_class_contracted_edge, k = 1)[0]
    return contracted_edge,contracted_edge_class


def Update_contracted_graph(G,contracted_edge,Edges_class,
                            Edges_class_weights):
    
    G_contracted=nx.contracted_edge(G,contracted_edge,False)
    del Edges_class[contracted_edge]
    del Edges_class_weights[contracted_edge]
    
    for node in G.nodes():
        if contracted_edge[0] in G.neighbors(node) and contracted_edge[1] in G.neighbors(node):
            e0=tuple(sorted([contracted_edge[0],node]))
            e1=tuple(sorted([contracted_edge[1],node]))
            
            #update weight
            Edges_class_weights[e0]=Edges_class_weights[e0]+Edges_class_weights[e1]
            del Edges_class_weights[e1]
            
            #update edge class
            Edges_class[e0]=Edges_class[e0].union(Edges_class[e1])
            del Edges_class[e1]
            
    return G_contracted,Edges_class,Edges_class_weights

#%%
#JITTER



def Sample_tree_by_jitter(p,jitter):
    #Add noise
    P=np.array(list(p.values()))
    P_jitter=P+np.random.normal(0,jitter,size=P.shape)
    p_jitter=dict(enumerate(P_jitter))
    # compute new limits
    
    
    
    G_noise=nx.Graph()

    weighted_edges=[(u,v,np.linalg.norm(np.array(P_jitter[u,:])-np.array(P_jitter[v,:]))) for u in range(P.shape[0]) for v in range(u)]
    
    G_noise.add_weighted_edges_from(weighted_edges)
    
    return nx.minimum_spanning_tree(G_noise, weight='weight'),p_jitter
    
    
#%%
#mST Random Subgraph

def Sample_random_subgraph_mST(G,p,percentage_sampled_nodes_):
    n=G.number_of_nodes()
    perc=random.randint(*percentage_sampled_nodes_)
    nodes_subgraph=random.sample(range(n),int(perc*n/100))
    G_sub=G.subgraph(nodes_subgraph)
    
    p_={i:p[i] for i in sorted(nodes_subgraph)}
    return nx.minimum_spanning_tree(G_sub, weight='weight'),p_