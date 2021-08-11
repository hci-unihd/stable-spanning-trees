#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 11:15:34 2021

@author: enric
"""
from FGW_library.lib import fgw_barycenters
import networkx as nx
import numpy as np 

def Compute_barycenter(Trees,positions,alpha=0.01,probs_nodes=None,num_nodes_bary=None,weight_T=True,lambdas=None):
    if num_nodes_bary is None:
        num_nodes_bary=Trees[0].number_of_nodes()
    
    if weight_T:
        weight='weight'
    else:
        weight=None
    
    D_t=[]
    for counter,T in enumerate(Trees):
        if  isinstance(positions,list) and counter==0:
            nodelist=list(positions[counter].values())
        elif  counter==0:
            nodelist=list(positions.values())
        D_t.append(nx.floyd_warshall_numpy(T,nodelist=nodelist,weight=weight))
    
    P_init=None
    if not isinstance(positions,list):
        P_init= np.array(list(positions.values()))
        positions=[P_init]*len(Trees)
    else:
        positions=[np.array(list(p.values())) for p in positions]
    
    if probs_nodes is None:
        probs_nodes=[np.ones(t.number_of_nodes())/t.number_of_nodes() for t in Trees]
    if lambdas is None:
        lambdas=np.ones()/len(Trees)
    
        
    D1,C1,log=fgw_barycenters(num_nodes_bary,positions,D_t,probs_nodes,lambdas,alpha=alpha,init_X=P_init)
    
    #TODO Transform features and metric into graph