#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 13:29:39 2021

@author: enfita
"""
import numpy as np
import networkx as nx 
import warnings
import scipy
from Stable_Tree.methods.Iterative.iterative_utils import pos_to_coord,add_edge_tree_by_edge_centrality_array_RSP
#%%

def RW_prob_matrix(W):
    D=np.diag(W.sum(1).reshape(-1)**(-1))
    return D@W


def edge_betweenness_fundamental_matrix(Z,W):
    Z_div=1/Z
    n=Z.shape[0]
    A=Z_div-n*np.diag(np.diagonal(Z_div))
    bet=W*(Z@A@Z)/(n*(n-1))
    bet=bet-n*(n-1)*np.eye(n)#Subtraction ensures that the terms in the diagonal are the minimum (used later)
    return bet


def updateZ_W_C_Pref(Z,W,C,Pref,e,new_cost,theta,theta_Pref):
    #Update C
    C[e[0],e[1]]=C[e[1],e[0]]=new_cost
    
    #Update e0 row Pref
    Pref[e[0],:]=np.exp(-theta_Pref*C[e[0],:])/np.sum(np.exp(-theta_Pref*C[e[0],:]))

    new_row_e0=np.exp(-theta*C[e[0],:])
    new_W_row_e0=new_row_e0*Pref[e[0],:]
    v0=new_W_row_e0-W[e[0],:]
    
    ##First 1-rank update 
    Zu0=-np.expand_dims(Z[:,e[0]],1)
    v0Z=np.expand_dims(v0@Z,0)
    #Apply morrison formula
    Z=Z-Zu0@v0Z/(1+v0@Zu0.reshape(-1))
    
    
    
    #Update e1 row Pref
    Pref[e[1],:]=np.exp(-theta_Pref*C[e[1],:])/np.sum(np.exp(-theta_Pref*C[e[1],:]))
    
    new_row_e1=np.exp(-theta*C[e[1],:])
    new_W_row_e1=new_row_e1*Pref[e[1],:]
    v1=new_W_row_e1-W[e[1],:]
    
    ##Second 1-rank update 
    Zu1=-np.expand_dims(Z[:,e[1]],1)
    v1Z=np.expand_dims(v1@Z,0)
    #Apply morrison formula
    Z=Z-Zu1@v1Z/(1+v1@Zu1.reshape(-1))

    
    #update W
    W[e[0],:]=new_W_row_e0
    W[e[1],:]=new_W_row_e1
    
    return Z,W,C,Pref

def updateZ_W(Z,W,e,new_weight):
    
    
    # ##First 1-rank update 
    # #init u
    # n=Z.shape[0]
    # u=np.zeros((n,1))
    # u[e[0],0]=np.sqrt(W[e[0],e[1]]-new_weight)
    
    # #Apply morrison formula
    # Zu=Z@u
    # uTZ=-u.T@Z
    # Z=Z-(Zu@uTZ)/(1+uTZ@u)
    
    
    ##Second 1-rank update 
    #init v
    # v=np.zeros((n,1))
    # v[e[1],0]=np.sqrt(W[e[0],e[1]]-new_weight)    
    
    # #Apply morrison formula
    # Zv=Z@v
    # vTZ=-v.T@Z
    # Z=Z-(Zv@vTZ)/(1+vTZ@v)

    w=np.sqrt(new_weight-W[e[0],e[1]])
    ##First 1-rank update 
    Zu0=np.expand_dims(w*Z[:,e[0]],1)
    v0Z=np.expand_dims(w*Z[e[1],:],0)
    #Apply morrison formula
    Z=Z+Zu0@v0Z/(1-w*Z[e[0],e[1]])
    
    ##Second 1-rank update 
    Zu1=np.expand_dims(w*Z[:,e[1]],1)
    v1Z=np.expand_dims(w*Z[e[0],:],0)
    #Apply morrison formula
    Z=Z+Zu1@v1Z/(1-w*Z[e[1],e[0]])


    #update weight W
    W[e[0],e[1]]=W[e[1],e[0]]=new_weight
    
    return Z,W
    


    
    
    


def Weight_fundamental_matrix(C,theta,theta_Pref=0):
    num_nodes=C.shape[0]
    W=np.zeros(C.shape)
    W[np.where(C!=0)]=np.exp(-theta*C[np.where(C!=0)])
    if theta_Pref >0:
        Pref=RW_prob_matrix(W)
        
    else:
        Pref=(C!=0)/(num_nodes-1)
        
    W=Pref*W
    assert((W@np.ones(num_nodes)<np.ones(num_nodes)).all())
    Z=np.linalg.inv(np.eye(num_nodes)-W)
    return Z,W

def RSP_centrality_tree(G,theta=15,Force_Tree=True,theta_Pref=0):

    #init
    num_nodes=G.number_of_nodes()
    nodelist=sorted(list(G.nodes()))
    C=nx.to_numpy_array(G,nodelist=nodelist,weight='weight')
    C=C/np.mean(C)
    
    
    Edges_T=[]
    Edges_cycle=[]
    node_classes_T={}
    T=nx.Graph()
    
    w_min=np.min(C+np.max(C)*np.eye(C.shape[0]))
    new_cost=w_min#/((G.number_of_nodes()-1)*G.number_of_nodes())
    new_weight=np.exp(-theta*new_cost)#.astype(np.longdouble)#/((G.number_of_nodes()-1)*G.number_of_nodes()))
    
    W=np.zeros(C.shape)
    W[np.where(C!=0)]=np.exp(-theta*C[np.where(C!=0)])
    if theta_Pref >0:
        Pref=RW_prob_matrix(W)
        
    else:
        Pref=(C!=0)/(num_nodes-1)
        
    W=Pref*W
    assert((W@np.ones(num_nodes)<np.ones(num_nodes)).all())
    Z=scipy.linalg.inv(np.eye(num_nodes)-W)#.astype(np.longdouble)
    # print(np.unique(Z))
    bet=edge_betweenness_fundamental_matrix(Z,W)*(C!=0)
    e = pos_to_coord(np.argmax(bet),num_nodes)
    
    T.add_edge(*e)
    T[e[0]][e[1]]['weight'] = G.get_edge_data(e[0],e[1])['weight']
    Edges_T.append(e)
    node_classes_T[min(e)]=set(e)

    for iteration in range(1,num_nodes-1):
        if theta_Pref>0:
            Z,W,C,Pref=updateZ_W_C_Pref(Z,W,C,Pref,e,new_cost,theta,theta_Pref)
        else:
            Z,W=updateZ_W(Z,W,e,Pref[e[0],e[1]]*new_weight)
        if not (W@np.ones(num_nodes)<np.ones(num_nodes)).all():
            warnings.warn('W*ones not <1 in iteration=%i'%iteration)
        
        # print(np.unique(Z))
        bet=edge_betweenness_fundamental_matrix(Z,W)*(C!=0)
        
        
        add_edge_tree_by_edge_centrality_array_RSP(bet,Edges_T,Edges_cycle,node_classes_T,G,Added_edges_per_iteration=1)

        e=Edges_T[-1]
        T.add_weighted_edges_from([(*e,G.get_edge_data(e[0],e[1])['weight'])])
                    
        
#        print(T.edges(data=True))
    



    return Edges_T,T