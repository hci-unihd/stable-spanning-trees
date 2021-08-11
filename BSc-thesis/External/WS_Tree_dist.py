#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 11:03:37 2021

@author: enfita
"""
from Stable_Tree.utils.utils import is_networkit_graph,substitute_weights_by_widths
import numpy as np
from scipy.spatial import distance
# from pyemd import emd_with_flow,emd
import ot

#%%
def discretize_tree(T,p,resolution=10):
    discr_measure={}
    landa_ls=np.linspace(0,1,resolution)
    if is_networkit_graph(T):
        iter_edges=T.iterEdges()
        get_weight= lambda e:T.weight(e[0],e[1])
        iter_neighbours= lambda u:T.iterNeighbors(u)
    else:
        iter_edges=T.edges()
        get_weight= lambda e:T[e[0]][e[1]]['weight']
        iter_neighbours= lambda u:T.neighbors(u)
    for e in iter_edges:
        u=e[0]
        v=e[1]
        w=get_weight(e)

        aux_ls=[]
        if tuple(p[u]) not in discr_measure.keys():
            for neigh in iter_neighbours(u):
                aux_ls.append(get_weight((u,neigh)))
            discr_measure[tuple(p[u])]=np.mean(aux_ls)
        
        aux_ls=[]
        if tuple(p[v]) not in discr_measure.keys():
            for neigh in iter_neighbours(v):
                aux_ls.append(get_weight((v,neigh)))
            discr_measure[tuple(p[v])]=np.mean(aux_ls)
        
        for landa in landa_ls[1:-1]:
            interpolation=landa*(p[u]-p[v])+p[v]
            discr_measure[tuple(interpolation)]=w
    
    return discr_measure


def Dist_matrix(dm1,dm2):
    coords1=list(dm1.keys())
    coords2=list(dm2.keys())
    dist_matrix=distance.cdist(coords1, coords2, 'euclidean')
    
    # dist_matrix=np.zeros((len(dm1),len(dm2)))
    # for i,coord1 in enumerate(dm1.keys()):
    #     coord1=np.array(coord1)
    #     for j,coord2 in enumerate(dm2.keys()):
    #         coord2=np.array(coord2)
    #         dist_matrix[i,j]=np.linalg.norm(coord1-coord2)
    return dist_matrix

def dist_edge(u1,v1,u2,v2):
    '''
    Computes the squared root of the average squared distance between the edges
    represented as segments
    
    sqrt(integral(||e1(t)-e2(t)||²,0,1))

    Parameters
    ----------
    u1 : TYPE
        coordinates of node 1 of edge e1.
    v1 : TYPE
        coordinates of node 2 of edge e1.
    u2 : TYPE
        coordinates of node 1 of edge e2.
    v2 : TYPE
        coordinates of node 2 of edge e2.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    u1=np.array(u1)
    u2=np.array(u2)
    v1=np.array(v1)
    v2=np.array(v2)
    #set orientation edges
    if np.linalg.norm(v1-u1)+np.linalg.norm(v2-u2)>np.linalg.norm(v2-u1)+np.linalg.norm(v1-u2):
        v1,v2=v2,v1
    
    #slopes edges
    m1=np.array(u1)-np.array(u2)
    m2=np.array(v1)-np.array(v2)
    
    a=m1-m2
    b=v1-v2
    #integral
    d=(a[0]**2+a[1]**2)/3+a[0]*b[0]+a[1]*b[1]+b[0]**2+b[1]**2
    return np.sqrt(d)
    
def Dist_matrix_edges(T1,T2,p1,p2):
    m=len(p1)-1
    if is_networkit_graph(T1):
        T1_iter_edges=T1.iterEdges()
    else:
        T1_iter_edges=T1.edges()
        
    if is_networkit_graph(T2):
        T2_iter_edges=list(T2.iterEdges())
    else:
        T2_iter_edges=T2.edges()
    dist_matrix=np.empty((m,m))
    for i,e1 in enumerate(T1_iter_edges):
        for j,e2 in enumerate(T2_iter_edges):
            dist_matrix[i,j]=dist_edge(p1[e1[0]],p1[e1[1]],p2[e2[0]],p2[e2[1]])


    return dist_matrix


def obtain_hist(dm):
    hist=np.array(list(dm.values()))
    hist/=np.sum(hist)
    return hist    

def EMD_dist_tree_discretized(T1,T2,p1,p2, widths_T1=None,widths_T2=None
                  ,resolution=5,return_flow=False,reg=0.1):
    '''
    Computes the earth mover's distance (EMD) between two trees. The edges of the 
    trees are discretized by sampling equidistant points in the segment forming
    the edge. The mass of a point in the discretized tree is proportional to the
    betweenness centrality of the edge it belongs to. For the points that belong
    to more than one edge is proportinal to the average of the centralities.

    Parameters
    ----------
    T1 :  nx or nk graph
        Tree 1.
    T2 : nx or nk graph
        Tree 2.
    p1 : dict
        coordinates of the nodes of T1.
    p2 : dict
        coordinates of the nodes of T2.
    widths_T1 : list, optional
        centralities of the edges of T1. They must have the same order as the
        edges of the tree. If None they are computed. The default is None.
    widths_T2 : list, optional
        Centralities of the edges of T1. They must have the same order as the
        edges of the tree. If None they are computed. The default is None.
    resolution : int, optional
        Number of points sampled per edge. The default is 5.
    return_flow : bool, optional
        if True, it returns the flow matrix. The default is False.
    reg : TYPE, optional
        regularization term of the EMD. The default is 0.1.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    if widths_T1 is not None:
        substitute_weights_by_widths(T1,widths_T1)
    if widths_T1 is not None:
        substitute_weights_by_widths(T2,widths_T2)
        
    dm1=discretize_tree(T1,p1,resolution=resolution)
    dm2=discretize_tree(T2,p2,resolution=resolution)
    hist1= obtain_hist(dm1)
    hist2= obtain_hist(dm2)
    dist_matrix=Dist_matrix(dm1,dm2)

    if reg==0:
        flow=ot.emd(hist1,hist2,dist_matrix)
    else:
        flow=ot.sinkhorn(hist1,hist2,dist_matrix,reg)
    EMD_dist=np.sum(flow*dist_matrix)
    if return_flow:
        return EMD_dist,flow,dm1,dm2
    else:
        return EMD_dist


def EMD_dist_tree_edge(T1,T2,p1,p2, widths_T1,widths_T2
                  ,return_flow=False,reg=0.1):

        
    '''
    Computes the earth mover's distance (EMD) between two trees. Instead of 
    discretize the edges, the cost matrix is computed between the edges as a 
    whole. The cost of transportation between two edges is equal to the square 
    root of the average squared distance between the segments representing the 
    edges.
    
    Parameters
    ----------
    T1 :  nx or nk graph
        Tree 1.
    T2 : nx or nk graph
        Tree 2.
    p1 : dict
        coordinates of the nodes of T1.
    p2 : dict
        coordinates of the nodes of T2.
    widths_T1 : list, optional
        centralities of the edges of T1. They must have the same order as the
        edges of the tree. If None they are computed. The default is None.
    widths_T2 : list, optional
        Centralities of the edges of T1. They must have the same order as the
        edges of the tree. If None they are computed. The default is None.
    return_flow : bool, optional
        if True, it returns the flow matrix. The default is False.
    reg : TYPE, optional
        regularization term of the EMD. The default is 0.1.
    '''
    hist1= np.array(widths_T1)/sum(widths_T1)
    hist2= np.array(widths_T2)/sum(widths_T2)
    dist_matrix=Dist_matrix_edges(T1,T2,p1,p2)
    if reg==0:
        flow=ot.emd(hist1,hist2,dist_matrix)
    else:
        flow=ot.sinkhorn(hist1,hist2,dist_matrix,reg)
    EMD_dist=np.sum(flow*dist_matrix)
    if return_flow:
        return EMD_dist,flow
    else:
        return EMD_dist
#%%
import networkx as nx
def plot_flow_discretized(flow,dm1,dm2,scale=2000):
    n=len(dm1)
    G_=nx.Graph()
    p_={}
    for i,node1 in enumerate(dm1.keys()):
        if i not in p_.keys():
            p_[i]=node1
        for j,node2 in enumerate(dm2.keys()):
            j=n+j
            if flow[i,j-n]>1e-8:
                G_.add_edge(i,j,weight=flow[i,j-n]*scale)
            if j not in p_.keys():
                p_[j]=node2
    widths= list(nx.get_edge_attributes(G_,'weight').values())

    return G_,p_,widths



def plot_flow_edge(flow,T1,T2,p1,p2,scale=2000):
    if is_networkit_graph(T1):
        T1_iter_edges=T1.iterEdges()
    else:
        T1_iter_edges=T1.edges()
        
    if is_networkit_graph(T2):
        T2_iter_edges=list(T2.iterEdges())
    else:
        T2_iter_edges=T2.edges()
    G_=nx.Graph()
    p_={}
    m=len(p1)-1
    for i,e1 in enumerate(T1_iter_edges):
        middle_e1=0.5*(np.array(p1[e1[0]])+np.array(p1[e1[1]]))
        if i not in p_.keys():
            p_[i]=middle_e1
        for j,e2 in enumerate(T2_iter_edges):
            middle_e2=0.5*(np.array(p2[e2[0]])+np.array(p2[e2[1]]))
            j=m+j
            if flow[i,j-m]>1e-8:
                G_.add_edge(i,j,weight=flow[i,j-m]*scale)
            if j not in p_.keys():
                p_[j]=middle_e2
    widths= list(nx.get_edge_attributes(G_,'weight').values())

    return G_,p_,widths