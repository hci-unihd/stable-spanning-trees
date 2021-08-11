#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 11:28:32 2020

@author: enfita
"""
import networkx as nx
import sys,os
sys.path.append(os.path.realpath('../'))
from tools import gilbert_steiner_cost,centrality_cost_Fred
from scipy.sparse import diags
import numpy as np
#%%
def Local_search(tree,cost_tree,priority,C,cost_function,pos=None):
    #order edges tree according to pheromones (priority) in ascending order

    ordered_edges_tree={e:priority[e[0],e[1]] for e in tree.edges()}
    ordered_edges_tree=dict(sorted(ordered_edges_tree.items(), key=lambda item: item[1]))
    for e_ in ordered_edges_tree.keys():
        tree.remove_edge(*e_)
        list_components=list(nx.connected_components(tree))
        #order edges not present in tree according to pheromones (priority) in descending order
        missing_edges=[(i,j)  for i in list_components[0] for j in list_components[1] if C[i,j]!=0]
        
        ordered_edges_not_tree={e:priority[e[0],e[1]] for e in missing_edges}
        ordered_edges_not_tree=dict(sorted(ordered_edges_not_tree.items(),reverse=True, key=lambda item: item[1]))
        improvement=False
        for e in ordered_edges_not_tree.keys():
            if (e[0]!=e_[0] or e[1]!=e_[1]) and (e[0]!=e_[1] or e[1]!=e_[0]):
                tree.add_edge(*e)
                tree[e[0]][e[1]]['weight']=C[e[0],e[1]]
                new_cost=cost_function(tree)
                if cost_tree>new_cost:
                    cost_tree=new_cost
                    improvement=True
                    break
                else:
                    tree.remove_edge(*e)
        if not improvement:
            tree.add_edge(*e_)
            tree[e_[0]][e_[1]]['weight']=C[e_[0],e_[1]]
    if pos is None:
        return tree,cost_tree
    else:
        return pos,tree,cost_tree
    

def Final_local_search(best_solution,best_cost,priority,C,cost_function,max_it=10):
    prev_best_cost=best_cost
    it=0
    while True:
        best_solution,best_cost=Local_search(best_solution,best_cost,priority,C,cost_function)
        if prev_best_cost<=best_cost or it>max_it:
            break
        else:
            prev_best_cost=best_cost
            it+=1
    return best_solution,best_cost
def GS_Cost_func_def(root,alpha):
    global F
    def F(tree):
        return gilbert_steiner_cost(tree,root,alpha)
    return F

def Centrality_Cost_func_def(alpha):
    global F
    def F(tree):
        return centrality_cost_Fred(tree,alpha)
    return F

def transition_matrix(A):
    D=diags(np.asarray(A.sum(1)).reshape(-1), format="csc")
    return D.power(-1)*A


def create_solution_from_probs(P,C,sources,targets=None):
    T=nx.Graph()
    n=P.shape[0]
    if not isinstance(sources,list):
        sources=[sources]
    current_sources=sources.copy()
    
    if targets is None:
        Empty=True
        T.add_nodes_from(range(n))
    else:
        Empty=False
        current_targets=targets.copy()
    
    np.random.seed()
    nodes_to_sample=list(set(T.nodes()).difference(set(current_sources)))
    while not (Empty and nx.is_tree(T)):
        while True:
            source=np.random.choice(current_sources)
            probs=P[source,nodes_to_sample].toarray().reshape(-1)
            probs_sum=probs.sum()
            if probs_sum>0:
                break
            else:
                current_sources.remove(source)
        node=np.random.choice(nodes_to_sample,p=probs/probs_sum)
        T.add_edge(source,node)
        T[source][node]['weight']=C[source,node]
        current_sources.append(node)
        nodes_to_sample.remove(node)
        if targets is not None:
            try:
                current_targets.remove(node)
            except:
                pass
            Empty=len(current_targets)==0
    return T
