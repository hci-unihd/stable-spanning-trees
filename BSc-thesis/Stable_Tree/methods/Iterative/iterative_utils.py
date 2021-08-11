#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 16:21:11 2021

@author: enfita
"""
import numpy as np
from Stable_Tree.utils.utils import is_networkit_graph
#%%

def iscycle(e,node_classes_T):
    '''
    Checks if edge e forms a cycle in the tree T. node_classes_T is a dictionary
    that forms a partition of the current nodes of the tree. If two nodes belong
    to the same connected component they have the same representant (same key).
    If e links to nodes of the same class a cycle is formed and output is True, 
    otherwise is False

    Parameters
    ----------
    e : tuple
        candiate edge to be added to the tree.
    node_classes_T : dict
        dictionary of connected components of the tree. key=representant node of
        the connected component(node with minimum ID), value: set of the nodes
        in the same connected component of the tree as the key.

    Returns
    -------
    bool
        DESCRIPTION.
    representant_u : int
        Representant of the class of the node u of the edge e=(u,v)
    representant_v : TYPE
        Representant of the class of the node v of the edge e=(u,v)

    '''
    u,v = e

    # print(e)
    # # print(node_classes_T)
    # print('len',len(node_classes_T))
    # print('total_nodes',sum([len(node_classes_T[i]) for i in node_classes_T.keys()]))
    representant_u,representant_v=None,None
    for node_key in node_classes_T.keys():
        if representant_u is None and u in node_classes_T[node_key]:
            representant_u=node_key
        if representant_v is None and v in node_classes_T[node_key]:
            representant_v=node_key
        if representant_u==representant_v and representant_u is not None:

            return True,representant_u,representant_v
    # print('ru',representant_u,'rv',representant_v)
    return False,representant_u,representant_v
        
def update_node_classes(e,representant_u, representant_v,node_classes_T):
    '''
    Updates dictionary of the classes of the connected components of the tree 
    after adding edge e

    Parameters
    ----------
    e : tuple
        candiate edge to be added to the tree.
    representant_u : int
        Representant of the class of the node u of the edge e=(u,v)
    representant_v : TYPE
        Representant of the class of the node v of the edge e=(u,v)
    node_classes_T : dict
        dictionary of connected components of the tree. key=representant node of
        the connected component(node with minimum ID), value: set of the nodes
        in the same connected component of the tree as the key.
   


    Returns
    -------
    node_classes_T : dict
        updated node_classes_T.

    '''
    
    if representant_u is None and representant_v is None:
        node_classes_T[min(e)]=set(e)
    elif representant_v is None and representant_u is not None:
        new_representant=min(min(*e),representant_u)
        node_classes_T[new_representant]=node_classes_T[representant_u].union(set(e))
        if new_representant!=representant_u:
            del node_classes_T[representant_u]
    elif representant_u is None and representant_v is not None:
        new_representant=min(min(*e),representant_v)
        node_classes_T[new_representant]=node_classes_T[representant_v].union(set(e))
        if new_representant!=representant_v:
            del node_classes_T[representant_v]
    else:
        if representant_u>representant_v:
            representant_u,representant_v=representant_v,representant_u
    
        node_classes_T[representant_u]=node_classes_T[representant_u].union(node_classes_T[representant_v])
        node_classes_T[representant_u]=node_classes_T[representant_u].union(set(e))
        del node_classes_T[representant_v]
    return node_classes_T
    
def add_edge_tree_by_edge_centrality_dict(E_centrality_dict,Edges_T,Edges_cycle,node_classes_T,G
                                     ,Added_edges_per_iteration=1,add_all_level_set=False):
    '''
    Adds edges with higher edge centrality to the tree T.

    Parameters
    ----------
    E_centrality_dict : dict
        key=edge, value=edge centrality
    Edges_T: list
        edges added to the tree
    Edges_cycle : list
        edges found that form a cycle with the current tree T
    node_classes_T : TYPE
        dictionary of connected components of the tree. key=representant node of
        the connected component(node with minimum ID), value: set of the nodes
        in the same connected component of the tree as the key..
    G : nx or nk Graph
        DESCRIPTION.
    Added_edges_per_iteration : int
        parameter that determines how many edges are added in each iteration to
        the ICT. Default=1
    add_all_level_set : bool, optional
        If True and there are more than one candidates to be added to the
        tree T all of them are added. If False then the edge with minimum 
        weight is added. In case they had the same weight and same edge 
        centrality then the first one found is added. The default is False.

    Returns
    -------
    Edges_T : list
        updated Edges_T.
    Edges_cycle : list
        updated Edges_cycle.
    node_classes_T : dict
        update node_classes_T.

    '''
    isnk=is_networkit_graph(G)
    if isnk:
        n=G.numberOfNodes()
    else:
        n=G.number_of_nodes()
    sorted_edges=sorted(E_centrality_dict, key=E_centrality_dict.get, reverse=True)
    initial_length_Edges_T=len(Edges_T)
    for counter,e in enumerate(sorted_edges):
        if e not in Edges_T and e not in Edges_cycle:
            if isnk:
                dict_edge_weights={e:G.weight(e[0],e[1])}
            else:
                dict_edge_weights={e:G.get_edge_data(e[0],e[1])['weight']}
            k=0
            while E_centrality_dict[e]==E_centrality_dict[sorted_edges[k+counter]] :
                alternative_e=sorted_edges[k+counter]
                if isnk:
                    dict_edge_weights[alternative_e]=G.weight(alternative_e[0],alternative_e[1])
                else:
                    dict_edge_weights[alternative_e]=G.get_edge_data(alternative_e[0],alternative_e[1])['weight']
                k+=1
                if k+counter==len(sorted_edges):
                    break
            #sort by weight
            dict_edge_weights={k: v for k, v in sorted(dict_edge_weights.items(), key=lambda item: item[1])}
            for e_winner in dict_edge_weights.keys():
                # e_winner=min(dict_edge_weights, key=dict_edge_weights.get)
                
                cycle_bool,representant_u,representant_v=iscycle(e_winner,node_classes_T)
                if cycle_bool:
                    Edges_cycle.append(e_winner)
                else:
                    node_classes_T=update_node_classes(e_winner,representant_u, representant_v,node_classes_T)
                    Edges_T.append(e_winner)
                    
                    if not add_all_level_set and len(Edges_T)==min(n-1,Added_edges_per_iteration+initial_length_Edges_T):
                        return Edges_T,Edges_cycle,node_classes_T
            
            print(Added_edges_per_iteration+initial_length_Edges_T-len(Edges_T))
            if len(Edges_T)>=min(n-1,Added_edges_per_iteration+initial_length_Edges_T):
                return Edges_T,Edges_cycle,node_classes_T
                        
                        
#%%
'''
RSP utils
'''
def pos_to_coord(pos,n):

        
    col=pos % n
    row=int(pos / n)
    
    #Due to symmetry order is not important
    if row>col:
        return col,row
    else:
        return row,col

def coord_to_pos(coord,n):

    row,col=coord
    pos=col+row*n
    return pos
    
    
def add_edge_tree_by_edge_centrality_array_RSP(centrality_array,Edges_T,Edges_cycle,node_classes_T,G
                                     ,Added_edges_per_iteration=1):
    num_nodes=G.number_of_nodes()
    sorted_bet=np.argsort(centrality_array.reshape(-1))[::-1]
    #removes (i,i) edges in the diagonal which are the minimums
    sorted_bet=sorted_bet[:-num_nodes]
    #Due to symmetry of bet we take only every two elements because they are repeated
    #In concrete we take the upper triangular part of bet
    sorted_bet=sorted_bet[::2]
    
    initial_length_Edges_T=len(Edges_T)
    
    
    for counter,pos_bet_e in enumerate(sorted_bet):
        e=pos_to_coord(pos_bet_e,num_nodes)
        
        if e not in Edges_T and e not in Edges_cycle and e in G.edges():
            dict_edge_weights={e:G.get_edge_data(e[0],e[1])['weight']}
            k=0
            alternative_e=pos_to_coord(sorted_bet[k+counter],num_nodes)
            while centrality_array[e[0],e[1]]==centrality_array[alternative_e[0],alternative_e[1]] :
                if alternative_e not in Edges_T and alternative_e not in Edges_cycle and alternative_e in G.edges():
                    if G.get_edge_data(e[0],e[1])['weight']>G.get_edge_data(alternative_e[0],alternative_e[1])['weight']:
                        e=alternative_e
                k+=1
                alternative_e=pos_to_coord(sorted_bet[k+counter],num_nodes)

                if k+counter==len(sorted_bet):
                    break
            #sort by weight
            dict_edge_weights={k: v for k, v in sorted(dict_edge_weights.items(), key=lambda item: item[1])}
            for e_winner in dict_edge_weights.keys():
                # e_winner=min(dict_edge_weights, key=dict_edge_weights.get)
                
                cycle_bool,representant_u,representant_v=iscycle(e_winner,node_classes_T)
                if cycle_bool:
                    Edges_cycle.append(e_winner)
                else:
                    node_classes_T=update_node_classes(e_winner,representant_u, representant_v,node_classes_T)
                    Edges_T.append(e_winner)
                    
                    if len(Edges_T)==min(G.number_of_nodes()-1,Added_edges_per_iteration+initial_length_Edges_T):
                        return Edges_T,Edges_cycle,node_classes_T
            


def filter_edges_not_used(G,E_c,update_Ec=False):
    '''
    Removes edges whose centrality is equal to 0.

    Parameters
    ----------
    G : nx.Graph or nk.Graph
        .
    E_c : dict
        Edge centrality. key=edge, value=edge centrality

    Returns
    -------
    None.

    '''
    if is_networkit_graph(G):
        iter_text='G.iterEdges()'
        remove_edge_text='G.removeEdge(*e)'
    else:
        iter_text='G.edges()'
        remove_edge_text='G.remove_edge(*e)'
    
    if update_Ec:
        for e in list(E_c.keys()):
            if E_c[e] == 0:
                exec(remove_edge_text)
                del E_c[e]
    else:
        for e in E_c.keys():
            if E_c[e] == 0:
                exec(remove_edge_text)
                

#%%



