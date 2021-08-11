#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 15:54:08 2020

@author: enric
"""
import networkx as nx
import numpy as np
import multiprocessing as mp
import time
from common_MCCF_tools import Local_search,Final_local_search,transition_matrix,create_solution_from_probs

#%%






def update_tau_max(rho,best_cost):
    return 1/(rho*best_cost)

def update_tau_min(tau_max,p_best,n):
    nom=tau_max*(1-p_best**(1/n))
    den=(n/2-1)*(p_best**(1/n))
    return nom/den



def update_probabilty_matrix(C,tau=None,a=1,b=2):
    if tau is not None:
        A=tau.power(a).multiply(C.power(b))
    else:
        A=C.power(b)
    return transition_matrix(A)





def ACO(G,cost_function,sources,targets=None,num_ants=200,tau_init=1e6,a=1,b=-2,rho=0.1, 
        Q=2, p_best=0.5,max_it=200,num_local_search=4,max_not_improvement_its=10):
    '''
    Implementation of the an Ant Colony Optimization algorithm from:
    Monteiro, M.S.R., Fontes, D.B.M.M. & Fontes, F.A.C.C. 
    Concave minimum cost network flow problems solved with a colony of ants. 
    J Heuristics 19, 1–33 (2013). https://doi.org/10.1007/s10732-012-9214-6

    Parameters
    ----------
    G : nx.Graph
        DESCRIPTION.
    cost_function : function
        DESCRIPTION.
    num_ants : number ants, optional
        DESCRIPTION. The default is 200.
    tau_init : float, optional
        Initialization of the pheromones. The default is 1e6.
    a : float, optional
        alpha power used to define P_ij. The default is 1.
    b : float, optional
        beta power used to define P_ij. The default is 2.
    rho : TYPE, optional
        pheromone evaporation. The default is 0.1.
    Q : TYPE, optional
        Proportionality parameter used for the update of the pheromones.
        The default is 2.
    p_best : float, optional
        Probability of the best solution. The default is 0.5.
    max_it : int, optional
        Maximum number of iterations. The default is 200.
    num_local_search : int, optional
        Number of solutions on which Local search will be applied. 
        The default is 4.
        
    max_not_improvement_its : int, optional
        Maximum number of iterations without improvement. If for such a number
        of iterations the cost does not decrease the algorithm stops. 
        Default is 10
    
    Returns
    -------
    Approximation of the minimizer of the cost function.

    '''
    
    #Initialization parameters
    
    C=nx.adjacency_matrix(G,nodelist=sorted(G.nodes()),weight='weight')
    n=G.number_of_nodes()
    tau=tau_init*nx.adjacency_matrix(G,nodelist=sorted(G.nodes()),weight=None)
    best_cost=np.infty
    
    P=update_probabilty_matrix(C=C,a=a,b=b)
    it=0
    not_improvement_its=0
    while it <=max_it and not_improvement_its<max_not_improvement_its:
        #Construct solutions
        # start=time.time()
        
        with mp.Pool(mp.cpu_count()) as pool:
            ant_solutions=[pool.apply(create_solution_from_probs,args=(P,C,sources,targets)) for i in range(num_ants)]
        
        # time1=time.time()-start
        # start=time.time()
        # ant_solutions=[create_solution_from_probs(P,C,sources,targets,nodelist) for i in range(num_ants)]
        # time2=time.time()-start
        # print('time1 %f, time2 %f'%(time1,time2))
        # start=time.time()
        with mp.Pool(mp.cpu_count()) as pool:
            cost_solutions=pool.map(cost_function,ant_solutions)
        # time3=time.time()-start
        # start=time.time()
        # cost_solutions=[cost_function(solution) for solution in ant_solutions]
        # time4=time.time()-start
        # print('time3 %f, time4 %f'%(time3,time4))
        # start=time.time()
        
        
        #Apply local search
        if num_local_search>=0:
            pos_current_best_solution=np.argmin(cost_solutions)
            list_local_search_posibilities=list(range(0,num_ants))
            list_local_search_posibilities.remove(pos_current_best_solution)
            pos_for_local_search=np.zeros(num_local_search+1,dtype=np.uint16)
            pos_for_local_search[1:]=np.random.choice(list_local_search_posibilities,
                                                  size=num_local_search,replace=False)
            pos_for_local_search[0]=pos_current_best_solution
            
            #PARALLEL
            # start=time.time()
            with mp.Pool(num_local_search+1) as pool:
                local_searched_solutions=[pool.apply(Local_search,args=(ant_solutions[pos],cost_solutions[pos],tau,C,cost_function,pos)) for pos in pos_for_local_search]
            for sol in local_searched_solutions:
                pos=sol[0]
                ant_solutions[pos]=sol[1]
                cost_solutions[pos]=sol[2]
                
            #SEQUENTIAL
            # time5=time.time()-start
            # start=time.time()
            # for pos in pos_for_local_search:
            #     tree=ant_solutions[pos]
            #     cost_tree=cost_solutions[pos]
            #     tree,cost_tree=Local_search(tree,cost_tree,tau,C,cost_function)
            #     ant_solutions[pos]=tree
            #     cost_solutions[pos]=cost_tree
            # time6=time.time()-start
            # print('time5 %f, time6 %f'%(time5,time6))
        #Update best solution
        pos_current_best_solution=np.argmin(cost_solutions)
        if best_cost>cost_solutions[pos_current_best_solution]:
            best_cost=cost_solutions[pos_current_best_solution]
            best_solution=ant_solutions[pos_current_best_solution]
            tau_max=update_tau_max(rho,best_cost)
            tau_min=update_tau_min(tau_max,p_best,n)
            not_improvement_its=0
        else:
            not_improvement_its+=1
        
        #update pheromones
        tau=(1-rho)*tau
        #reinforce pheromone values
        for e in ant_solutions[pos_current_best_solution].edges():
            u=e[0]
            v=e[1]
            tau[u,v]+=Q/best_cost
            tau[v,u]=tau[u,v]
        
        #bound tau between tau_min and tau_max
        tau.data=np.maximum(np.minimum(tau.data,tau_max),tau_min)
        it+=1
    
    #Final local search
    
    best_solution,best_cost=Final_local_search(best_solution,best_cost,tau,C,cost_function)
    print('number of iterations performed=',it)
    return best_solution,best_cost