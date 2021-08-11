#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 11:33:24 2020

@author: enfita
"""
import numpy as np
from common_MCCF_tools import Local_search,transition_matrix,create_solution_from_probs
import multiprocessing as mp
import networkx as nx
from scipy.sparse import csr_matrix
import time
from math import ceil

#%%


def generate_chromosomes(nnz,row,col,shape,subchromosome=False):
    np.random.seed()
    data=np.random.uniform(1,10,size=nnz)
    A=csr_matrix((data, (row, col)), shape=shape)
    if subchromosome:
        return A/A.sum()
    return transition_matrix(A)
    
def Crossover(chromosome1,chromosome2,num_crossovers):
    chromosome1=chromosome1.tolil()
    chromosome2=chromosome2.tolil()
    np.random.seed()
    rows=np.random.choice(range(chromosome1.shape[1]),size=num_crossovers,replace=False)
    for row in rows:
        auxiliar=chromosome1[row,:]
        chromosome1[row,:]=chromosome2[row,:]
        chromosome2[row,:]=auxiliar
    
    return chromosome1,chromosome2

def Mutation(chromosome,num_mutations):
    chromosome=chromosome.tolil()
    np.random.seed()
    rows=np.random.choice(range(chromosome.shape[1]),size=num_mutations,replace=False)
    for sampled_row in rows:

        nnz=chromosome[sampled_row,:].nnz
        row, col=chromosome[sampled_row,:].nonzero()
        shape=chromosome[sampled_row,:].shape
        chromosome[sampled_row,:]=generate_chromosomes(nnz,row,col,shape,subchromosome=True)
    
    return chromosome



def filter_solutions(solutions,cost_solutions,N):
    best_solutions=[]
    best_costs=[]
    for i in range(int(len(solutions)/N)):
        pos=np.argmin(cost_solutions[i*N:(i+1)*N])
        best_solutions.append(solutions[i*N+pos])
        best_costs.append(cost_solutions[i*N+pos])
    return best_solutions,best_costs


def tournament_selection(cost_solutions,participants=10):
    parents = np.random.choice(range(len(cost_solutions)), size=participants,replace=False)
    costs_participants=[cost_solutions[parent] for parent in parents]

    argsorted_costs=np.argsort(costs_participants)
    positions_winners=[parents[argsorted_costs[i]] for i in range(2)]
    return positions_winners[0],positions_winners[1]


def PTbGA(G,cost_function,sources,targets=None,population_size=50,N=10,percentage_remaining=80,max_it=200,
          max_not_improvement_its=10,num_tournaments=20):
    
    
    n=G.number_of_nodes()
    #Cost matrix
    C=nx.adjacency_matrix(G,nodelist=sorted(G.nodes()),weight='weight')
    row,col=C.nonzero()
    nnz=C.nnz
    shape=C.shape

    best_cost=np.infty

    
    # start=time.time()
    # with mp.Pool(mp.cpu_count()) as pool:
    #     results=[pool.apply_async(generate_chromosomes,args=(nnz,row,col,shape)) for _ in range(population_size-1)]
    #     Generated_chromosomes = [r.get() for r in results]
    # time2=time.time()-start
    # start=time.time()
    Generated_chromosomes=[generate_chromosomes(nnz,row,col,shape) for _ in range(population_size-1)]
    # time3=time.time()-start
    Generated_chromosomes.append(transition_matrix(C))
    
    it=0
    cost_solutions=None
    not_improvement_its=0
    while it <=max_it and not_improvement_its<max_not_improvement_its:
        
        if it!=0:
            #Genetic operators
            for _ in range(num_tournaments):
                pos1,pos2=tournament_selection(cost_solutions,participants=ceil(population_size*0.4))
                chromosome1,chromosome2=Crossover(Generated_chromosomes[pos1],Generated_chromosomes[pos2],num_crossovers=ceil(n*0.1))
                Generated_chromosomes[pos1]=chromosome1
                Generated_chromosomes[pos2]=chromosome2
            
            pos_mutations=np.random.choice(range(population_size),size=ceil(population_size*0.1))
            for pos in pos_mutations:
                Generated_chromosomes[pos]=Mutation(Generated_chromosomes[pos],num_mutations=ceil(n*0.01))
                
        Generated_chromosomes_=[chromosome for chromosome in Generated_chromosomes for _ in range(N)]
        with mp.Pool(mp.cpu_count()) as pool:
                solutions=[pool.apply(create_solution_from_probs,args=(P,C,sources,targets)) for P in Generated_chromosomes_]
    
        with mp.Pool(mp.cpu_count()) as pool:
            cost_solutions=pool.map(cost_function,solutions)
    
    # print('time2 %f time3 %f'%(time2,time3))
        solutions,cost_solutions=filter_solutions(solutions,cost_solutions,N)
    

        pos_current_best_solution=np.argmin(cost_solutions)
        if best_cost>cost_solutions[pos_current_best_solution]:
            best_cost=cost_solutions[pos_current_best_solution]
            best_solution=solutions[pos_current_best_solution]
            not_improvement_its=0
        else:
            not_improvement_its+=1
    
        it+=1
    return best_solution,best_cost
    