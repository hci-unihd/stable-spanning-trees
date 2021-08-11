#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 14:52:32 2021

@author: enfita
"""
from matplotlib.image import imread
from tqdm.notebook import tqdm
from Stable_Tree.utils.utils import compute_xy_lim
import numpy as np
import networkx as nx
import random
import queue
try:
    import networkit as nk
    imported_nk=True
except ImportError:
    imported_nk=False
    
    
try:
    from sklearn.neighbors import kneighbors_graph
except:
    print('knn graph not possible. skelarn not installed')
from scipy.spatial import Delaunay
from itertools import  product
from Stable_Tree.utils.utils import ensure_dir,is_networkit_graph

def load_image(filename):
    a=imread('Data/'+filename+'.png')
    rgb_weights = [0.2989, 0.5870, 0.1140]
    
    grayscale_image = np.dot(a[...,:3], rgb_weights)>0

    return grayscale_image
    
def sample_points_from_image(n,img,Random=True):
    if not Random:
        random.seed(42)
    non_zero=np.where(img!=0)
    # non_zero=np.vstack((non_zero[0],non_zero[1])).T
    
    
    idx=random.sample(range(len(non_zero[0])),n)
    
    max_side=max(img.shape[0],img.shape[1])
    x_coord=non_zero[0][idx]*4/max_side
    y_coord=non_zero[1][idx]*4/max_side
    
    
    return x_coord,y_coord


def space_filling_tree_vertices(num_levels):
    
    vertices=np.array([[0,0]])
    roots_for_addition=queue.Queue()
    roots_for_addition.put(np.array([0,0]))
    for level in range(num_levels):
        num_iters=roots_for_addition.qsize()
        i=0
        step=np.sqrt(2)/(2**(level+1))
        while i<num_iters:
            i+=1
            v=roots_for_addition.get()
            new_vertices=add_vertices_filling_tree(v,step)
            for new_vertex in new_vertices:
                roots_for_addition.put(new_vertex)
            vertices=np.vstack([vertices,new_vertices])
    return vertices

def add_vertices_filling_tree(v,step):
    new_vertices=np.zeros((4,2))
    sign=[1,-1]
    for i,direction in enumerate(product(sign,repeat=2)):
        new_vertices[i,:]=v+step*np.array(direction)
    
    return new_vertices
                


def create_points(n=100,distribution='gauss',Random=True,seed=None,**kwargs):


    if seed is not None:
        np.random.seed(seed)
    elif not Random:
        np.random.seed(42)
    #Gauss
    if 'gauss' in distribution:
        
        
        if 'sigma1' in kwargs.keys():
            sigma1=kwargs['sigma1']
        else:
            sigma1=1
        if 'sigma2' in kwargs.keys():
            sigma2=kwargs['sigma2']
        else:
            sigma2=1
            

        x_coord=np.random.normal(0, sigma1,n)
        y_coord=np.random.normal(0, sigma2,n)
    
    #Uniform
    elif distribution=='uniform':
        
        x_coord=np.random.uniform(0, 4,n)
        y_coord=np.random.uniform(0, 1,n)
    
    elif distribution== 'circle':
        theta=np.random.uniform(0,2*np.pi,n)
        r=3.8#+np.random.normal(0,0.1,n)
        x_coord=np.cos(theta)*r
        y_coord=np.sin(theta)*r
        
    elif distribution=='line':
        y_coord=np.ones(n)
        x_coord=np.linspace(0,10,n)
        x_lim=(-0.2,10.2)
        y_lim=(0,2.5)
    elif distribution=='meshgrid':
        sqrt_n=n**.5
        assert (sqrt_n % 1 == 0)
        sqrt_n=int(sqrt_n)
        x_lim=y_lim=(0,4)
        x=y=np.linspace(x_lim[0],x_lim[1],sqrt_n)
        x_coord,y_coord=np.meshgrid(x,y)
        x_coord=x_coord.reshape(-1)
        y_coord=y_coord.reshape(-1)
        
    elif distribution=='space_filling_tree':
        num_levels=int(np.floor(np.log(1+3*n)/np.log(4)-1))
        P=space_filling_tree_vertices(num_levels)
    elif distribution== 'non_convex' or distribution== '8_circle' or distribution=='2_circles':
        img=load_image(distribution)
       
        x_coord,y_coord=sample_points_from_image(n,img,Random)

        
    #Set limits axis plots
    try:
        P
    except:
        P=np.vstack([x_coord,y_coord]).T
    if distribution !='line':
        x_lim,y_lim=compute_xy_lim(P)
        
    
    #dictionary points
    p=dict(enumerate(P))
    # p = {i: (x_coord[i], y_coord[i]) for i in range(n)}

    return p,x_lim,y_lim



def k_nearest_neighbor_graph(p,num_neighbors=10,nk_flag=False):
    points=np.array(list(p.values()))
    A=kneighbors_graph(points,n_neighbors=num_neighbors,mode='distance')
    
    if imported_nk and nk_flag:
        G=nk.Graph(n=len(p),weighted=True)
        rows,cols=A.nonzero()
        for u,v in zip(rows,cols):
            if u<v:
                G.addEdge(u,v,np.linalg.norm(np.array(p[u])-np.array(p[v])))
    else:
        G = nx.from_scipy_sparse_matrix(A)

    return G


def Delaunay_graph(p,nk_flag=False):
    
    n=len(p)
    
    points=np.array(list(p.values()))

    tri = Delaunay(points)
    simplices=np.hstack([tri.simplices,np.expand_dims(tri.simplices[:,0],axis=1)])
    
    if imported_nk and nk_flag:
        G_Delaunay=nk.Graph(n=len(p),weighted=True)
        for path in simplices:
            for i in range(3):
                G_Delaunay.addEdge(path[i],path[i+1],np.linalg.norm(np.array(p[path[i]])-np.array(p[path[i+1]])))
            
    else:
        G_Delaunay = nx.Graph()
        G_Delaunay.add_nodes_from(range(n))
        for path in simplices:
            nx.add_path(G_Delaunay, path)
            
        for e in G_Delaunay.edges():
            G_Delaunay[e[0]][e[1]]['weight']=np.linalg.norm(np.array(list(p[e[0]]))-np.array(list(p[e[1]])))                                        

    return G_Delaunay

def Complete_graph(p,nk_flag=False):
       
    if imported_nk and nk_flag:
        G=nk.Graph(n=len(p),weighted=True)
        for u in tqdm(p.keys(), desc="Create the full graph"):
            for v in range(u):
                G.addEdge(u,v,np.linalg.norm(np.array(p[u])-np.array(p[v])))
    else:
        weighted_edges=[(u,v,np.linalg.norm(np.array(p[u])-np.array(p[v]))) for u in p.keys() for v in range(u)]
        G=nx.Graph()
        
        G.add_nodes_from(range(len(p)))    
        G.add_weighted_edges_from(weighted_edges)

    return G


#TODO 
#Create graph class that includes all the internal parameters of the graph, e.g. x_lim
def Create_graph(n=100,mode='complete', distribution='uniform',sigma1=3, sigma2=1,num_neighbors=10,
                 Random=True,seed=None,nk_flag=False):
    '''
    

    Parameters
    ----------
    n : int, optional
        number of nodes. The default is 100.
    mode : str, optional
        Type of graph
            -complete: complete graph (default)
            -delaunay: complete graph
            -knn: k-nearest neighbor graph
    distribution : str, optional
        Sampling distribution of the points.
            -uniform: uniform distribution on the rectangle [0,1]x[0,4]
            -gauss: gaussian distribution centered at (0,0)
    sigma1 : float, optional
        standard deviation first dimension of the gaussian distribution. The default is 3.
    sigma2 : float, optional
        standard deviation second dimension of the gaussian distribution. The default is 1.
    num_neighbors : TYPE, optional
        number of neigbors (k) of the knn mode. The default is 10.
    Random : bool, optional
        If False the points are sampled with fixed seed =42. The default is True.
    seed : int, optional
        seed for sampling the points. The default is None.
    nk_flag : TYPE, optional
        if True the graph is a networkit graph, else is networkx graph. The default is False.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    G : nk or nx raph
        DESCRIPTION.
    p : dict
        coordinates of the points. key =node ID, value=coordinate
    x_lim : tuple
        range of the values of the points in the x coordinate . (Used for plotting)
    y_lim : tuple
        range of the values of the points in the y coordinate . (Used for plotting).

    '''
    p,x_lim,y_lim=create_points(n=n,distribution=distribution,sigma1=sigma1,sigma2=sigma2,Random=Random,seed=seed)
    # P=np.array(list(p.values()))
    
    if mode=='complete':
        G=Complete_graph(p,nk_flag=nk_flag)
    elif mode=='delaunay':
        G=Delaunay_graph(p,nk_flag=nk_flag)
    elif mode=='knn':
        G=k_nearest_neighbor_graph(p,num_neighbors,nk_flag=nk_flag)
    else:
        raise Exception('no mode '+mode)
        

    return G,p,x_lim,y_lim

#%%
def load_graph(n=100,mode='ICT',seed=0,distribution='uniform',folder=None,filename=None,**kwargs):
    if folder is None and filename is None:
        folder='Data/Trees/%s/n=%i/%s/'%(distribution,n,mode)
        filename='%s_n=%i'%(mode,n)
        for k,v in kwargs.items():
            folder+=k+'=%0.2f/'%v
            filename+='_%s=%0.2f'%(k,v)
        filename+='_%i'%seed
    
    G=nx.read_weighted_edgelist(folder+filename,nodetype=int)
    # p=np.load(folder+filename+'.npy',allow_pickle='TRUE').item()
    
    return G

def save_graph(T,n=100,mode='ICT',seed=0,distribution='uniform',folder=None,filename=None,**kwargs):
    if folder is None and filename is None:
        folder='Data/Trees/%s/n=%i/%s/'%(distribution,n,mode)
        filename='%s_n=%i'%(mode,n)
        for k,v in kwargs.items():
            folder+=k+'=%0.2f/'%v
            filename+='_%s=%0.2f'%(k,v)
        filename+='_%i'%seed
    if is_networkit_graph(T):
        T=nk.nxadapter.nk2nx(T)
    
    ensure_dir(folder)

    nx.write_weighted_edgelist(T,folder+filename)


def load_p(n=100,mode='G_complete',seed=0,distribution='uniform',folder=None,filename=None,**kwargs):
    if folder is None and filename is None:
        folder='Data/Trees/%s/n=%i/%s/'%(distribution,n,mode)
        filename='p_%s_n=%i'%(mode,n)
        for k,v in kwargs.items():
            folder+=k+'=%0.2f/'%v
            filename+='_%s=%0.2f'%(k,v)
        filename+='_%i'%seed
    
    return np.load(folder+filename+'.npy',allow_pickle='TRUE').item()

def save_p(p,n=100,mode='G_complete',seed=0,distribution='uniform',folder=None,filename=None,**kwargs):
    if folder is None and filename is None:
        folder='Data/Trees/%s/n=%i/%s/'%(distribution,n,mode)
        filename='p_%s_n=%i'%(mode,n)
        for k,v in kwargs.items():
            folder+=k+'=%0.2f/'%v
            filename+='_%s=%0.2f'%(k,v)
        filename+='_%i'%seed
    
    np.save(folder+filename+'.npy',p)