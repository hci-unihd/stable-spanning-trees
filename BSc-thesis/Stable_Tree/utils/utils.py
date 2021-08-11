import matplotlib.pyplot as plt
import matplotlib.colors as colors

import networkx as nx
import numpy as np

from tqdm.notebook import tqdm

try:
    import cv2
except ImportError:
    print('cv2 package not present. No movies possible')
    
try:
    import networkit as nk
    imported_nk=True
except ImportError:
    imported_nk=False
    print('networkit package not present. networkx will be used instead')
import os
import glob


#%%
'''
Plot graph
'''

cmap_edges = colors.ListedColormap ( np.concatenate([np.zeros((1,3)),np.random.rand ( 256,3)]))
cmap_nodes = colors.ListedColormap ( np.concatenate([np.array([[0,0.75,1]]),np.random.rand ( 256,3)]))

def compute_figsize(x_lim,y_lim):
    x_diff=x_lim[1]-x_lim[0]
    y_diff=y_lim[1]-y_lim[0]
    scale=x_diff/y_diff
    if scale<1:
        figsize=(np.round(16*scale).astype(np.int8),16)
    else:
        figsize=(16,np.round(16/scale).astype(np.int8))
    return figsize

def compute_xy_lim(P):
    if isinstance(P,dict):
        P=np.array(list(P.values()))
    else:
        assert(isinstance(P,np.ndarray))
        
    X_offset=(np.max(P[:,0])-np.min(P[:,0]))/10
    Y_offset=(np.max(P[:,1])-np.min(P[:,1]))/10
    x_lim=(np.min(P[:,0])-X_offset,np.max(P[:,0])+X_offset)
    y_lim=(np.min(P[:,1])-Y_offset,np.max(P[:,1])+Y_offset)
    
    return x_lim,y_lim

def plot_graph(G,p,k1=1,k2=1,counter_plot=0,x_lim=None,y_lim=None,widths=None,title='',
               edge_colors=None,cmap_edges=None,node_colors=None,cmap_nodes=None,node_size=1,return_ax=False,
               new=True,label_nodes=False,label_edges=False,axis=True,vmin_edge=None,
               vmax_edge=None,figsize=None,edgelist=None,return_fig=False):
    '''
    

    Parameters
    ----------
    G : nx or nk graph
        DESCRIPTION.
    p : dict
        coordinates of the points. key =node ID, value=coordinate
    k1 : int, optional
        number of rows for the subplot. The default is 1.
    k2 : int, optional
        number of columns for the subplot. The default is 1.
    counter_plot : TYPE, optional
        DESCRIPTION. The default is 0.
    x_lim : tuple
        range of the values of the points in the x coordinate. 
    y_lim : tuple
        range of the values of the points in the y coordinate.
    widths : list, optional
        widths of the edges. It must have the same order as the edges in the graph G.
        The default is None.
    title : str, optional
        Title of the plot. The default is ''.
    edge_colors : list, optional
        color values of the edges. It must have the same order as the edges in the graph G.
        The default is None.
    cmap_edges : plt.colormap, optional
        colormap of the edges. The default is None.
    node_colors : TYPE, optional
        color values of the nodes. It must have the same order as the nodes in the graph G.
    cmap_nodes : plt.colormap, optional
        colormap of the nodes. The default is None.
    node_size : int, optional
        size of the nodes. The default is 1.
    return_ax : bool, optional
        if True returns axis of the figure. The default is False.
    new : bool, optional
        if True, a new figure is created. The default is True.
    label_nodes : bool, optional
        if True, ID of nodes is shown. The default is False.
    label_edges :bool, optional
        if True, weight of the edges is shown The default is False.
    axis : bool, optional
        If True, the axis coordinate is shown. The default is True.
    vmin_edge : float, optional
        minimum value of the colormap of the edges. The default is None.
    vmax_edge : float, optional
        maximum value of the colormap of the edges. The default is None.
    figsize : TYPE, optional
        DESCRIPTION. The default is None.
    edgelist : list, optional
        list with edges to be plotted. The default is None.
    return_fig : bool, optional
        returns figure if True. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    if x_lim is None or y_lim is None:
        x_lim,y_lim=compute_xy_lim(p)
    if figsize is None:
        figsize=compute_figsize(x_lim,y_lim)
        
    try:
        G=nk.nxadapter.nk2nx(G)
    except:
        pass
    
    if edgelist==None:
        edgelist=G.edges()
        
    if counter_plot==0 and new:
        if return_ax:
            fig=plt.figure(figsize=figsize,dpi=100)
        else:
            plt.figure(figsize=figsize,dpi=100)
    if k1!=1 or k2!=1:
        ax=plt.subplot(int(k1),int(k2),counter_plot+1,axisbelow=True)
    else:
        ax=plt.axes()
    plt.title(title)

    if G.number_of_nodes()<=30 or label_nodes:
        
        node_size=1
        nx.draw_networkx_labels(G,p)
        nx.draw_networkx_nodes(G,p)
        nx.draw_networkx_edges(G,pos=p,width=widths,edge_color=edge_colors,edge_cmap=cmap_edges,
                               edge_vmin=vmin_edge,edge_vmax=vmax_edge,edgelist=edgelist)
        if label_edges:
            E_labels = dict([((u,v,), f"{d['weight']:.3f}") for u,v,d in G.edges(data=True)])
            nx.draw_networkx_edge_labels(G,p,edge_labels=E_labels)
        # nx.draw_networkx_nodes(G,p,with_labels=True)
    else:
        nx.draw_networkx_edges(G,pos=p,width=widths,edge_color=edge_colors,edge_cmap=cmap_edges,
                               edge_vmin=vmin_edge,edge_vmax=vmax_edge,edgelist=edgelist)
        # else:
        #     nx.draw_networkx_edges(G,pos=p,edge_color=edge_colors,edge_cmap=cmap_edges)#,width=weights_T_Delaunay_sp_centrality)
        nx.draw_networkx_nodes(G,pos=p, node_size=node_size,ax=ax,node_color=node_colors,cmap=cmap_nodes)
        # nx.draw_networkx(G_Delaunay,pos=p, node_size=10,ax=ax,with_labels=True)
    
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.set(xlim=x_lim, ylim=y_lim)
    ax.set_aspect('equal')
    if k1==1 and k2==1:
        plt.tight_layout()
    plt.draw()
    if axis==False:
        plt.axis('off')
    #plt.pause(0.00001)
    
    if return_ax and return_fig and counter_plot==0 and new:
        return fig, ax
    elif return_ax:
        return ax


#%%
'''
Width edges tree based on centrality
'''

def count_nodes_sides(adj_list, cur, par,nodes_side_count=None):
    '''
    count nodes side edges

    Parameters
    ----------
    adj_list : TYPE
        DESCRIPTION.
    cur : TYPE
        DESCRIPTION.
    par : TYPE
        DESCRIPTION.
    nodes_side_count : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    nodes_side_count : TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    '''
    if nodes_side_count==None:
        nodes_side_count={}
        for v in adj_list.keys():
            for u in adj_list[v]:
                e=tuple(sorted((u,v)))
                nodes_side_count[e]=0
    
    e=tuple(sorted((par,cur)))
    #If current nodes is leaf node and is not the node provided by the calling function then return 1 
    if len(adj_list[cur]) == 1 and par != 0:
        nodes_side_count[e] = 1
        return nodes_side_count,nodes_side_count[e] 
    count = 1
    #count the number of nodes recursively for each neighbor of current node.
    for neighbor in adj_list[cur]:
        if neighbor != par:
            nodes_side_count,count_aux= count_nodes_sides(adj_list, neighbor, cur,nodes_side_count)
            count+=count_aux
    
    # while returning from recursion assign the result obtained in the edge[][] matrix.
    
    nodes_side_count[e] = count
    return nodes_side_count,nodes_side_count[e] 

def compute_nodes_side_count(tree):
    '''
    Count nodes side edges

    Parameters
    ----------
    tree : TYPE
        DESCRIPTION.

    Returns
    -------
    nodes_side_count : TYPE
        DESCRIPTION.

    '''
    isnk=is_networkit_graph(tree)

    if isnk:
        adj_list={node: [neigh for neigh in tree.iterNeighbors(node)] for node in tree.iterNodes()}
        
        nodes_side_count={tuple(sorted(e)):0 for e in tree.iterEdges()}
        for node in tree.iterNodes():
            if tree.degree(node)==1:
                par=node
                cur=next(tree.iterNeighbors(node))
                break
    else:
        adj_list=nx.to_dict_of_lists(tree)
        nodes_side_count={tuple(sorted(e)):0 for e in tree.edges()}
        for node in tree:
            if tree.degree(node)==1:
                par=node
                cur=next(tree.neighbors(node))
                break
    nodes_side_count,_=count_nodes_sides(adj_list,cur,par,nodes_side_count)
    return nodes_side_count





def centrality_weights_tree(tree,norm=True,max_width=1):
    '''
     Computes the edge centrality of the edges in a tree. This is used for the
    widths when plotting. The edge centrality on a tree coincides with the number
    of nodes to the left of the edge times the number of nodes to the right of the
    edge of e: c(e)=l(e)*r(e)

    Parameters
    ----------
    tree : TYPE
        DESCRIPTION.
    norm : TYPE, optional
        DESCRIPTION. The default is True.
    max_width : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    nodes_side_count : TYPE
        DESCRIPTION.
    widths : TYPE
        DESCRIPTION.

    '''
    nodes_side_count=compute_nodes_side_count(tree)
    widths=compute_centrality_edges(tree,nodes_side_count)
    if norm==True:
        widths=max_width*np.array(widths)/max(widths)
    return nodes_side_count,widths


    
    
def compute_centrality_edges(tree,nodes_side_count,alpha=1,weight=False):
    
    
    widths=[]
    if is_networkit_graph(tree):
        n=tree.numberOfNodes()
    
        for e in tree.iterEdges():
            e=tuple(sorted(e))
            n1=nodes_side_count[e]
            n2=n-n1
            if weight:
                w=tree.weight(e[0],e[1])
            else:
                w=1
            #widths.append(np.log(n1)+np.log(n2))
            widths.append(w*(n1*n2)**alpha)
    else:
        n=tree.number_of_nodes()
    
        for e in tree.edges():
            e=tuple(sorted(e))
            n1=nodes_side_count[e]
            n2=n-n1
            if weight:
                w=tree.get_edge_data(e[0],e[1])['weight']
            else:
                w=1
            #widths.append(np.log(n1)+np.log(n2))
            widths.append(w*(n1*n2)**alpha)
    return widths


#%%
def tree_node_centrality(tree):
    if is_networkit_graph(tree):
        return tree_node_centrality_nk(tree)
    else:
        return tree_node_centrality_nx(tree)
def tree_node_centrality_nx(tree):
    centrality=nx.betweenness_centrality(tree,normalized=False)
    n=tree.number_of_nodes()
    nodes_centrality=np.zeros(n)
    list_nodes=list(tree.nodes())
    for node in tree:
        u= list_nodes.index(node)
        nodes_centrality[u]=n-1+centrality[u]
    return nodes_centrality

def tree_node_centrality_nk(tree):
    centrality=nk.centrality.Betweenness(tree,normalized=False).run().scores()
    n=tree.numberOfNodes()
    nodes_centrality=np.zeros(n)
    for u in tree.iterNodes():
        nodes_centrality[u]=n-1+centrality[u]
    return nodes_centrality
    
#%%


#%%

def bounce_update(q,v,or_v=None):
    
    if or_v is None:
        or_v=v.copy()

    if q[0]+v[0]>4:
        v[0]=4-(q[0]+v[0])
        q[0]=4
        or_v[0]=-or_v[0]
        return bounce_update(q,v,or_v)
    elif q[0]+v[0]<0:
        
        v[0]=-(q[0]+v[0])
        q[0]=0
        or_v[0]=-or_v[0]
        return bounce_update(q,v,or_v)
    elif q[1]+v[1]>1:
        
        v[1]=1-(q[1]+v[1])
        q[1]=1
        or_v[1]=-or_v[1]
        return bounce_update(q,v,or_v)
    elif q[1]+v[1]<0:
        
        v[1]=-(q[1]+v[1])
        q[1]=0
        or_v[1]=-or_v[1]
        return bounce_update(q,v,or_v)
    else:
        q=q+v
    return q,or_v
    
def update_p(P,frame,directions,mode='peridic'):
    if mode=='peridic':
        P=P+frame*directions
        P[:,0]=P[:,0]%4
        P[:,1]=P[:,1]%1
    elif mode=='bouncing':
        for i in range(P.shape[0]) :
            P[i,:],directions[i,:]=bounce_update(P[i,:],directions[i,:])

    return P,directions/np.linalg.norm(directions,axis=1)[:,np.newaxis]


def Langevin(P,stepsize=0.01,sigmas=[1,4]):
    n=P.shape[0]
    D=np.diag([1/i for i in sigmas])
    directions=np.random.normal(0, 1, size=(n,2))
    P_updated=P+stepsize*(-D@P.T).T+np.sqrt(2*stepsize)*directions
    return P_updated
def move_nodes_direction(p,stepsize=0.01,directions=None,mode='bouncing',distribution='uniform',
                         sigmas=[1,4]):
    P=np.array(list(p.values()))
    if 'gaussian' in distribution:
        P_updated=Langevin(P,stepsize=stepsize,sigmas=sigmas)
    elif distribution=='uniform':
        n=len(p)
        if directions is None:
            np.random.seed(42)
            directions=np.random.normal(0, 1, size=(n,2))
            directions=directions/np.linalg.norm(directions,axis=1)[:,np.newaxis]
        
        P_updated,directions=update_p(P,1,stepsize*directions,mode=mode)
    p_updated=dict(enumerate(P_updated))
    return p_updated,directions
#%%

def is_networkit_graph(G):
    try:
        G.numberOfEdges()
        return True
    except:
        return False

def check_iterator(a):
    try:
        iterator = iter(a)
        return True
    except TypeError:
        # not iterable
        return False
    
def create_movie(folders,titles):
    fps = 2
    if isinstance(folders, str) :
        folders=[folders]
    if isinstance(titles, str):
        titles=[titles]
        
    for folder,title in zip(folders,titles):

    
        title=title.split('/')[-1]
        video_name=folder+title+'.avi'
        
        
        images = [img for img in sorted(glob.glob(folder+'*.png'))]
        print(images)
        frame = cv2.imread(images[0])
#         cv2.imshow('video', frame)
        
        height, width, layers = frame.shape
        
        video = cv2.VideoWriter(video_name,cv2.VideoWriter_fourcc(*'DIVX'), fps, (width,height))
     
        for image in images:
            video.write(cv2.imread(image))
        video.release()
        
        
def formation_tree(T,E_T,p,G,x_lim,y_lim,title_save,widths=None,title_fig=None,background_G=False,
                   seq_edge_centralities=None,figsize=(16,4)):
    if is_networkit_graph(T):
        T=nk.nxadapter.nk2nx(T)
    plt.switch_backend('Agg')
    if title_fig is None:
        title_fig=', '.join(title_save.split('/')[-1].split('_'))
        
    if '/' in title_save:
        folder='/'.join(title_save.split('/')[:-1])+'/'
        title=title_save.split('/')[-1]
    else:
        folder=''
    n=len(p)
    try:
        m=G.numberOfEdges()
    except:
        m=G.number_of_edges()
        
    folder='Movies/Formation_tree/n=%i/'%n+folder
    ensure_dir(folder)

    if background_G and seq_edge_centralities is None:
        if 2*m<n*(n-1):
            width_full=100/m
        else:
            width_full=0.05
        ax=plot_graph(G,p,k1=1,k2=1,counter_plot=0,
                x_lim=x_lim, y_lim=y_lim,
                title=title_fig,node_size=0,edge_colors='r',widths=width_full,
                new=True,return_ax=True,figsize=figsize)
    
    P=np.array(list(p.values()))
    
    if widths is None and not background_G or seq_edge_centralities is None:
        widths=2
    else:
        widths_=np.zeros(m)
     
        
    ploted_scatter=False
    
    # iter_edges='.iterEdges()' if is_networkit_graph(T) else '.edges()'
    
    for i in range(n-1):
        # T.add_edge(*e)
        if background_G and seq_edge_centralities is not None:
            for j,e in enumerate(E_T):
                if j<=i:
                    widths_[j]=seq_edge_centralities[i][e] 
                else: 
                    break

        # T[e[0]][e[1]]['weight']=G[e[0]][e[1]]['weight']
        # p_={k: v for k, v in p.items() if k in T.nodes()}
        
        if background_G and seq_edge_centralities is not None:
            plt.close()
            ax=plot_graph(G,p,k1=1,k2=1,counter_plot=0,
                x_lim=x_lim, y_lim=y_lim,
                title=title_fig,node_size=0,edge_colors='r',
                widths=list(seq_edge_centralities[i].values()),
                edgelist=list(seq_edge_centralities[i].keys()),
                new=True,return_ax=True,figsize=figsize)
            ax.scatter(P[:,0],P[:,1],c='g',s=1)
            ploted_scatter=True
        if i==0:
            ax=plot_graph(T,p,k1=1,k2=1,counter_plot=0,
            x_lim=x_lim, y_lim=y_lim,widths=widths_,
            title=title_fig,node_size=0,edge_colors='b',new= not background_G,
            return_ax=True,figsize=figsize,edgelist=E_T)
        else:
            ax=plot_graph(T,p,k1=1,k2=1,counter_plot=0,
                x_lim=x_lim, y_lim=y_lim,widths=widths_,
                title=title_fig,node_size=0,edge_colors='b',new=False,return_ax=False,
                edgelist=E_T)
        if i==0 and not ploted_scatter:
           ax.scatter(P[:,0],P[:,1],c='g',s=1)

        title_end='_%s.png'%(str(i).zfill(len(str(G.number_of_nodes()))))
        plt.tight_layout()
        plt.savefig(folder+title+title_end,dpi=100)
    
    create_movie(folder,title)
    plt.switch_backend('QT5Agg')
    
    
#%%
'''
Transform weights to break triangle inequality
'''
def piecewise_expx_xsquared(x,gamma=0.5):
    th=np.log(2)/(gamma*(np.sqrt(2)-1))
    return np.piecewise(x,[x<th,x>=th],[lambda x : exponential(x,gamma),lambda x: (exponential(th,gamma)/th**2)*x**2])
    
def squared(x,gamma=1):
    return x**2   

def exponential(x,gamma):
    return np.exp(gamma*x)-1
def break_triangle_inequality(G,function='piecewise_exp_squared',gamma=1):
    if function=='squared':
        func=squared
    elif function=='exponential':
        func=exponential
    elif function=='piecewise_exp_squared':
        func=piecewise_expx_xsquared
        
    if is_networkit_graph(G):
        for u,v,w in tqdm(G.iterEdgesWeights(), desc="Break triangle equation"):
            G.setWeight(u, v ,func(w,gamma))
    else:
        for u,v,w in G.edges(data=True):
            G[u][v]['weight']=func(w['weight'],gamma)
#%%
def centroid_graph(G):
    if is_networkit_graph(G):
        close = nk.centrality.Closeness(G, False, nk.centrality.ClosenessVariant.Generalized)
        close.run()
        return close.ranking() [:1][0]
    else:
        close=nx.closeness_centrality(G,distance='weight')
        return max(close, key=close.get)
#%%
'''
Miscellaneaous
'''
from numpy import inf

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
        
def min_edge_weight(G):
    w_min=inf
    try:
        for u,v,w in G.iterEdgesWeights():
            if w_min>w:
                w_min=w
    except:
        for u,v,weight in G.edges(data=True):
            w=weight['weight']
            if w_min>w:
                w_min=w

    return w_min

def max_edge_weight(G):
    w_max=-inf
    try:
        for u,v,w in G.iterEdgesWeights():
            if w_max<w:
                w_max=w
    except:
        for u,v,weight in G.edges(data=True):
            w=weight['weight']
            if w_max<w:
                w_max=w
    return w_max

def order_tuple(s,t):
    if s<t:
        return s,t
    else:
        return t,s


def substitute_weights_by_widths(T,widths):
    if is_networkit_graph(T):
        for i,e in enumerate(T.iterEdges()):
            T.setWeight(e[0],e[1],widths[i])
    else:
        for i,e in enumerate(T.edges()):
            T[e[0]][e[1]]['weight']=widths[i]

def copy_nkGraph(G,keep_ids=False):
    if G.upperNodeIdBound()==G.numberOfNodes():
        G_copy=nk.Graph(n=G.numberOfNodes(),weighted=True)
        for u,v,w in G.iterEdgesWeights():
            G_copy.addEdge(u,v,w)
    elif keep_ids:
        G_copy=nk.Graph(n=G.upperNodeIdBound(),weighted=True)
        list_nodes=[node for node in G.iterNodes()]
        for u,v,w in G.iterEdgesWeights():
            G_copy.addEdge(u,v,w)
        for node in G_copy.iterNodes():
            if node not in list_nodes:
                G_copy.removeNode(node)
    else:
        list_nodes=[node for node in G.iterNodes()]
        G_copy=nk.Graph(n=G.numberOfNodes(),weighted=True)
        for u,v,w in G.iterEdgesWeights():
            G_copy.addEdge(list_nodes.index(u),list_nodes.index(v),w)
    return G_copy



