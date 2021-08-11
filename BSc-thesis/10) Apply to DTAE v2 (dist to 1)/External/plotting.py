import networkit as nk
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from copy import deepcopy
from numpy.linalg import norm
from External.ICT.calculate_ICT import compute_widths

def arr_to_dict(arr):
        dct = {}
        for idx, elem in tqdm(enumerate(arr), desc="Create a dict from the position array"):
            dct[idx] = elem
        return dct

def plot_graph(G, position, name, ax, cluster_centers=None, edge_scale=1, node_size=10, edge_color="black"):
    
    widths = np.array(compute_widths(G))
    
    nk.viztasks.drawGraph(G, pos=position, ax=ax, width=widths*edge_scale,
                          node_size=node_size, edge_color=edge_color)
    if cluster_centers is not None:
        ax.plot(*position[cluster_centers].T, marker="o", color = "Red")
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.set_axis_on()
    ax.set_title(name)
    ax.axis("equal")
    
def plot_points(position, name, ax, labels=None):
    
    if labels is None:
        labels = np.array([0 for _ in range(len(position))])
        
    indices = np.unique(labels, return_index=True)[1]
    unique_labels = [labels[index] for index in sorted(indices)]
        
    for label in unique_labels:
        ax.plot(*position[np.argwhere(labels == label).T[0]].T, label=label)
        
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.set_axis_on()
    ax.legend()
    ax.set_title(name)
    ax.axis("equal")
    
def no_intersections(tree, position, ax, labels=None):
    
    ICT = deepcopy(tree)

    for u, v in ICT.iterEdges():
        ICT.setWeight(u, v, norm(position[u] - position[v]))

    ICT_nx = nk.nxadapter.nk2nx(ICT)

    mapping = nx.planar_layout(ICT_nx)

    mapping_ndarray = np.zeros((ICT.upperEdgeIdBound()+1,2))

    for key, val in zip(mapping.keys(), mapping.values()):
        mapping_ndarray[key] = np.array([-val[1], -val[0]])

    mapping = mapping_ndarray
    
    widths = np.array(compute_widths(ICT))
    
    nk.viztasks.drawGraph(ICT, pos=mapping, ax=ax, width=widths**1.3, node_size=0)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.set_axis_on()
    ax.set_title("No intersection layout")
    
    
    
    if labels is None:
        labels = np.array([0 for _ in range(len(position))])
        
    indices = np.unique(labels, return_index=True)[1]
    unique_labels = [labels[index] for index in sorted(indices)]
        
    for label in unique_labels:
        ax.plot(*mapping[np.argwhere(labels == label).T[0]].T, label=label)
        
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.set_axis_on()
    ax.legend()

