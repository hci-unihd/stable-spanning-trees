import networkit as nk
import numpy as np

from sklearn.metrics import pairwise_distances
from copy import deepcopy


def MST(ICT, position, centers):
    new_ICT = deepcopy(ICT)

    dist = pairwise_distances(position)

    for u, v, w in ICT.iterEdgesWeights():
#         new_ICT.setWeight(u, v, dist[u, v])
        new_ICT.setWeight(u, v, 1)

    mst = find_tree(centers, new_ICT)
    
    for u, v, w in mst.iterEdgesWeights():
        mst.setWeight(u, v, dist[u, v])

    mst.indexEdges()
    
    return mst
    
    
def find_tree(centers, ICT):
    
    graph = nk.graph.Graph(n=len(centers), weighted=True)
    
    distances = [[] for i in range(len(centers))]
    
    for i, source in enumerate(centers):
        Dijkstra = nk.distance.Dijkstra(G=ICT, source=source)
        Dijkstra.run()
        
        for target in centers:
            
            if target == source:
                
                # Later we are looking for the shortest distance
                # and we dont wont the trivial 0 distance!
                
                distances[i].append(np.inf)
                continue
                
            distances[i].append(Dijkstra.distance(target))
            
    distances = np.array(distances)

    for u in range(len(centers)):
        for v in range(u-1):
            if u == v:
                continue
            graph.addEdge(u, v, 1/distances[u, v])
            
    MST = nk.graph.RandomMaximumSpanningForest(graph)
    MST.run()
    mst = MST.getMSF(True)
    
    return mst