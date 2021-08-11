from copy import deepcopy
import numpy as np
from sklearn.metrics import pairwise_distances

def cfilter(centers, labels, t=5, position_likes=None, labels_likes=None):
    """
    This function returns the clusters together with node labels. If t>0, all clusters with size len(cluster) < t will be reoved from the dataset an the position_like arrays
    will be adjusted accordingly. The labels likes will not be adjusted, but only cutted!
    """
    if position_likes is None:
        position_likes = []
        
    if labels_likes is None:
        labels_likes = []
    
    # backup the labels and the clusters
    centers, labels = deepcopy(centers), deepcopy(labels)
    unique_labels = np.unique(labels)
    
    # Create lists for each cluster contaning the related nodes
    components = []
    node_labels = [None for _ in range(len(labels))]
    
    for label in np.sort(unique_labels):
        component = np.argwhere(label == labels).T
        assert len(component) == 1

        for node in component[0]:
            node_labels[node] = label

        components.append(component[0].tolist())
    
    # Find the cluster indices of the to small clusters
    removal = []
    for idx, component in enumerate(components):
        if len(component) < t:
            removal.append(idx)
    
    # delete the too small centers
    new_centers = np.delete(centers, removal)
    
    # remove the old labels from the labels array
    position_removal = []
    labels_np = np.array(labels)
    
    for label in removal:
        position_removal += np.argwhere(label==labels_np).T[0].tolist()
    
    new_labels = np.delete(labels, position_removal)
    
    
    # adjust the position arrays
    new_position_likes = []
    for pos_like in position_likes:
        new_position_likes.append(np.delete(pos_like, position_removal, axis=0))
        
    # adjust the labels arrays
    new_labels_likes = []
    for lab_like in labels_likes:
        new_labels_likes.append(np.delete(lab_like, position_removal))
    
    
    # remove te gaps in the labels array (They sholud be the indices of the clusters array...):
    difference = [0 for i in range(len(unique_labels))]
    last_difference = 0
    
    for label in np.sort(unique_labels):
        if label in removal:
            last_difference += 1
            
        difference[label] = last_difference
        
    for idx, label in enumerate(new_labels):
        new_labels[idx] -= difference[label]
        
    
    # adjust the cluster centers to the fact, that the position array changed (they are the indices of it...):    
    pos_difference = [0 for i in range(len(labels))]
    last_difference = 0
      
    for idx in range(len(labels)):
        if idx in position_removal:
            last_difference += 1
            
        
        pos_difference[idx] = last_difference
        
    for idx, center in enumerate(new_centers):
        new_centers[idx] -= pos_difference[center]
        
        
        
    # Adjust the output to the input
    if len(new_position_likes) != 0:
        if len(new_labels_likes) != 0:
            return new_centers, new_labels, new_position_likes, new_labels_likes
        
        else:
            return new_centers, new_labels, new_position_likes
    
    elif len(new_labels_likes) != 0:
        return new_centers, new_labels, new_labels_likes
    
    return new_centers, new_labels


def cassign(centers, labels, position, t):
    
    number_of_nodes = len(labels)
    unique_labels = np.unique(labels)
    
    # Create the lists containing the nodes of each component
    components = []
    node_labels = [None for _ in range(number_of_nodes)]
    for label in np.sort(unique_labels):
        component = np.argwhere(label == labels).T
        assert len(component) == 1

        for node in component[0]:
            node_labels[node] = label

        components.append(component[0].tolist())

    
    distances = pairwise_distances(position, position)

    cleaned_components = [False for _ in range(len(unique_labels))]

    
    while True:

        for idx, component in enumerate(components):

            if cleaned_components[idx] == True:
                continue

            if len(component) <= t:

                for node in component:
                    nearests = np.argsort(distances[node])
                    for elem in nearests:
                        if elem not in component:
                            nearest = elem
                            break
                    node_labels[node] = node_labels[nearest]
                    components[node_labels[nearest]].append(node)
                cleaned_components[idx] = True
                continue

        break
        
    new_components = []
    new_node_labels = [None for _ in range(number_of_nodes)]
    
    
    shift = 0
    for idx, component in enumerate(components):
        if cleaned_components[idx] == False:
            new_components.append(component)
            for node in component:
                new_node_labels[node] = idx - shift
        else:
            shift += 1
           
    new_centers = []
    
    for component in new_components:
        center = np.mean(position[component], axis=0)
        new_centers.append(np.argmin(np.linalg.norm(position-center, axis=1)))
    
    return new_centers, new_node_labels