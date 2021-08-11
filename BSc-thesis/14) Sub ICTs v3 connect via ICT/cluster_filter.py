from copy import deepcopy
import numpy as np

def cfilter(centers, labels, position, pca, embedding, pos_labels=None):
    centers, labels = deepcopy(centers), deepcopy(labels)
    number_of_nodes = len(position)
    unique_labels = np.unique(labels)

    components = []
    node_labels = [None for _ in range(number_of_nodes)]
    for label in np.sort(unique_labels):
        component = np.argwhere(label == labels).T
        assert len(component) == 1

        for node in component[0]:
            node_labels[node] = label

        components.append(component[0].tolist())
    removal = []
    for idx, component in enumerate(components):
        if len(component) < 5:
            removal.append(idx)

    new_centers = np.delete(centers, removal)
    
    position_removal = []
    labels_np = np.array(labels)
    
    for label in removal:
        position_removal += np.argwhere(label==labels_np).T[0].tolist()
    
    
    new_labels = np.delete(labels, position_removal)
    new_position = np.delete(position, position_removal, axis=0)
    new_embedding = np.delete(embedding, position_removal, axis=0)
    new_pca = np.delete(pca, position_removal, axis=0)
    
    
    # remove te gaps in the labels array:
    
    difference = [0 for i in range(len(unique_labels))]
    last_difference = 0
    
    for label in np.sort(unique_labels):
        if label in removal:
            last_difference += 1
            
        difference[label] = last_difference
        
    for idx, label in enumerate(new_labels):
        new_labels[idx] -= difference[label]
        
    
    # adjust the cluster centers to the new position arrays:
    
    pos_difference = [0 for i in range(len(position))]
    last_difference = 0
    
    
    for idx in range(len(position)):
        if idx in position_removal:
            last_difference += 1
            
        
        pos_difference[idx] = last_difference
        
    for idx, center in enumerate(new_centers):
        new_centers[idx] -= pos_difference[center]
        
    
    
    if pos_labels is not None:
        new_pos_labels = np.delete(pos_labels, position_removal)
        
        return new_centers, new_labels, new_position, new_pca, new_embedding, new_pos_labels
    
    return new_centers, new_labels, new_position, new_pca, new_embedding