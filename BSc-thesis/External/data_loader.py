import pandas as pd
import numpy as np
import phate
import umap
from sklearn.decomposition import PCA

def data_loader(dataset, return_embedding=False):
    """
    loads the given dataset and returns the data as (X, labels)
    
    dataset = (pancreas_full, pancreas_reduced, eth_CA, eth_PC, eth_CC, phate100,
    pancreas_full_DTAE_embedding, pancreas_reduced_DTAE_embedding, CC_DTAE_embedding)
    
    eth symbols meaning:
    CA 19029 chronic + acute
    C  13727 chronic subset of CA (not implemented)
    PC 13763 pure chronic dataset with phenotypic annotations (uses different naming convention!)
    CC 13707 common chronic cells between PC and CA
    """
    
    if dataset == "pancreas_full":
    
        labels = pd.read_csv("~/Data/pancreas/obs.csv",header=0)
        position = pd.read_csv("~/Data/pancreas/X.csv",header=None)

        labels = labels["clusters_fig6_broad_final"].to_numpy()
        position = position.to_numpy()
        
        if not return_embedding:
            return position, labels
        
        reducer = umap.UMAP(min_dist=0.8,random_state=42)
        embedding = reducer.fit_transform(position)
        
        return position, labels, embedding
    
    elif dataset == "pancreas_reduced":
        
        labels = pd.read_csv("~/Data/pancreas/obs.csv",header=0)
        position = pd.read_csv("~/Data/pancreas/X.csv",header=None)

        mask = labels.index[(labels['day'] <= 15.5) & 
                            ((labels["clusters_fig6_broad_final"] == "Ngn3 low EP") |
                             (labels["clusters_fig6_broad_final"] == "Ngn3 high EP") |
                             (labels["clusters_fig6_broad_final"] == "Fev+") |
                             (labels["clusters_fig6_broad_final"] == 'Delta') | 
                             (labels["clusters_fig6_broad_final"] ==  'Alpha') | 
                             (labels["clusters_fig6_broad_final"] ==  'Epsilon') | 
                             (labels["clusters_fig6_broad_final"] ==  'Beta') )]

        labels = labels.loc[mask]["clusters_fig6_broad_final"].to_numpy()
        position = position.loc[mask].to_numpy()

        if not return_embedding:
            return position, labels
        
        try:
            embedding = np.load("/net/hcihome/storage/flernst/Data/pancreas/umap_embedding.npy")
            
        except:
            reducer = umap.UMAP(min_dist=0.8,random_state=42)
            embedding = reducer.fit_transform(position)
            np.save("/net/hcihome/storage/flernst/Data/pancreas/umap_embedding.npy", embedding)
        
        return position, labels, embedding
    
    elif dataset == "eth_CA":
        
        labels = pd.read_csv("~/Data/eth/obs.csv",header=0)
        position = pd.read_csv("~/Data/eth/X.csv",header=None)
        
        labels = labels["phase"]
        
        if not return_embedding:
            return position.to_numpy(), labels.to_numpy
        
        labels = labels.to_numpy()
        position = position.to_numpy()
        
        pca = PCA(n_components=50).fit_transform(position)
        reducer = umap.UMAP(min_dist=0.8, random_state=42)
        embedding = reducer.fit_transform(pca)
        
        return position, labels, embedding
    
    elif dataset == "eth_PC":
        
        labels = pd.read_csv("~/Data/eth_reduced/obs.csv",header=0)
        labels = labels["annot"]
        position = pd.read_csv("~/Data/eth_reduced/X_chronic.csv",header=None)
        
        if not return_embedding:
            return position.to_numpy(), labels.to_numpy
        
        
        labels = labels.to_numpy()
        position = position.to_numpy()
        
        pca = PCA(n_components=50).fit_transform(position)
        reducer = umap.UMAP(min_dist=0.8, random_state=42)
        embedding = reducer.fit_transform(pca)
        
        return position, labels, embedding
    
    elif dataset == "eth_CC":
        
        PC_to_CC = np.load("/net/hcihome/storage/flernst/Data/eth_CC/PC_to_CC.npy")
        
        labels = pd.read_csv("~/Data/eth_reduced/obs.csv",header=0)[PC_to_CC]
        labels = labels["annot"]
        position = pd.read_csv("~/Data/eth_reduced/X_chronic.csv",header=None)[PC_to_CC]
        
        if not return_embedding:
            return position.to_numpy(), labels.to_numpy()
        
        labels = labels.to_numpy()
        position = position.to_numpy()
        
        try:
            embedding = np.load("/net/hcihome/storage/flernst/Data/eth_CC/umap_embedding.npy")
        
        except:
            pca = PCA(n_components=50).fit_transform(position)
            reducer = umap.UMAP(min_dist=0.8, random_state=42)
            embedding = reducer.fit_transform(pca)
            np.save("/net/hcihome/storage/flernst/Data/eth_CC/umap_embedding.npy", embedding)
        
        return position, labels, embedding
    
    elif dataset == "phate100":
        
        if not return_embedding:
            return phate.tree.gen_dla(n_dim=2, n_branch=3, branch_length=2500,
                                      rand_multiplier=2, seed=37, sigma=1)
        
        position, labels = phate.tree.gen_dla(n_dim=2, n_branch=3, branch_length=2500,
                                      rand_multiplier=2, seed=37, sigma=1)
        
        return position, labels, position
    
    elif dataset == "pancreas_full_DTAE_embedding":
        
        if return_embedding:
            raise KeyError("The dataset is allready an embedding. Please choose return_embedding=False")
        
        DTAE_embedding = pd.read_csv("~/Data/pancreas/embedding.csv",header=None)
        
        labels = pd.read_csv("~/Data/pancreas/obs.csv",header=0)
        
        labels = labels["clusters_fig6_broad_final"]
        
        return DTAE_embedding.to_numpy(), labels.to_numpy()    
    
    elif dataset == "pancreas_reduced_DTAE_embedding":
        
        if return_embedding:
            raise KeyError("The dataset is allready an embedding. Please choose return_embedding=False")
        
        DTAE_embedding = pd.read_csv("~/Data/pancreas/embedding.csv",header=None)
        
        labels = pd.read_csv("~/Data/pancreas/obs.csv",header=0)
        
        mask = labels.index[(labels['day'] <= 15.5) & 
                            ((labels["clusters_fig6_broad_final"] == "Ngn3 low EP") |
                             (labels["clusters_fig6_broad_final"] == "Ngn3 high EP") |
                             (labels["clusters_fig6_broad_final"] == "Fev+") |
                             (labels["clusters_fig6_broad_final"] == 'Delta') | 
                             (labels["clusters_fig6_broad_final"] ==  'Alpha') | 
                             (labels["clusters_fig6_broad_final"] ==  'Epsilon') | 
                             (labels["clusters_fig6_broad_final"] ==  'Beta') )]
        
        labels = labels.loc[mask]["clusters_fig6_broad_final"]

        DTAE_embedding = DTAE_embedding.loc[mask]
        
        return DTAE_embedding.to_numpy(), labels.to_numpy()
    
    elif dataset == "CC_DTAE_embedding":
        
        if return_embedding:
            raise KeyError("The dataset is allready an embedding. Please choose return_embedding=False")
        
        CA_to_CC = np.load("/net/hcihome/storage/flernst/Data/eth_CC/CA_to_CC.npy")
        PC_to_CC = np.load("/net/hcihome/storage/flernst/Data/eth_CC/PC_to_CC.npy")
        
        DTAE_embedding = pd.read_csv("~/Data/eth/embedding.csv",header=None).to_numpy()[CA_to_CC]
        labels = pd.read_csv("~/Data/eth_reduced/obs.csv",header=0)[PC_to_CC]
        labels = labels["annot"]
        
        return DTAE_embedding, labels.to_numpy()
    
    else:
        raise KeyError("Not implemented")