import torch
import torch.nn as nn
import numpy as np
import tqdm
import pyro
import scipy
from torch.nn.functional import softplus, softmax
import scanpy as sc
import math
import scvi

def null_function(x):
    return x

def safe_sigmoid(x,eps=1e-10):
    #return torch.clamp(torch.sigmoid(x),min=eps,max=(1.-eps))
    return (torch.sigmoid(x)+1e-6)*(1-1e-5)

def centered_sigmoid(x):
    return (2*(torch.sigmoid(x)-0.5))

def numpy_centered_sigmoid(x):
    return((scipy.special.expit(x)-0.5)*2)

def numpy_relu(x):
    return x*(x>0)

def safe_softmax(x,dim=-1,eps=1e-10):
    x=torch.softmax(x,dim)
    x=x+eps
    return (x/x.sum(dim,keepdim=True))

def minmax(x):
    return(x.min(),x.max())

def param_store_to_numpy():
    store={}
    for name in pyro.get_param_store():
        store[name]=pyro.param(name).cpu().detach().numpy()
    return store

def get_field(adata,loc):
    return adata.__getattribute__(loc[0]).__getattribute__(loc[1])

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def numpy_onehot(x,num_classes=None):
    n_values = np.max(x) + 1
    if num_classes is None or num_classes<n_values:
        num_classes=n_values
    return np.eye(num_classes)[x]

def numpy_hardmax(x,axis=-1):
    return(numpy_onehot(x.argmax(axis).flatten(),num_classes=x.shape[axis]))

def calculate_layered_tree_means(X, level_assignments):
    """
    Calculates and adjusts means for clusters at each level according to the hierarchical structure,
    dynamically handling any number of layers, make sure layer size L+1 > L.
    :param level_assignments: A list of arrays, where each array gives the cluster assignment at each level.
    :return: A dictionary containing the adjusted means for each level.
    """
    means = {}
    adjusted_means = {}

    # Calculate initial means for each level
    for level, assignments in enumerate(level_assignments, start=1):
        unique_clusters = np.unique(assignments)
        means[level] = {cluster: X[assignments.flatten() == cluster].mean(axis=0) for cluster in unique_clusters}

    # Adjust means for each level
    for level in range(1, len(level_assignments) + 1):
        adjusted_means[level] = {}
        for cluster_id, cluster_mean in means[level].items():
            adjusted_mean = np.copy(cluster_mean)
            # Subtract the means of all ancestor levels
            for ancestor_level in range(1, level):
                # Find the closest ancestor cluster for each cluster
                ancestor_cluster = level_assignments[ancestor_level - 1][np.argwhere(level_assignments[level-1] == cluster_id)[0][0]]
                adjusted_mean -= means[ancestor_level][ancestor_cluster[0]]
            adjusted_means[level][cluster_id] = adjusted_mean

    return adjusted_means

def create_edge_matrices(level_assignments):
    """
    Creates adjacency matrices for each layer based on level assignments (like output from scipy.cluster.hierarchy.cut_tree) .
    
    :param level_assignments: A list of np arrays, where each array gives the cluster assignment at each level.
    :return: A list of adjacency matrices for each layer transition.
    """
    adjacency_matrices = []

    for i in range(len(level_assignments) - 1):
        current_level = level_assignments[i]
        next_level = level_assignments[i+1]
        
        # Determine the unique number of clusters at each level for the dimensions of the one-hot encodings
        num_clusters_current = len( np.unique(current_level))
        num_clusters_next = len(np.unique(next_level))
        
        # Create one-hot encodings for each level
        one_hot_current = numpy_onehot(current_level.flatten(), num_classes=num_clusters_current)
        one_hot_next = numpy_onehot(next_level.flatten(), num_classes=num_clusters_next)
        adjacency_matrix = one_hot_current.T @ one_hot_next
        
        # Normalize the adjacency matrix to have binary entries (1 for connection, 0 for no connection)
        adjacency_matrix = (adjacency_matrix > 0).astype(np.float64)

        adjacency_matrices.append(adjacency_matrix)
    
    return adjacency_matrices


def group_aggr_anndata(ad, category_column_names, agg_func=np.mean, layer=None, obsm=False):
    """
    Calculate the aggregated value (default is mean) for each column for each group combination in an AnnData object,
    returning a numpy array of the shape [cat_size0, cat_size1, ..., num_variables] and a dictionary of category orders.
    
    :param ad: AnnData object
    :param category_column_names: List of column names in ad.obs pointing to categorical variables
    :param agg_func: Aggregation function to apply (e.g., np.mean, np.std). Default is np.mean.
    :param layer: Specify if a particular layer of the AnnData object is to be used.
    :param obsm: Boolean indicating whether to use data from .obsm attribute.
    :return: Numpy array of calculated aggregates and a dictionary with category orders.
    """
    if not category_column_names:
        raise ValueError("category_column_names must not be empty")
    
    # Ensure category_column_names are in a list if only one was provided
    if isinstance(category_column_names, str):
        category_column_names = [category_column_names]

    # Initialize dictionary for category orders
    category_orders = {}

    # Determine the size for each categorical variable and prepare indices
    for cat_name in category_column_names:
        categories = ad.obs[cat_name].astype('category')
        category_orders[cat_name] = categories.cat.categories.tolist()

    # Calculate the product of category sizes to determine the shape of the result array
    category_sizes = [len(category_orders[cat]) for cat in category_column_names]
    num_variables = ad.shape[1] if not obsm else ad.obsm[layer].shape[-1]
    result_shape = category_sizes + [num_variables]
    result = np.zeros(result_shape, dtype=np.float64)

    # Iterate over all combinations of category values
    for indices, combination in enumerate(tqdm.tqdm(np.ndindex(*category_sizes), total=np.prod(category_sizes))):
        # Convert indices to category values
        category_values = [category_orders[cat][index] for cat, index in zip(category_column_names, combination)]
        
        # Create a mask for rows matching the current combination of category values
        mask = np.ones(len(ad), dtype=bool)
        for cat_name, cat_value in zip(category_column_names, category_values):
            mask &= ad.obs[cat_name].values == cat_value
        
        selected_indices = np.where(mask)[0]
        
        if selected_indices.size > 0:
            if obsm:
                data = ad.obsm[layer][selected_indices]
            else:
                data = ad[selected_indices].X if layer is None else ad[selected_indices].layers[layer]

            # Convert sparse matrix to dense if necessary
            if isinstance(data, np.ndarray):
                dense_data = data
            else:
                dense_data = data.toarray()
            
            # Apply the aggregation function and assign to the result
            result[combination] = agg_func(dense_data, axis=0)

    
    return result, category_orders