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

def hardmax(x,axis=-1):
    return(torch.nn.functional.one_hot(x.max(axis)[1],x.shape[axis]).float())

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

def prop_zeros(x,axis=-1):
    return(np.mean(x>0.,axis=axis))

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
    :param level_assignments: A list of arrays, where each array gives the cluster's assignment at each level. Root should be array of zeros, leaves should be range(X.shape[0]).
    :return: A dictionary containing the adjusted means for each level.
    """
    means = {}
    adjusted_means = {}
    cumulative_adjustments = np.zeros_like(X[0])
    
    for level, assignments in enumerate(level_assignments, start=1):
        unique_clusters = np.unique(assignments)
        level_mean_adjustments = np.zeros_like(X[0])
        
        # Calculate the initial means for each cluster at the current level
        means[level] = {}
        for cluster in sorted(unique_clusters):
            cluster_mean = X[assignments.flatten() == cluster].mean(axis=0)
            means[level][cluster] = cluster_mean
            
            # Apply adjustments from previous levels
            adjusted_cluster_mean = cluster_mean - cumulative_adjustments
            adjusted_means.setdefault(level, {})[cluster] = adjusted_cluster_mean
            
            # Accumulate adjustments for the current level
            level_mean_adjustments += adjusted_cluster_mean
        
        # Update cumulative adjustments for the next level
        num_clusters = len(unique_clusters)
        if num_clusters > 0:  # Avoid division by zero
            cumulative_adjustments += level_mean_adjustments / num_clusters

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


def group_aggr_anndata(ad, category_column_names, agg_func=np.mean, layer=None, obsm=False, normalize=False, batch_size=1000):
    """
    Calculate the aggregated value (default is mean) for each column for each group combination in an AnnData object,
    returning a numpy array of the shape [cat_size0, cat_size1, ..., num_variables] and a dictionary of category orders.
    
    :param ad: AnnData object
    :param category_column_names: List of column names in ad.obs pointing to categorical variables
    :param agg_func: Aggregation function to apply (e.g., np.mean, np.std). Default is np.mean.
    :param layer: Specify if a particular layer of the AnnData object is to be used.
    :param obsm: Boolean indicating whether to use data from .obsm attribute.
    :param normalize: Boolean indicating whether to normalize the data.
    :param batch_size: Size of the batches for processing the data.
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
            agg_results = []
            for start in range(0, selected_indices.size, batch_size):
                end = min(start + batch_size, selected_indices.size)
                batch_indices = selected_indices[start:end]
                
                if obsm:
                    data = ad.obsm[layer][batch_indices]
                else:
                    data = ad[batch_indices].X if layer is None else ad[batch_indices].layers[layer]
                
                # Convert sparse matrix to dense if necessary
                if isinstance(data, np.ndarray):
                    dense_data = data
                else:
                    dense_data = data.toarray()
                
                if normalize:
                    dense_data = dense_data / (1. + dense_data.sum(-1, keepdims=True))
                
                agg_results.append(agg_func(dense_data, axis=0)*((end-start)/selected_indices.size))
            
            # Combine results from all batches
            result[combination] = np.sum(np.array(agg_results), axis=0)
    
    return result, category_orders

def get_real_leaf_means(adata,discov_key,leaf_key,layer=None, count_threshold=1000):
    '''convenience function wraps aggregation and safe_log_transform. Returns [discov_levels,leaf_clusters,genes] array of log means'''
    aggr_means=group_aggr_anndata(adata,[discov_key,leaf_key],layer=layer,normalize=True)
    aggr_sums=group_aggr_anndata(adata,[discov_key,leaf_key],layer=layer,normalize=False,agg_func=np.sum)
    log_real_means=safe_log_transform(aggr_means[0],aggr_sums[0].sum(-1)[...,np.newaxis], count_threshold=1000)
    return log_real_means, aggr_means[1]

def safe_log_transform(x, sums=None, count_threshold=1000):
    x = np.asarray(x)
    if sums is not None:
        sums = np.asarray(sums) 
        offset = 0.5 / (sums + 1.) # Geometric distribution memorylessness result
        result = np.log(x + offset)
        if count_threshold is not None:
            mask = sums > count_threshold
            result = np.where(mask, result, np.nan)
    else:
        nonzero = x[x > 0]
        offset = nonzero.min() * 0.9 if nonzero.size > 0 else 1e-10
        result = np.log(x + offset)
    return result

# def safe_log_transform(x, sums=None):
#     if sums is not None:
#         offset = 0.5 / (sums+1.) #naive expected value halfway between smallest observable value and 0
#     else:
#         nonzero = x[x > 0]
#         offset = nonzero.min() * 0.9 if nonzero.size > 0 else 1e-10 #Sets to just below lowest observed value in dataset (extreme)
#     return np.log(x + offset)

def pandas_numericategorical(col):
    col = col.astype('category')
    return col.cat.reorder_categories(col.cat.categories[np.argsort(col.cat.categories.astype(float))])

