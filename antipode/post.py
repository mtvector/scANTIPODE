import pandas as pd
import scanpy as sc
import os
import scipy
import numpy as np
import tqdm
from . import plotting
from .antipode_model import ANTIPODE

def moving_average_values(x,y,window_size=1001):
    moving_average = antipode.plotting.moving_average(y[np.argsort(x)],window_size)
    return(np.sort(x[int(window_size/2):-int(window_size/2)]),moving_average)
    
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def select_marker_genes(marker_df, expr_df, threshold, marker=True, other_quantile=0.95):
    """
    Select marker or antimarker genes for each cluster.

    For marker==True (markers):
      - marker_df should be computed with a high quantile (e.g. 0.95 or 0.99).
      - For each cluster, genes are sorted in descending order (largest difference first).
      - The function returns the first gene whose expression in the cluster exceeds `threshold`.
      - If none pass, the top-ranked gene is returned.

    For marker==False (antimarkers):
      - marker_df should be computed with a low quantile (e.g. 0.1).
      - Genes are sorted in ascending order (lowest difference first).
      - For each candidate gene, the expression in all other clusters is examined.
      - If `other_threshold` is provided, the gene is chosen only if the specified quantile
        (default 0.95) of its expression in the other clusters exceeds `other_threshold`.
      - If no gene meets this filter, the top candidate (i.e. the gene with the minimum score)
        is returned.
    
    :param marker_df: DataFrame from get_quantile_markers (clusters as rows, genes as columns).
    :param expr_df: DataFrame of the original expression values (clusters as rows, genes as columns).
    :param threshold: In 'marker' mode, the minimum expression required in the target cluster.
                      In 'antimarker' mode, not used.
    :param marker: Boolean to use markers or antimarker strategy
    :param other_quantile: In 'antimarker' mode, the quantile of the non-target clusters’ expression to consider.
    :return: Dictionary mapping cluster -> selected gene.
    """
    selected_markers = {}
    clusters = marker_df.index

    for cluster in clusters:
        if marker:
            # For markers, higher score is better.
            sorted_genes = marker_df.loc[cluster].sort_values(ascending=False)
            top_gene = sorted_genes.index[0]
            chosen_gene = None
            for gene in sorted_genes.index:
                if expr_df.loc[cluster, gene] > threshold:
                    chosen_gene = gene
                    break
            selected_markers[cluster] = chosen_gene if chosen_gene is not None else top_gene

        else:
            # For antimarkers, lower score is better.
            sorted_genes = marker_df.loc[cluster].sort_values(ascending=True)
            top_gene = sorted_genes.index[0]
            chosen_gene = None
            # Identify the other clusters
            other_clusters = expr_df.index.difference([cluster])
            for gene in sorted_genes.index:
                other_expr = expr_df.loc[other_clusters, gene]
                # If no threshold is provided, pick the top candidate.
                # Otherwise, require that a high quantile of expression in others exceeds the threshold.
                if (threshold is None) or (np.quantile(other_expr, other_quantile) > threshold):
                    chosen_gene = gene
                    break
            selected_markers[cluster] = chosen_gene if chosen_gene is not None else top_gene

    return selected_markers

def get_quantile_markers(df,q=0.95):
    """
    Get the markers for the rows of a dataframe.

    :param df: GEX means where rows are categories and columns are named features.
    :param q: Quantile of mean to subtract.
    :return: A matrix which has the difference of each gene in each cluster vs the quantile value in all other clusters.
    """
    df_array=df.to_numpy()
    coefs=[]
    for i in tqdm.tqdm(range(df.shape[0])):
        others=list(set(list(range(df.shape[0])))-set([i]))
        coefs.append((df_array[i:(i+1),:]-np.nanquantile(df_array[others,:],q,axis=0)))#/(cluster_params.std(0)+cluster_params.std(0).mean()))
    coefs=np.concatenate(coefs,axis=0)
    marker_df=pd.DataFrame(coefs,index=df.index,columns=df.columns)
    return(marker_df)

def get_conserved_quantile_markers(exprs,index,columns,q=0.95,antimarker=False):
    """
    Get the conserved markers for axis 1 of mean expression tensor.

    :param df: GEX means where rows are categories and columns are named features.
    :param q: Quantile of mean to subtract.
    :return: A matrix which has the difference of each gene in each cluster vs the quantile value in all other clusters.
    """
    direction_operation = np.max if antimarker else np.min
    coefs=[]
    for i in tqdm.tqdm(range(exprs.shape[1])):
        others=list(set(list(range(exprs.shape[1])))-set([i]))
        diff_from_quantile = exprs[:,i,:]-np.nanquantile(exprs[:,others,:],q,axis=1)
        coefs.append(direction_operation(diff_from_quantile,axis=0))#/(cluster_params.std(0)+cluster_params.std(0).mean()))
    coefs_cat=np.stack(coefs,axis=0)
    marker_df=pd.DataFrame(coefs_cat,index=index,columns=columns)
    return(marker_df)

def get_n_largest(n):
    def get_top_n(x):
        return x.nlargest(n).index.tolist()
    return(get_top_n)

def resampling_p_value(data, group_labels,fun, num_iterations=1000):
    """
    Calculate the resampling p-value for the magnitude of input values, partitioned by two groups.
    
    Arguments:
    data -- A list or NumPy array of input values.
    group_labels -- A list or NumPy array of group labels corresponding to each value in the data.
    num_iterations -- The number of iterations to perform for the resampling (default: 1000).
    
    Returns:
    p_value -- The resampling p-value.
    """
    group_labels = np.array(group_labels)
    data = np.array(data)
    group1_data = data[group_labels]
    group2_data = data[~group_labels]
    observed_difference = np.abs(np.mean(fun(group1_data)) - np.mean(fun(group2_data)))

    combined_data = np.concatenate((group1_data, group2_data))
    num_group1 = len(group1_data)
    num_group2 = len(group2_data)
    num_total = num_group1 + num_group2
    larger_difference_count = 0

    for _ in tqdm.tqdm(range(num_iterations)):
        np.random.shuffle(combined_data)
        perm_group1 = combined_data[:num_group1]
        perm_group2 = combined_data[num_group1:]
        perm_difference = np.abs(np.mean(fun(perm_group1)) - np.mean(fun(perm_group2)))
        if perm_difference >= observed_difference:
            larger_difference_count += 1

    p_value = (larger_difference_count + 1) / (num_iterations + 1)
    return p_value

def resampling_slope_p_value(x, y, num_iterations=1000):
    """
    Calculate the resampling p-value for the slope of the linear fit of two variables.
    
    Arguments:
    x -- A 1D NumPy array or list representing the independent variable.
    y -- A 1D NumPy array or list representing the dependent variable.
    num_iterations -- The number of iterations to perform for the resampling (default: 1000).
    
    Returns:
    p_value -- The resampling p-value.
    """
    x = np.array(x)
    y = np.array(y)
    observed_slope = np.polyfit(x, y, 1)[0]
    num_data = len(x)
    larger_slope_count = 0

    for _ in tqdm.tqdm(range(num_iterations)):
        indices = np.random.choice(num_data, num_data, replace=True)
        resampled_x = x
        resampled_y = y[indices]
        resampled_slope = np.polyfit(resampled_x, resampled_y, 1)[0]
        if np.abs(resampled_slope) >= np.abs(observed_slope):
            larger_slope_count += 1

    p_value = (larger_slope_count + 1) / (num_iterations + 1)
    return p_value

def uniqlist(seq):
    #from https://stackoverflow.com/questions/480214/how-do-i-remove-duplicates-from-a-list-while-preserving-order
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


