import seaborn
import numpy as np
import pandas as pd
import scipy
import torch
import matplotlib
import matplotlib.pyplot as plt
import tqdm
import scanpy as sc
import sklearn
from . import model_functions
try:
    import gseapy
except:
    print("GSEApy not found. Can't get module enrichments")

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def plot_loss(loss_tracker):
    '''Plots vector of values along with moving average'''
    seaborn.scatterplot(x=list(range(len(loss_tracker))),y=loss_tracker,alpha=0.5,s=2)
    w=300
    mvavg=model_functions.moving_average(np.pad(loss_tracker,int(w/2),mode='edge'),w)
    seaborn.lineplot(x=list(range(len(mvavg))),y=mvavg,color='coral')
    plt.show()

def plot_grad_norms(antipode_model):
    plt.figure(figsize=(20, 5), dpi=100).set_facecolor("white")
    ax = plt.subplot(111)
    w=300
    for i,(name, grad_norms) in enumerate(antipode_model.gradient_norms.items()):
        mvavg=model_functions.moving_average(np.pad(grad_norms,int(w/2),mode='edge'),w)
        seaborn.lineplot(x=list(range(len(mvavg))),y=mvavg,label=name,color=sc.pl.palettes.godsnot_102[i%102],ax=ax,linewidth = 1.)
    plt.xlabel("iters")
    plt.ylabel("gradient norm")
    plt.yscale("log")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title("Gradient norms during SVI");
    plt.show()

def clip_latent_dimensions(matrix, x):
    """
    Clips each latent dimension of the matrix at the 0+x and 100-x percentile.

    Parameters:
    - matrix: A 2D NumPy array of shape [number of observations, latent dimensions].
    - x: The percentage for the lower and upper bounds (0 < x < 50).

    Returns:
    - A 2D NumPy array with the same shape as the input matrix, with values clipped.
    """
    # Ensure x is within the valid range
    if x < 0 or x > 50:
        raise ValueError("x must be between 0 and 50")

    # Initialize a clipped matrix with the same shape as the input matrix
    clipped_matrix = np.zeros_like(matrix)

    # Iterate over each column (latent dimension) to apply clipping
    for col_idx in range(matrix.shape[1]):
        # Calculate the percentiles for the current column
        lower_percentile = np.percentile(matrix[:, col_idx], x)
        upper_percentile = np.percentile(matrix[:, col_idx], 100-x)
        
        # Clip the values in the current column based on the calculated percentiles
        clipped_matrix[:, col_idx] = np.clip(matrix[:, col_idx], lower_percentile, upper_percentile)

    return clipped_matrix

def plot_batch_embedding_pca(antipode_model):
    try:
        pca=sklearn.decomposition.PCA(n_components=2)
        batch_eye=torch.eye(antipode_model.num_batch)
        batch_pca=pca.fit_transform(model_functions.centered_sigmoid(antipode_model.be_nn.cpu()(batch_eye)).detach().numpy())
        df=pd.DataFrame(batch_pca)
        batch_species=antipode_model.adata_manager.adata.obs.groupby(antipode_model.batch_key)['species'].value_counts().unstack().idxmax(axis=1).to_dict()
        
        df[antipode_model.batch_key]=antipode_model.adata_manager.adata.obs[antipode_model.batch_key].cat.categories
        df[antipode_model.discov_key]=[batch_species[x] for x in antipode_model.adata_manager.adata.obs[antipode_model.batch_key].cat.categories]
        seaborn.scatterplot(df,x=0,y=1,hue='species')
        plt.xlabel('batch embedding PC1')
        plt.ylabel('batch embedding PC2')
        return(df)
    except:
        print('plot failed')

def plot_d_hists(antipode_model):
    """plot deltas from antipode model parameters"""
    categories=antipode_model.adata_manager.registry['field_registries']['discov_ind']['state_registry']['categorical_mapping']
    colors=antipode_model.adata_manager.adata.uns[antipode_model.adata_manager.registry['field_registries']['discov_ind']['state_registry']['original_key']+'_colors']
    param_store=antipode_model.adata_manager.adata.uns['param_store']
    
    seaborn.histplot(param_store['locs'].flatten(),color="slategray",label='shared',bins=50,stat='proportion')
    for i in range(len(categories)):
        seaborn.histplot(param_store['discov_dm'][i,...].flatten(),color=colors[i],bins=50,label=categories[i],stat='proportion')
    seaborn.histplot(param_store['batch_dm'].flatten(),color='lightgrey',bins=50,label='batch',stat='proportion')
    plt.legend()
    plt.title('DM')
    plt.show()
    
    seaborn.histplot(param_store['cluster_intercept'].flatten(),color='slategray',bins=50,label='shared',stat='proportion')
    for i in range(len(categories)):
        seaborn.histplot(param_store['discov_di'][i,...].flatten(),color=colors[i],bins=50,label=categories[i],stat='proportion')
    seaborn.histplot(param_store['batch_di'].flatten(),color='lightgrey',bins=50,label='batch',stat='proportion')
    plt.legend()
    plt.title('DI')
    plt.show()
    
    seaborn.histplot(param_store['z_decoder_weight'].flatten(),color='slategray',bins=50,label='shared',stat='proportion')
    for i in range(len(categories)):
        seaborn.histplot(param_store['discov_dc'][i,...].flatten(),color=colors[i],bins=50,label=categories[i],stat='proportion')
    plt.legend()
    plt.title('DC')
    plt.show()

def plot_tree_edge_weights(antipode_model):
    for name in antipode_model.adata_manager.adata.uns['param_store'].keys():
        if 'edge' in name:
            seaborn.heatmap(scipy.special.softmax(antipode_model.adata_manager.adata.uns['param_store'][name],axis=-1))
            plt.show()

def plot_gmm_heatmaps(antipode_model):
    categories=antipode_model.adata_manager.registry['field_registries']['discov_ind']['state_registry']['categorical_mapping']
    colors=antipode_model.adata_manager.adata.uns[antipode_model.adata_manager.registry['field_registries']['discov_ind']['state_registry']['original_key']+'_colors']
    param_store=antipode_model.adata_manager.adata.uns['param_store']

    seaborn.clustermap(antipode_model.z_transform(torch.tensor(param_store['locs'])).numpy(),cmap='coolwarm')
    plt.title('locs')
    plt.show()

    seaborn.clustermap(param_store['locs_dynam'],cmap='coolwarm')
    plt.title('locs_dynam')
    plt.show()

    seaborn.clustermap(param_store['scales'],cmap='coolwarm')
    plt.title('scales')
    plt.show()

    seaborn.histplot(param_store['s_inverse_dispersion'].flatten(),color='grey',bins=50)
    plt.title('inverse_dispersion')
    plt.show()
    

def match_categorical_order(source, target):
    """
    Generate indices to sort the 'source' array to match the order of the 'target' array.
    
    Parameters:
    - source: An iterable of categorical values.
    - target: An iterable of categorical values with a desired ordering.
    
    Returns:
    - An array of indices that will sort 'source' to match the order of 'target'.
    """
    # Create a mapping from target values to their indices
    order_mapping = {val: i for i, val in enumerate(target)}
    
    # Generate a list of indices in 'source' sorted by the order defined in 'target'
    sorted_indices = sorted(range(len(source)), key=lambda x: order_mapping.get(source[x], -1))
    
    # If there are values in 'source' not found in 'target', they are placed at the end by default.
    # You can customize the behavior as needed.
    
    return np.array(sorted_indices)


def ndarray_top_n_indices(arr, n, axis,descending=True):
    """
    Replace the specified axis of a numpy array with the indices of the top n values along that axis.

    :param arr: Multidimensional numpy array.
    :param n: Number of top values to consider.
    :param axis: Axis along which to find the top values.
    :return: Modified array with the specified axis replaced by indices of the top n values.
    """
    if n > arr.shape[axis]:
        raise ValueError(f"n is larger than the size of axis {axis}")

    if descending:
        mul=-1
    else:
        mul=1
    # Get the indices of the top n values along the specified axis
    top_indices = np.argsort(mul*arr, axis=axis)

    # Prepare an indices array that matches the dimensions of the input array
    shape = [1] * arr.ndim
    shape[axis] = n
    indices_shape = np.arange(n).reshape(shape)

    # Use take_along_axis to select the top n indices
    top_n_indices = np.take_along_axis(top_indices, indices_shape, axis=axis)

    # Create the result array
    result_shape = list(arr.shape)
    result_shape[axis] = n
    result = np.empty(result_shape, dtype=int)

    # Use put_along_axis to place the indices in the result array
    np.put_along_axis(result, indices_shape, top_n_indices, axis=axis)
    
    return result

def double_triu_mat(cor_matrix_upper, cor_matrix_lower, diagonal_vector=None):
    """Function to plot one triangular matrix in the upper, another on the lower and a normalized diagonal"""
    # Ensure the input matrices and vector are of compatible sizes
    if cor_matrix_upper.shape != cor_matrix_lower.shape:
        raise ValueError("Matrices and vector sizes are not compatible.")
    
    size = cor_matrix_upper.shape[0]
    dtriu_matrix = np.zeros((size, size))
    
    # Fill the upper triangle of the matrix 
    dtriu_matrix[np.triu_indices(size, k=1)] = cor_matrix_upper[np.triu_indices(size, k=1)]
    
    # Fill the lower triangle of the matrix
    dtriu_matrix[np.tril_indices(size, k=-1)] = cor_matrix_lower[np.tril_indices(size, k=-1)]
    
    # Scale the diagonal vector to match correlation scale
    if diagonal_vector is not None:
        scaled_diagonal = np.interp(diagonal_vector, (diagonal_vector.min(), diagonal_vector.max()), (-1, 1))
    else:
        scaled_diagonal=np.nan
    np.fill_diagonal(dtriu_matrix, scaled_diagonal)
    return(dtriu_matrix)

def plot_genes_by_category(ad, category_column_names, gene_indices):
    """
    Plot the expression of specified genes for each discov and prop_level_3 category.

    :param ad: AnnData object
    :param category_column_names: List of category column names
    :param gene_indices: Indices of the genes to be plotted
    """

    # Calculate mean values for each category
    agg_values, cat_indices = group_aggr_anndata(ad, category_column_names, agg_func=np.mean)

    # Different line styles for each discov
    line_styles = ['-', '--', '-.', ':']
    colors = plt.cm.rainbow(np.linspace(0, 1, len(gene_indices)))

    plt.figure(figsize=(12, 6))

    for discov_idx in range(len(cat_indices[category_column_names[0]])):
        for gene_idx, color in zip(gene_indices, colors):
            gene_name = ad.var_names[gene_idx]
            prop_level_3_categories = list(cat_indices[category_column_names[1]].keys())
            expression_values = agg_values[discov_idx, :, gene_idx]

            # Line style cycles through discov, color cycles through genes
            seaborn.lineplot(x=prop_level_3_categories, y=expression_values, 
                         label=f'discov {discov_idx} - Gene {gene_name}', 
                         linestyle=line_styles[discov_idx % len(line_styles)], color=color)

    plt.title("Expression of Specified Genes")
    plt.xlabel("prop_level_3 Categories")
    plt.ylabel("Expression")
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.show()

import itertools

def plot_top_genes_by_category(ad, category_column_names, top_n, reference_matrix, agg_func=np.mean):
    """
    Plot the expression of top genes for each category based on a reference matrix.

    :param ad: AnnData object
    :param category_column_names: List of category column names
    :param top_n: Number of top genes to consider
    :param reference_matrix: Matrix used to determine the top genes
    :param agg_func: Aggregation function to apply
    """

    # Calculate aggregated values
    agg_values, cat_indices = group_aggr_anndata(ad, category_column_names, agg_func)

    # Different line styles for each discov
    line_styles = ['-', '--', '-.', ':']
    colors = plt.cm.rainbow(np.linspace(0, 1, top_n))

    # Assuming each row of reference_matrix is a different dimension
    for dim_idx in range(reference_matrix.shape[0]):
        plt.figure(figsize=(12, 6))

        # Find the top genes for this dimension
        top_genes_indices = np.argsort(-reference_matrix[dim_idx])[:top_n]

        for discov_idx in range(len(cat_indices[category_column_names[0]])):
            for gene_rank, (gene_idx, color) in enumerate(zip(top_genes_indices, colors)):
                gene_name = ad.var_names[gene_idx]
                prop_level_3_categories = list(cat_indices[category_column_names[1]].keys())
                expression_values = agg_values[discov_idx, :, gene_idx]

                # Line style cycles through discov, color cycles through top genes
                seaborn.lineplot(x=prop_level_3_categories, y=expression_values, 
                             label=f'{discov_idx} - {gene_name}', 
                             linestyle=line_styles[discov_idx % len(line_styles)], color=color)

        plt.title(f"Top {top_n} Genes Expression for Dimension {dim_idx}")
        plt.xlabel("prop_level_3 Categories")
        plt.ylabel("Expression")
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        plt.show()

def plot_gene_mean_ecdf(adata,discov_key):
    adata.obs[discov_key]=adata.obs[discov_key].astype('category')
    outs=antipode.model_functions.group_aggr_anndata(adata,[discov_key])
    seaborn.ecdfplot(pd.DataFrame(outs[0],index=outs[1][discov_key]).T)

def get_prerank_custom_list(input_series,gene_list_dict,**kwargs):
    """
    Run GSEApy prerank on a custom gene list.

    Parameters:
    input_series (pd.Series): A pandas Series where the index is gene names and the values are numerical.
    gene_list (list): A list of HGNC gene symbols.

    Returns:
    Pandas dataframe of res2d
    """

    df = input_series.reset_index()
    df.columns = ['gene_name', 'rank_metric']
    pre_res = gseapy.prerank(rnk=df,
                         gene_sets=gene_list_dict, 
                         processes=1,
                         outdir=None,
                         seed=13,
                         **kwargs)
    return(pre_res.res2d)


def get_prerank_from_mat(mat,gene_list_dict,**kwargs):
    """
    Run GSEApy prerank on a custom gene list.

    Parameters:
    mat (pd.DataFrame): A pandas Dataframe where the index is gene names and the values are numerical.
    gene_list (list): A list of HGNC gene symbols.

    Returns:
    Pandas dataframe of res2ds concatenated
    """
    import warnings
    results={}
    for x in tqdm.tqdm(mat.columns):
        warnings.filterwarnings(action='ignore')
        warnings.catch_warnings(action="ignore")
        results[x]=get_prerank_custom_list(mat[x],gene_list_dict,**kwargs)
        results[x]['input_column']=x
    enrichdf=pd.concat(results.values())
    return(enrichdf)

def select_features_by_pca(A, N,n_components=20):
    '''    
    Get the top N loaded features across n_components using PCA. You should standardize your columns yourself (features).

    Parameters:
    A (np.matrix): A numpy matrix.
    N (integer): Number of features to return.
    n_components (integer): Number of components to compute for PCA

    Returns:
    Array of N indices
    '''
    # Standardize the data (features as columns)

    pca = sklearn.decomposition.PCA(n_components=n_components)
    S_ = pca.fit_transform(A)  # Reconstruct signals
    A_ = np.abs(pca.components_.T)  # Get the mixing matrix

    # Get the absolute weights and rank features within each component
    component_ranks = np.argsort(-np.abs(A_), axis=0)

    # Select N unique features by cycling through components
    selected_features = set()
    num_components = A_.shape[1]
    idx = 0
    while len(selected_features) < N:
        component = idx % num_components  # Cycle through components
        feature_candidates = component_ranks[:, component]
        for feature in feature_candidates:
            if feature not in selected_features:
                selected_features.add(feature)
                break
        idx += 1
        if idx > 1000:  # Safety break to avoid infinite loop
            break

    return list(selected_features)


def select_features_by_ica(A, N,n_components=20):
    '''    
    Get the top N loaded features across n_components using ICA. You should standardize your columns yourself (features).

    Parameters:
    A (np.matrix): A numpy matrix.
    N (integer): Number of features to return.
    n_components (integer): Number of components to compute for ICA

    Returns:
    Array of N indices
    '''
    
    # Apply ICA
    ica = sklearn.decomposition.FastICA(n_components=n_components)
    S_ = ica.fit_transform(A)  # Reconstruct signals
    A_ = ica.mixing_  # Get the mixing matrix

    # Get the absolute weights and rank features within each component
    component_ranks = np.argsort(-np.abs(A_), axis=0)

    # Select N unique features by cycling through components
    selected_features = set()
    num_components = A_.shape[1]
    idx = 0
    while len(selected_features) < N:
        component = idx % num_components  # Cycle through components
        feature_candidates = component_ranks[:, component]
        for feature in feature_candidates:
            if feature not in selected_features:
                selected_features.add(feature)
                break
        idx += 1
        if idx > 1000:  # Safety break to avoid infinite loop
            break

    return list(selected_features)