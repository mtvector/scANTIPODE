import seaborn
import numpy as np
import pandas as pd
import scipy
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.patches as patches
import tqdm
import scanpy as sc
import sklearn
import os
import itertools
from typing import List, Union, Tuple, Literal
import glasbey
from colorspacious import cspace_convert

from . import model_functions
try:
    import gseapy
except:
    print("GSEApy not found. Can't get module enrichments")

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def moving_average_values(x,y,window_size=1001):
    ma = moving_average(y[np.argsort(x)],window_size)
    return(np.sort(x[int(window_size/2):-int(window_size/2)]),ma)
    
    
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

def plot_batch_embedding_pca(antipode_model,color_key=None, save=False):
    """Plot PCA of batch embeddings; save as SVG if save is True."""
    try:
        if color_key is None:
            color_key = antipode_model.discov_key
        colors = antipode_model.adata_manager.adata.uns[color_key + '_colors']        
        pca = sklearn.decomposition.PCA(n_components=2)
        batch_eye = torch.eye(antipode_model.num_batch)
        batch_pca = pca.fit_transform(
            model_functions.centered_sigmoid(antipode_model.be_nn.cpu()(batch_eye))
            .detach().numpy()
        )
        df = pd.DataFrame(batch_pca)
        batch_species = (antipode_model.adata_manager.adata.obs
                         .groupby(antipode_model.batch_key)[color_key]
                         .value_counts()
                         .unstack()
                         .idxmax(axis=1)
                         .to_dict())

        batches = antipode_model.adata_manager.adata.obs[antipode_model.batch_key].cat.categories
        df[antipode_model.batch_key] = batches
        df[color_key] = [batch_species[x] for x in batches]
        df[color_key] = df[color_key].astype('category')
        df[color_key] = df[color_key].cat.reorder_categories(antipode_model.adata_manager.adata.obs[color_key].cat.categories)
        seaborn.scatterplot(data=df, x=0, y=1, hue=color_key,palette = colors)
        plt.xlabel('batch embedding PC1')
        plt.ylabel('batch embedding PC2')
        if save:
            filename = os.path.join(sc.settings.figdir, 'plot_batch_embedding_pca_'+color_key+'.svg')
            plt.savefig(filename, format='svg')
            plt.show()
        else:
            plt.show()
        return df
    except Exception as e:
        print('plot failed:', e)

import os
import matplotlib.pyplot as plt

def plot_d_hists(antipode_model, bins=200, save=False, ecdf=False):
    """Plot delta parameter distributions as histograms or ECDFs.
    
    For each of DM, DI, and DC parameters, plots either histograms (default) or ECDF plots
    if ecdf=True. If save is True, each plot is saved as an SVG in the folder specified by sc.settings.figdir.
    
    Parameters
    ----------
    antipode_model : object
        The model object containing data and parameter stores.
    bins : int, optional
        Number of bins for histograms (ignored for ECDF plots), by default 200.
    save : bool, optional
        If True, save the plot as an SVG file, by default False.
    ecdf : bool, optional
        If True, render ECDF plots instead of histograms, by default False.
    """
    categories = antipode_model.adata_manager.registry['field_registries'][
        'discov_ind']['state_registry']['categorical_mapping']
    colors = antipode_model.adata_manager.adata.uns[
        antipode_model.adata_manager.registry['field_registries'][
            'discov_ind']['state_registry']['original_key'] + '_colors']
    param_store = antipode_model.adata_manager.adata.uns['param_store']

    # DM plot
    plt.figure()
    if ecdf:
        seaborn.ecdfplot(x=param_store['locs'].flatten(), color="slategray", label='shared')
        for i in range(len(categories)):
            seaborn.ecdfplot(x=param_store['discov_dm'][i, ...].flatten(), color=colors[i],
                              label=categories[i])
        seaborn.ecdfplot(x=param_store['batch_dm'].flatten(), color='lightgrey', label='batch')
    else:
        seaborn.histplot(param_store['locs'].flatten(), color="slategray",
                          label='shared', bins=bins, stat='proportion')
        for i in range(len(categories)):
            seaborn.histplot(param_store['discov_dm'][i, ...].flatten(), color=colors[i],
                              bins=bins, label=categories[i], stat='proportion')
        seaborn.histplot(param_store['batch_dm'].flatten(), color='lightgrey',
                          bins=bins, label='batch', stat='proportion')
    plt.legend()
    plt.title('DM')
    if save:
        filename = os.path.join(sc.settings.figdir, 'plot_d_hists_DM.svg')
        plt.savefig(filename, format='svg')
    plt.show()

    # DI plot
    plt.figure()
    if ecdf:
        seaborn.ecdfplot(x=param_store['cluster_intercept'].flatten(), color="slategray",
                          label='shared')
        for i in range(len(categories)):
            seaborn.ecdfplot(x=param_store['discov_di'][i, ...].flatten(), color=colors[i],
                              label=categories[i])
        seaborn.ecdfplot(x=param_store['batch_di'].flatten(), color='lightgrey',
                          label='batch')
    else:
        seaborn.histplot(param_store['cluster_intercept'].flatten(), color="slategray",
                          label='shared', bins=bins, stat='proportion')
        for i in range(len(categories)):
            seaborn.histplot(param_store['discov_di'][i, ...].flatten(), color=colors[i],
                              bins=bins, label=categories[i], stat='proportion')
        seaborn.histplot(param_store['batch_di'].flatten(), color='lightgrey',
                          bins=bins, label='batch', stat='proportion')
    plt.legend()
    plt.title('DI')
    if save:
        filename = os.path.join(sc.settings.figdir, 'plot_d_hists_DI.svg')
        plt.savefig(filename, format='svg')
    plt.show()

    # DC plot
    plt.figure()
    if ecdf:
        seaborn.ecdfplot(x=param_store['z_decoder_weight'].flatten(), color="slategray",
                          label='shared')
        for i in range(len(categories)):
            seaborn.ecdfplot(x=param_store['discov_dc'][i, ...].flatten(), color=colors[i],
                              label=categories[i])
    else:
        seaborn.histplot(param_store['z_decoder_weight'].flatten(), color="slategray",
                          label='shared', bins=bins, stat='proportion')
        for i in range(len(categories)):
            seaborn.histplot(param_store['discov_dc'][i, ...].flatten(), color=colors[i],
                              bins=bins, label=categories[i], stat='proportion')
    plt.legend()
    plt.title('DC')
    if save:
        filename = os.path.join(sc.settings.figdir, 'plot_d_hists_DC.svg')
        plt.savefig(filename, format='svg')
    plt.show()
   
    plt.figure()
    if ecdf:
        for i in range(len(categories)):
            seaborn.ecdfplot(x=(param_store['discov_constitutive_de']-param_store['discov_constitutive_de'].mean(0))[i, ...].flatten(), color=colors[i],
                              label=categories[i])
    else:
        for i in range(len(categories)):
            seaborn.histplot((param_store['discov_constitutive_de']-param_store['discov_constitutive_de'].mean(0))[i, ...].flatten(), color=colors[i],
                              bins=bins, label=categories[i], stat='proportion')
    plt.legend()
    plt.title('DA')
    if save:
        filename = os.path.join(sc.settings.figdir, 'plot_d_hists_DA.svg')
        plt.savefig(filename, format='svg')
    plt.show()



def plot_tree_edge_weights(antipode_model, save=False):
    """Plot heatmaps of tree edge weights; save each as SVG if save is True."""
    param_store = antipode_model.adata_manager.adata.uns['param_store']
    for name, param in param_store.items():
        if 'edge' in name:
            seaborn.heatmap(scipy.special.softmax(param, axis=-1))
            plt.title(name)
            if save:
                filename = os.path.join(sc.settings.figdir, f'plot_tree_edge_weights_{name}.svg')
                plt.savefig(filename, format='svg')
                plt.show()
            else:
                plt.show()


def plot_gmm_heatmaps(antipode_model, save=False):
    """Plot GMM clustermaps and histograms; save each plot as SVG if save is True."""
    categories = antipode_model.adata_manager.registry['field_registries'][
        'discov_ind']['state_registry']['categorical_mapping']
    colors = antipode_model.adata_manager.adata.uns[
        antipode_model.adata_manager.registry['field_registries'][
            'discov_ind']['state_registry']['original_key'] + '_colors']
    param_store = antipode_model.adata_manager.adata.uns['param_store']

    # locs clustermap
    data = antipode_model.z_transform(torch.tensor(param_store['locs'])).numpy()
    g = seaborn.clustermap(data, cmap='coolwarm', center=0)
    plt.title('locs')
    if save:
        filename = os.path.join(sc.settings.figdir, 'plot_gmm_heatmaps_locs.svg')
        g.fig.savefig(filename, format='svg')
        plt.show(g.fig)
    else:
        plt.show()

    # locs_dynam clustermap
    g = seaborn.clustermap(param_store['locs_dynam'], cmap='coolwarm', center=0)
    plt.title('locs_dynam')
    if save:
        filename = os.path.join(sc.settings.figdir, 'plot_gmm_heatmaps_locs_dynam.svg')
        g.fig.savefig(filename, format='svg')
        plt.show(g.fig)
    else:
        plt.show()

    # scales clustermap
    g = seaborn.clustermap(param_store['scales'], cmap='coolwarm')
    plt.title('scales')
    if save:
        filename = os.path.join(sc.settings.figdir, 'plot_gmm_heatmaps_scales.svg')
        g.fig.savefig(filename, format='svg')
        plt.show(g.fig)
    else:
        plt.show()

    # inverse_dispersion histogram
    seaborn.histplot(param_store['s_inverse_dispersion'].flatten(), color='grey', bins=50)
    plt.title('inverse_dispersion')
    if save:
        filename = os.path.join(sc.settings.figdir, 'plot_gmm_heatmaps_inverse_dispersion.svg')
        plt.savefig(filename, format='svg')
        plt.show()
    else:
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

# Helper: convert a hex string (e.g. "#RRGGBB") to an sRGB triple in [0, 1].
def hex_to_rgb(hex_color: str) -> np.ndarray:
    hex_color = hex_color.lstrip('#')
    return np.array([int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4)], dtype=np.float32)

def create_hierarchical_palette(adata, levels, palette_func, global_hue_bounds=(0, 360), uns_key_prefix="", hue_cushion: float = 0, **palette_kwargs):
    """
    Create hierarchical color palettes for sequential levels such that the lower levels
    are arranged in blocks corresponding to the top-level. For each lower level, the full
    hue range (global_hue_bounds) is divided into as many equal wedges as there are top-level
    categories. Then, each wedge is re-centered on the hue of the corresponding top-level color,
    with an optional cushion (hue_cushion, in degrees) reserved between groups. The restricted
    hue range is passed to palette_func to generate the block for that top-level category.
    
    The function reassigns the categorical order in adata.obs for each level and stores both a
    dictionary mapping and an ordered color list in adata.uns. Probably shouldn't have been based
    on anndata, but here we are...
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object with categorical columns in .obs for each level.
    levels : list of str
        List of level names from top to bottom, e.g.
        ["Class", "Subclass_markers", "Group", "consensus_cluster"].
    palette_func : function
        Function that generates a palette given a block size (an int) and (optionally) hue_bounds.
        (For example, it could be a restricted version of glasbey.create_palette.)
    global_hue_bounds : tuple (float, float)
        The full hue range, e.g. (0, 360).
    uns_key_prefix : str, optional
        Optional prefix for keys in adata.uns.
    hue_cushion : float, optional
        The proportino of color space to reserve as a gap between top-level groups.
        Default is 0.
    **palette_kwargs :
        Additional keyword arguments passed to palette_func (e.g. grid_size, grid_space, lightness_bounds,
        chroma_bounds, colorblind_safe, as_hex, etc.).
    
    Returns
    -------
    palette_mappings : dict
        Dictionary with one entry per level. Each entry is a dict with:
          - "order": list of category names (in hierarchical order)
          - "mapping": dict mapping each category to its assigned color.
    """
    if "theme_color_spacing" in palette_kwargs:
        del palette_kwargs["theme_color_spacing"]
    hue_cushion = min(hue_cushion,0.99)
    hue_cushion = max(hue_cushion,0.)
    
    palette_mappings = {}
    
    # --- Top Level ---
    top_level = levels[0]
    top_order = list(adata.obs[top_level].cat.categories)
    # Generate one color per top-level category in one call.
    top_palette = glasbey.create_palette(len(top_order), **palette_kwargs)
    top_mapping = {cat: col for cat, col in zip(top_order, top_palette)}
    palette_mappings[top_level] = {"order": top_order, "mapping": top_mapping}
    adata.uns[f"{uns_key_prefix}{top_level}_colors_dict"] = top_mapping
    adata.uns[f"{uns_key_prefix}{top_level}_colors"] = [top_mapping[cat] for cat in top_order]
    
    # For each subsequent level, group by the top-level.
    for i in range(1, len(levels)):
        current_level = levels[i]
        current_global_order = list(adata.obs[current_level].cat.categories)
        
        # Group by top-level: determine for each child its top-level parent.
        grouping_dict = (
            adata.obs.groupby(current_level)[top_level]
            .value_counts().unstack()
            .idxmax(1).to_dict()
        )
        
        # Build hierarchical ordering: for each top-level category in order, select those child categories.
        current_order = []
        top_to_children = {}
        for t in top_order:
            children = [child for child in current_global_order if grouping_dict.get(child) == t]
            top_to_children[t] = children
            current_order.extend(children)
        
        adata.obs[current_level] = adata.obs[current_level].cat.reorder_categories(current_order)
        
        # For each top-level category, get its block size.
        block_sizes = [len(top_to_children[t]) for t in top_order]
        wedge_size = (global_hue_bounds[1] - global_hue_bounds[0]) / len(top_order)
        
        current_palette = []
        for idx, t in enumerate(top_order):
            n_children = len(top_to_children[t])
            if n_children == 0:
                continue
            top_color = top_mapping[t]
            if isinstance(top_color, str):
                srgb = hex_to_rgb(top_color)
            else:
                srgb = np.array(top_color, dtype=np.float32)
            top_color_jch = cspace_convert(np.array([srgb]), "sRGB1", "JCh")[0]
            center_hue = top_color_jch[2]
            # Apply hue cushion: available width equals wedge_size minus hue_cushion.
            available_width = wedge_size - wedge_size*hue_cushion
            if available_width < 0:
                available_width = 0
            restricted_hue_bounds = (center_hue - available_width/2, center_hue + available_width/2)
            block_palette = glasbey.create_palette(n_children, hue_bounds=restricted_hue_bounds, **palette_kwargs)
            current_palette.extend(block_palette)
        
        current_mapping = {}
        start_idx = 0
        for t in top_order:
            n_children = len(top_to_children[t])
            block_colors = current_palette[start_idx:start_idx + n_children]
            start_idx += n_children
            for child, col in zip(top_to_children[t], block_colors):
                current_mapping[child] = col
        
        palette_mappings[current_level] = {"order": current_order, "mapping": current_mapping}
        adata.uns[f"{uns_key_prefix}{current_level}_colors_dict"] = current_mapping
        adata.uns[f"{uns_key_prefix}{current_level}_colors"] = [current_mapping[cat] for cat in current_order]
    
    return palette_mappings

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

def pie_dotplot(means_list, proportions_list, column_names, row_names,
                colormaps=None, max_radius=0.4, figsize=(10, 8),
                scale_by='column', mean_vmin=None, mean_vmax=None):
    """
    Plots a grid of pie charts where:
      - Each cell is drawn as a pie chart with equal angular slices.
      - The outer radius of each slice is scaled by its proportion value.
      - The color of each slice is determined by the mean expression value,
        normalized either per column (gene), per row (cluster), or using global
        values per slice (if scale_by is None).
    
    In addition, a legend is added to the right of the plot for each slice
    showing its colormap with the global min and max values.
    
    Parameters:
    -----------
    means_list : list of np.ndarray
        List of matrices (shape: [n_clusters, n_genes]) containing mean expression values.
    proportions_list : list of np.ndarray
        List of matrices (shape: [n_clusters, n_genes]) containing proportion values.
    column_names : list of str
        Names for genes to label the X-axis.
    row_names : list of str
        Names for clusters to label the Y-axis.
    colormaps : list of str, optional
        List of colormap names (one per slice). Defaults to a set of common colormaps.
    max_radius : float, optional
        Maximum radius for a full (proportion = 1) slice.
    figsize : tuple, optional
        Figure size.
    scale_by : str or None, optional
        If "column", each gene column is normalized to [0, 1] using the min and max
        across all clusters and slices. If "row", each cluster row is normalized.
        If None, each slice is normalized using its own global min/max (computed from means_list).
    mean_vmin, mean_vmax : list of floats, optional
        When scale_by is None, these can be provided for each slice; otherwise they are computed.
    
    Returns:
    --------
    fig, ax : matplotlib Figure and Axes
    """
    if len(means_list) != len(proportions_list):
        raise ValueError("means_list and proportions_list must have the same length")
    n_slices = len(means_list)
    n_clusters, n_genes = means_list[0].shape

    # Set default colormaps if not provided.
    if colormaps is None:
        default_cmaps = ['Blues', 'Oranges', 'Greens', 'Purples', 'Reds', 'Greys']
        if n_slices > len(default_cmaps):
            colormaps = [default_cmaps[i % len(default_cmaps)] for i in range(n_slices)]
        else:
            colormaps = default_cmaps[:n_slices]

    # For legend purposes, compute global min/max for each slice.
    global_vmin = []
    global_vmax = []
    for k in range(n_slices):
        global_vmin.append(np.min(means_list[k]))
        global_vmax.append(np.max(means_list[k]))

    # Precompute normalization ranges for cell-level scaling.
    if scale_by is None:
        if mean_vmin is None:
            mean_vmin = [np.min(m) for m in means_list]
        if mean_vmax is None:
            mean_vmax = [np.max(m) for m in means_list]
    elif scale_by == 'column':
        # For each gene (column), compute the min and max over all clusters and slices.
        col_vmin = np.empty(n_genes)
        col_vmax = np.empty(n_genes)
        for j in range(n_genes):
            col_vmin[j] = min(np.min(matrix[:, j]) for matrix in means_list)
            col_vmax[j] = max(np.max(matrix[:, j]) for matrix in means_list)
    elif scale_by == 'row':
        # For each cluster (row), compute the min and max over all genes and slices.
        row_vmin = np.empty(n_clusters)
        row_vmax = np.empty(n_clusters)
        for i in range(n_clusters):
            row_vmin[i] = min(np.min(matrix[i, :]) for matrix in means_list)
            row_vmax[i] = max(np.max(matrix[i, :]) for matrix in means_list)
    else:
        raise ValueError("scale_by must be either 'column', 'row', or None.")

    fig, ax = plt.subplots(figsize=figsize)

    # Adjust font sizes based on number of genes/clusters.
    x_fontsize = max(6, min(10, 300 / n_genes))
    y_fontsize = max(6, min(10, 300 / n_clusters))
    ax.tick_params(axis='x', labelsize=x_fontsize)
    ax.tick_params(axis='y', labelsize=y_fontsize)

    # Loop over each grid cell.
    for i in range(n_clusters):
        for j in range(n_genes):
            center = (j, i)
            for k in range(n_slices):
                mean_val = means_list[k][i, j]
                prop_val = proportions_list[k][i, j]
                # Define angular range for the slice.
                theta1 = 360 * k / n_slices + 30
                theta2 = 360 * (k + 1) / n_slices + 30
                radius = prop_val * max_radius

                # Determine normalization range.
                if scale_by == 'column':
                    vmin = col_vmin[j]
                    vmax = col_vmax[j]
                elif scale_by == 'row':
                    vmin = row_vmin[i]
                    vmax = row_vmax[i]
                else:
                    vmin = mean_vmin[k]
                    vmax = mean_vmax[k]

                if vmax > vmin:
                    norm_val = (mean_val - vmin) / (vmax - vmin)
                else:
                    norm_val = 0.5

                cmap = plt.get_cmap(colormaps[k])
                color = cmap(norm_val)
                wedge = Wedge(center, radius, theta1, theta2, facecolor=color, edgecolor='none', linewidth=0)
                ax.add_patch(wedge)

    ax.set_xlim(-0.5, n_genes - 0.5)
    ax.set_ylim(-0.5, n_clusters - 0.5)
    ax.set_xticks(range(n_genes))
    ax.set_xticklabels(column_names, rotation=90)
    ax.set_yticks(range(n_clusters))
    ax.set_yticklabels(row_names)
    ax.set_aspect('equal')

    # Adjust the main axis to leave space on the right for colorbars.
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Create a colorbar for each slice (colormap).
    # We'll stack them vertically on the right side.
    n_colorbars = n_slices
    cbar_width = 0.03
    cbar_gap = 0.02
    total_height = 0.9  # relative to figure height
    cbar_height = (total_height - (n_colorbars - 1) * cbar_gap) / n_colorbars
    # Starting bottom position.
    bottom_start = 0.05

    for k in range(n_slices):
        # For legend, we use the global min/max of each slice.
        norm = Normalize(vmin=global_vmin[k], vmax=global_vmax[k])
        sm = ScalarMappable(norm=norm, cmap=plt.get_cmap(colormaps[k]))
        sm.set_array([])

        # Compute position for this colorbar.
        cbar_bottom = bottom_start + (n_colorbars - k - 1) * (cbar_height + cbar_gap)
        cax = fig.add_axes([0.87, cbar_bottom, cbar_width, cbar_height])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label(f'Slice {k+1}\n({colormaps[k]})', fontsize=8)

    return fig, ax


def plot_tricolor_heatmap(data, x_tick_labels=None, y_tick_labels=None,color_axes=['Blue','Red','Yellow'],
                          heatmap_figsize=(10, 4), legend_figsize=(6, 6),save_prefix=None):
    """
    Plots a tricolor (RBY) heatmap using the mapping:
      R = 1 - a
      G = 1 - (a + b)
      B = 1 - (b + c)
    where data is a (3, height, width) array. This mapping gives:
      [max, 0, 0] -> blue   (0,0,1)
      [0, max, 0] -> red    (1,0,0)
      [0, 0, max] -> yellow (1,1,0)
    with white at [0,0,0] and black at [max,max,max].
    
    It also creates a ternary legend (in a separate figure) illustrating the 
    color gradient with the vertices labeled.
    
    Parameters:
      data            : np.array of shape (3, height, width)
      x_tick_labels   : list/array of x-axis tick labels (rotated 90°)
      y_tick_labels   : list/array of y-axis tick labels
      heatmap_figsize : tuple for the heatmap figure size (default (10,4))
      legend_figsize  : tuple for the legend figure size (default (6,6))
    """
    # -----------------------------
    # Part 1: Create and plot the tricolor heatmap
    # -----------------------------
    # Normalize the data so that the maximum value is 1.
    data_norm = data / np.max(data)
    
    # Unpack channels:
    a = data_norm[0, :, :]  # will control blue
    b = data_norm[1, :, :]  # will control red
    c = data_norm[2, :, :]  # will control yellow
    
    # Mapping from (a, b, c) to RGB:
    #   [max, 0, 0] (a=1)  -> blue:   (0,0,1)
    #   [0, max, 0] (b=1)  -> red:    (1,0,0)
    #   [0, 0, max] (c=1)  -> yellow: (1,1,0)
    # with white at [0,0,0] and black at [max, max, max].
    R = 1 - a
    G = 1 - (a + b)
    B = 1 - (b + c)
    
    # Clip values to ensure valid RGB in [0,1]
    R = np.clip(R, 0, 1)
    G = np.clip(G, 0, 1)
    B = np.clip(B, 0, 1)
    
    # Form the image with shape (height, width, 3)
    img = np.stack([R, G, B], axis=2)
    
    plt.figure(figsize=heatmap_figsize)
    plt.imshow(img)
    plt.title("Tricolor Heatmap (RBY)")
    if x_tick_labels is not None:
        plt.xticks(ticks=np.arange(len(x_tick_labels)), labels=x_tick_labels, rotation=90)
    if y_tick_labels is not None:
        plt.yticks(ticks=np.arange(len(y_tick_labels)), labels=y_tick_labels)
    else:
        plt.yticks([])
    plt.tight_layout()
    if save_prefix is not None:
        plt.savefig(save_prefix+'_heatmap.svg')
    plt.show()
    
    # -----------------------------
    # Part 2: Create a triangular (ternary) legend
    # -----------------------------
    # The legend shows a ternary plot where each vertex corresponds to:
    #   (a=1, b=0, c=0) -> blue   (0,0,1)
    #   (a=0, b=1, c=0) -> red    (1,0,0)
    #   (a=0, b=0, c=1) -> yellow (1,1,0)
    def project(a, b, c):
        # Center the point by subtracting the mean (to handle translation invariance)
        mu = (a + b + c) / 3.0
        a_prime = a - mu
        b_prime = b - mu
        c_prime = c - mu
        # Apply affine projection:
        #   T(1,0,0) -> (0, 1)   [blue]
        #   T(0,1,0) -> (sqrt3/2, -1/2)  [red]
        #   T(0,0,1) -> (-sqrt3/2, -1/2) [yellow]
        x = (np.sqrt(3) / 2) * (b_prime - c_prime)
        y = a_prime - 0.5 * (b_prime + c_prime)
        return x, y
    
    # Generate many points uniformly in the simplex S (a+b+c <= 1)
    N = 500000  # number of random points for a smooth legend
    points = []
    colors = []
    for _ in range(N):
        pt = np.random.rand(3)
        if pt.sum() <= 1:
            a_pt, b_pt, c_pt = pt
            # Compute color according to RBY mapping:
            R_pt = 1 - a_pt
            G_pt = 1 - (a_pt + b_pt)
            B_pt = 1 - (b_pt + c_pt)
            R_pt = np.clip(R_pt, 0, 1)
            G_pt = np.clip(G_pt, 0, 1)
            B_pt = np.clip(B_pt, 0, 1)
            colors.append((R_pt, G_pt, B_pt))
            points.append((a_pt, b_pt, c_pt))
    points = np.array(points)
    colors = np.array(colors)
    
    # Project each (a, b, c) to 2D
    xy = np.array([project(a_pt, b_pt, c_pt) for a_pt, b_pt, c_pt in points])
    x_vals = xy[:, 0]
    y_vals = xy[:, 1]
    
    # Define vertices for the pure channels.
    x_blue, y_blue     = project(1, 0, 0)    # blue: [max,0,0]
    x_red, y_red       = project(0, 1, 0)    # red:  [0,max,0]
    x_yellow, y_yellow = project(0, 0, 1)    # yellow: [0,0,max]
    
    plt.figure(figsize=legend_figsize)
    plt.scatter(x_vals, y_vals, c=colors, s=10, marker='s', edgecolors='none', alpha=0.5)
    
    # Draw the boundary triangle connecting the three pure colors.
    triangle_x = [x_blue, x_red, x_yellow, x_blue]
    triangle_y = [y_blue, y_red, y_yellow, y_blue]
    plt.plot(triangle_x, triangle_y, color='black', lw=1.5)
    
    # Annotate vertices.
    plt.text(x_blue, y_blue + 0.05, f"[max,0,0]\n({color_axes[0]})", color='blue',
             fontsize=10, ha='center', va='bottom')
    plt.text(x_red, y_red - 0.05, f"[0,max,0]\n({color_axes[1]})", color='red',
             fontsize=10, ha='center', va='top')
    plt.text(x_yellow, y_yellow - 0.05, f"[0,0,max]\n({color_axes[2]})", color='goldenrod',
             fontsize=10, ha='center', va='top')
    
    # Mark the center (which represents [0,0,0] white)
    plt.scatter(0, 0, s=50, color='white', edgecolor='black', zorder=10)
    plt.text(0, 0.05, "[0,0,0]", color='black', fontsize=10, ha='center', va='bottom')
    
    plt.axis('off')
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    if save_prefix is not None:
        plt.savefig(save_prefix+'_legend.svg')
    plt.show()

def plot_tile_heatmap(data, dim_0_names, dim_1_names, dim_2_names, mini_grid_dims=None, 
                      cell_size=1, cmap_name='tab20', heavy_linewidth=2, light_linewidth=0.2,save_path=None):
    """
    Plot a nested heatmap where each cell (grid pair) is subdivided into mini tiles
    representing ligand–receptor interaction strengths, and adds a legend for ligands.
    
    Parameters
    ----------
    data : np.ndarray
        3D array of shape (n_rows, n_cols, n_ligands) with values in [0, 1].
    dim_0_names : list of str
        Labels for the rows (y-axis). Length should equal n_rows.
    dim_1_names : list of str
        Labels for the columns (x-axis). Length should equal n_cols.
    dim_2_names : list of str
        Names for each ligand (used to assign colors and for the legend).
    mini_grid_dims : tuple of ints, optional
        Dimensions (rows, cols) of the mini grid inside each cell. If None, the grid
        will be as square as possible.
    cell_size : float, optional
        The size of each cell in the heatmap.
    cmap_name : str, optional
        Name of a qualitative colormap (e.g. 'tab20') to assign each ligand a unique base color.
    heavy_linewidth : float, optional
        Line width for the boundaries between cells.
    light_linewidth : float, optional
        Line width for the subgrid (mini-tile) borders.
    """
    n_rows_data, n_cols_data, n_ligands = data.shape

    # Validate that provided labels match the data dimensions.
    if len(dim_0_names) != n_rows_data:
        raise ValueError(f"Expected {n_rows_data} row labels (dim_0_names), but got {len(dim_0_names)}.")
    if len(dim_1_names) != n_cols_data:
        raise ValueError(f"Expected {n_cols_data} column labels (dim_1_names), but got {len(dim_1_names)}.")

    # Determine mini-grid dimensions (rows x cols) for subdividing each cell.
    if mini_grid_dims is None:
        n_rows = int(np.floor(np.sqrt(n_ligands)))
        n_rows = max(n_rows, 1)
        n_cols = int(np.ceil(n_ligands / n_rows))
    else:
        n_rows, n_cols = mini_grid_dims

    # Get base colors for each ligand using the specified qualitative colormap.
    cmap = plt.get_cmap(cmap_name)
    base_colors = [cmap(i / n_ligands) for i in range(n_ligands)]
    
    # Create figure and axis.
    fig, ax = plt.subplots(figsize=(n_cols_data * cell_size, n_rows_data * cell_size))
    ax.set_xlim(0, n_cols_data * cell_size)
    ax.set_ylim(0, n_rows_data * cell_size)
    ax.set_aspect('equal')
    # Invert y-axis so that the first row appears at the top.
    ax.invert_yaxis()
    
    # Loop over each cell in the grid.
    for i in tqdm.tqdm(range(n_rows_data)):
        for j in range(n_cols_data):
            cell_x = j * cell_size
            cell_y = i * cell_size
            # Draw heavy border for the cell.
            ax.add_patch(patches.Rectangle((cell_x, cell_y), cell_size, cell_size, 
                                           fill=False, edgecolor='black', lw=heavy_linewidth))
            # Dimensions of each mini-tile.
            tile_w = cell_size / n_cols
            tile_h = cell_size / n_rows
            # Loop over each ligand (mini-tile).
            for k in range(n_ligands):
                mini_row = k // n_cols
                mini_col = k % n_cols
                tile_x = cell_x + mini_col * tile_w
                # Remove the inversion of mini_row to flip the order relative to the legend.
                tile_y = cell_y + mini_row * tile_h
                v = data[i, j, k]
                base = np.array(base_colors[k][:3])  # Drop alpha if present.
                color = (1 - v) * np.array([1, 1, 1]) + v * base
                ax.add_patch(patches.Rectangle((tile_x, tile_y), tile_w, tile_h, 
                                               facecolor=color, edgecolor='gray', lw=light_linewidth))
    
    # Set tick positions: columns on x-axis and rows on y-axis.
    xticks = np.arange(cell_size/2, n_cols_data * cell_size, cell_size)
    yticks = np.arange(cell_size/2, n_rows_data * cell_size, cell_size)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    # For rectangular grids, use dim_1_names for x-axis (columns) and dim_0_names for y-axis (rows).
    ax.set_xticklabels(dim_1_names,rotation=90)
    ax.set_yticklabels(dim_0_names)
    
    # Create a legend with color patches for each ligand.
    legend_handles = [patches.Patch(color=base_colors[k], label=dim_2_names[k]) 
                      for k in range(n_ligands)]
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', title='Ligands')
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

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