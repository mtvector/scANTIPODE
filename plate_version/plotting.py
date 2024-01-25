import seaborn
import numpy as np
import scipy
import torch
import matplotlib
import matplotlib.pyplot as plt
import model_functions

def plot_d_hists(antipode_model):
    categories=antipode_model.adata_manager.registry['field_registries']['species_ind']['state_registry']['categorical_mapping']
    colors=antipode_model.adata_manager.adata.uns[antipode_model.adata_manager.registry['field_registries']['species_ind']['state_registry']['original_key']+'_colors']
    param_store=antipode_model.adata_manager.adata.uns['param_store']
    
    seaborn.histplot(param_store['locs'].flatten(),color="lightgrey",label='ancestral',bins=50,stat='proportion')
    for i in range(len(categories)):
        seaborn.histplot(param_store['species_dm'][i,...].flatten(),color=colors[i],bins=50,label=categories[i],stat='proportion')
    plt.legend()
    plt.title('DM')
    plt.show()
    
    seaborn.histplot(param_store['cluster_intercept'].flatten(),color='lightgrey',bins=50,label='ancestral',stat='proportion')
    for i in range(len(categories)):
        seaborn.histplot(param_store['species_di'][i,...].flatten(),color=colors[i],bins=50,label=categories[i],stat='proportion')
    plt.legend()
    plt.title('DI')
    plt.show()
    
    seaborn.histplot(param_store['z_decoder_weight'].flatten(),color='lightgrey',bins=50,label='ancestral',stat='proportion')
    for i in range(len(categories)):
        seaborn.histplot(param_store['species_dc'][i,...].flatten(),color=colors[i],bins=50,label=categories[i],stat='proportion')
    plt.legend()
    plt.title('DC')
    plt.show()

def plot_tree_edge_weights(antipode_model):
    for name in antipode_model.adata_manager.adata.uns['param_store'].keys():
        if 'edge' in name:
            seaborn.heatmap(scipy.special.softmax(antipode_model.adata_manager.adata.uns['param_store'][name],-1))
            plt.show()

def plot_gmm_heatmaps(antipode_model):
    categories=antipode_model.adata_manager.registry['field_registries']['species_ind']['state_registry']['categorical_mapping']
    colors=antipode_model.adata_manager.adata.uns[antipode_model.adata_manager.registry['field_registries']['species_ind']['state_registry']['original_key']+'_colors']
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
