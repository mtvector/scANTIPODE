#!/usr/bin/env python
# coding: utf-8

# various import statements
import os
import inspect
import seaborn
import matplotlib
import matplotlib.pyplot as plt
import torch
import scanpy as sc
import pyro

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print("GPU is available")
    print("Number of GPUs:", torch.cuda.device_count())
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("GPU is not available")

import sys
sys.path.append('/home/matthew.schmitz/Matthew/code/scANTIPODE/')
import antipode
from antipode.antipode_model import *
import antipode.model_functions
from antipode.model_functions import *
import antipode.model_distributions
from antipode.model_distributions import *
import antipode.model_modules
from antipode.model_modules import *
import antipode.train_utils
from antipode.train_utils import *
import antipode.plotting
from antipode.plotting import *



adata=sc.read_h5ad(os.path.expanduser('/allen/programs/celltypes/workgroups/rnaseqanalysis/EvoGen/Team/Matthew/data/taxtest/HvQvM/HvQvMall_cere_clean.h5ad'),backed='r')
batch_key='batch_name'
discov_key='species'
layer_key='spliced'


num_var=adata.shape[1]
batch_size=32
level_sizes=[1,50,350]
num_latent=200
steps=0
max_steps=200000
print_every=5000

# Clear Pyro param store so we don't conflict with previous run
try:
    pyro.clear_param_store()
    del antipode_model
    torch.cuda.empty_cache()
except:
    pass
# Fix random number seed to a lucky number
pyro.util.set_rng_seed(13)
# Enable optional validation warnings
pyro.enable_validation(False)

# Instantiate instance of model/guide and various neural networks
antipode_model = ANTIPODE(num_latent=num_latent,level_sizes=level_sizes,bi_depth=2,num_batch_embed=10,
                adata=adata,discov_pair=('obs',discov_key),batch_pair=('obs',batch_key),layer=layer_key,
                use_psi=True,use_q_score=True,prior_scale=50.,sampler_category='species',dist_normalize=False,
                scale_init_val=0.01,loc_as_param=False,zdw_as_param=False,intercept_as_param=False)


# # Training Phase 1: Block Approximation




antipode_model.train_phase(phase=1,max_steps=max_steps,print_every=10000,num_particles=7,device=device, max_learning_rate=0.001, one_cycle_lr=True, steps=0, batch_size=32)






antipode_model.clear_cuda()



antipode_model.store_outputs(device=device,prefix='')






antipode_model.clear_cuda()






plot_loss(antipode_model.losses)
plot_gmm_heatmaps(antipode_model)
plot_d_hists(antipode_model)
plot_batch_embedding_pca(antipode_model)





MDE_KEY = "X_antipode_MDE"
adata.obsm[MDE_KEY] = clip_latent_dimensions(scvi.model.utils.mde(adata.obsm['X_antipode'],init='random'),0.1)
sc.pl.embedding(
    adata,
    basis=MDE_KEY,
    color=["antipode_cluster"],legend_fontsize=6,legend_fontweight='normal',
    legend_loc='on data',palette=sc.pl.palettes.godsnot_102
)

sc.pl.embedding(
    adata,
    basis=MDE_KEY,
    color=['q_score',discov_key,batch_key],palette=sc.pl.palettes.godsnot_102,cmap='coolwarm'
)


# # Training Phase 2: Intializing layered tree




antipode_model.prepare_phase_2(epochs=2,device=device)





antipode_model.train_phase(phase=2,max_steps=int(max_steps/2),print_every=10000,num_particles=3,device=device, max_learning_rate=5e-4, one_cycle_lr=True, batch_size=64)






antipode_model.clear_cuda()






antipode_model.store_outputs(device=device,prefix='')






antipode_model.clear_cuda()






plot_loss(antipode_model.losses)
plot_gmm_heatmaps(antipode_model)
plot_d_hists(antipode_model)
plot_tree_edge_weights(antipode_model)





MDE_KEY = "X_antipode_MDE"
adata.obsm[MDE_KEY] = clip_latent_dimensions(scvi.model.utils.mde(adata.obsm['X_antipode'],init='random'),0.1)
sc.pl.embedding(
    adata,
    basis=MDE_KEY,
    color=["antipode_cluster","kmeans"],legend_fontsize=6,legend_fontweight='normal',
    legend_loc='on data',palette=sc.pl.palettes.godsnot_102
)

sc.pl.embedding(
    adata,
    basis=MDE_KEY,
    color=["psi",'q_score',discov_key,batch_key],palette=sc.pl.palettes.godsnot_102,cmap='coolwarm'
)

sc.pl.embedding(
    adata,
    basis=MDE_KEY,
    color=[x for x in adata.obs.columns if 'level' in x],
    palette=sc.pl.palettes.godsnot_102,
    legend_loc='on data'
)








# # Training Phase 3: Refining the final tree




antipode_model.train_phase(phase=3,max_steps=max_steps,print_every=10000,num_particles=5,device=device, max_learning_rate=5e-4, one_cycle_lr=True, steps=0, batch_size=32)





print(pyro.param('discov_mul'))






antipode_model.clear_cuda()






plot_loss(antipode_model.losses)





antipode_model.store_outputs(device=device,prefix='')






antipode_model.clear_cuda()






plot_gmm_heatmaps(antipode_model)
plot_d_hists(antipode_model)
plot_tree_edge_weights(antipode_model)





MDE_KEY = "X_antipode_MDE"
adata.obsm[MDE_KEY] = clip_latent_dimensions(scvi.model.utils.mde(adata.obsm['X_antipode'],init='random'),0.1)
sc.pl.embedding(
    adata,
    basis=MDE_KEY,
    color=["antipode_cluster","kmeans"],legend_fontsize=6,legend_fontweight='normal',
    legend_loc='on data',palette=sc.pl.palettes.godsnot_102
)

sc.pl.embedding(
    adata,
    basis=MDE_KEY,
    color=[x for x in adata.obs.columns if 'level' in x],
    palette=sc.pl.palettes.godsnot_102,
    legend_loc='on data'
)

sc.pl.embedding(
    adata,
    basis=MDE_KEY,
    color=["psi",'q_score',discov_key,batch_key],palette=sc.pl.palettes.godsnot_102,cmap='coolwarm'
)





seaborn.histplot(adata.obs['q_score'])





random_choice=np.random.choice(adata.obs.index,size=100000,replace=False)
random_choice=np.where(adata.obs.index.isin(random_choice))[0]
xdata=adata[random_choice,:]
xdata=xdata.to_memory().copy()





xdata.X=xdata.layers[layer_key]
sc.pp.normalize_per_cell(xdata)
sc.pp.log1p(xdata)
#sc.pp.scale(xdata,max_value=10)

gene_list=['RBFOX3','PDGFRA','AQP4','FOXJ1','AIF1','MOG','COL1A2','CD34','COL4A1','FOXG1','SATB2','RORB','SLC17A7','TLE4','FEZF2',
           'DLX2','PROX1','SCGN','NKX2-1','LHX6','SST','PVALB','CRABP1','MEIS2','TSHZ1','NPY','FOXP1','FOXP2','PDYN','PENK','ISL1',
           'MKI67','RPL7','RPS17','RPL13A','MEF2C',
           'HMX3','TH','LMX1A','TFAP2A','TFAP2B','RSPO1','NKX3-1','IGF1','ITPR2','OTX2','HOXB3','PAX1','PAX2','PAX3','PAX5','PAX6','PAX7','PAX8']
gene_list=[x for x in gene_list if x in xdata.var.index]
sc.pl.embedding(
    xdata,
    basis=MDE_KEY,
    color=gene_list,cmap='Purples',
    palette=sc.pl.palettes.godsnot_102,legend_fontsize=6,
    legend_loc='on data',use_raw=False
)





adata.write_h5ad('/home/matthew.schmitz/Matthew/1.9.1.6_DevTighter.h5ad')
antipode_model.adata_manager.adata=None
torch.save(antipode_model, '/home/matthew.schmitz/Matthew/1.9.1.6_DevTighter.antipode')
pyro.get_param_store().save('/home/matthew.schmitz/Matthew/1.9.1.6_DevTighter.antipode.paramstore')





pyro.get_param_store().load('/home/matthew.schmitz/Matthew/1.9.1.5_Dev.antipode.paramstore')
antipode=torch.load('/home/matthew.schmitz/Matthew/1.9.1.5_Dev.antipode')
adata=sc.read('/home/matthew.schmitz/Matthew/1.9.1.5_Dev.h5ad')

