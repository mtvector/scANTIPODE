#Just run for longer
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

max_steps=2000000
batch_key=sys.argv[3]
discov_key=sys.argv[4]
layer_key=sys.argv[5]
model_tag=str(sys.argv[2])

adata = sc.read_h5ad(sys.argv[1],backed='r')

adata.obs['female']=adata.obs['batch_name'].replace(pd.read_csv('/home/matthew.schmitz/Matthew/data/taxtest/extra/inferred_sex.csv',index_col=0).to_dict()['female'])
adata.obsm['phase_sex']=np.concatenate([adata.obs['S_score'].to_numpy().reshape(-1,1),adata.obs['G2M_score'].to_numpy().reshape(-1,1),adata.obs['log10_n_counts'].to_numpy().reshape(-1,1),antipode.model_functions.numpy_onehot(adata.obs['female'].cat.codes)],axis=1)

antipode_model=antipode.antipode_model.ANTIPODE.load(sys.argv[2],adata=adata,prefix='p4_',device=device)
antipode_model.train()

antipode_model.train_phase(phase=3,max_steps=max_steps,print_every=10000,num_particles=5,device=device, max_learning_rate=1e-3, one_cycle_lr=True, steps=0, batch_size=64)

antipode_model.store_outputs(device=device,prefix='')
antipode_model.clear_cuda()
MDE_KEY = "X_antipode_MDE"
adata.obsm[MDE_KEY] = clip_latent_dimensions(scvi.model.utils.mde(adata.obsm['X_antipode'],init='random'),0.1)

antipode_model.save(sys.argv[2],save_anndata=True,prefix='p5_')

# import rapids_singlecell as rsc
# dim_weights=np.absolute(adata.uns['param_store']['z_decoder_weight']).mean(1)
# #adata.obsm['X_weighted_antipode']=adata.obsm['X_antipode']*dim_weights
# rsc.pp.neighbors(adata, n_neighbors=50, use_rep='X_antipode')
# rsc.tl.umap(adata,spread=4.,min_dist=0.)
# adata.obsm['X_umap']=antipode.plotting.clip_latent_dimensions(adata.obsm['X_umap'],0.01)

antipode_model.save(sys.argv[2],save_anndata=True,prefix='p5_')
np.savetxt(os.path.join(sys.argv[2],"X_umap.csv"), adata.obsm['X_umap'], delimiter=",")
