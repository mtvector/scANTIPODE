# various import statements
import os
import sys
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

max_steps=300000
batch_key='batch_name'
discov_key='species'
layer_key='spliced'

adata = sc.read_h5ad(sys.argv[1],backed='r')
model_tag=str(sys.argv[2])

antipode_model=ANTIPODE.load(sys.argv[2],adata=adata,prefix='p2_')
antipode_model.train()

antipode_model.train_phase(phase=3,max_steps=max_steps,print_every=10000,num_particles=7,device=device, max_learning_rate=5e-4, one_cycle_lr=True, steps=0, batch_size=64)

antipode_model.store_outputs(device=device,prefix='')
MDE_KEY = "X_antipode_MDE"
adata.obsm[MDE_KEY] = clip_latent_dimensions(scvi.model.utils.mde(adata.obsm['X_antipode'],init='random'),0.1)

antipode_model.save(sys.argv[2],save_anndata=True,prefix='p3_')