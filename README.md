<h1 style="text-align: center;"> scANTIPODE</h1>
<a href="https://en.wikipedia.org/wiki/Antipodes"> 
<div align="center">
  <img src="assets/antipode_logo_alternate.png" alt="antipode logo!" width="200">
</div>
</a>

**NOTICE: This package will be under heavy development until publication, and will be subject to changes until release 0.1**




**S**ingle **C**ell **A**ncestral **N**ode **T**axonomy **I**nference by **P**artitioning **O**f **D**ifferential **E**xpression. The model is an extension of the SCVI paradigm--a structured generative, variational inference model developed for the simultaneous analysis (DE) and categorization (taxonomy generation) of cell types across evolution (or now any covariate) using single-cell RNA-seq data. Long ago it began as a hack of a simplified model of [scANVI](https://pyro.ai/examples/scanvi.html) and is built on the pytorch-based PPL [pyro](https://pyro.ai/). The model acts as an integration method, that learns interpretable differential expression in the process. Note that this means ANTIPODE will fail to integrate datasets of different datasets, or datasets with large disparities in quality or gene mean dispersions.

The complete procedure runs in 3 phases (but can also run fully supervised using only phase 2):

1. **The Fuzzy Phase:** Cells may belong to multiple types sampled from a bernoulli distribution, learns an integrated latent space with covariate effects, but is less straightforward to interpret.

2. **The Supervised Phase:** Discrete clustering is initialized from a supervised initialization (or defaults to a de novo k-means clustering in the latent space). Can take a supervised clustering and/or latent space for cells.

3. **The Free Phase:** All parameters are released for unconstrained learning.


You can read about the generative model [here](https://www.overleaf.com/read/nmcmcjtvmfcb#acf7a4). You can look at example runs [here](examples/outputs/).


## Installation
First create a conda environment with python >= 3.10
```
git clone git@github.com:mtvector/scANTIPODE.git
#cuda 11.7 should work too
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install jax jaxlib -c conda-forge
cd scANTIPODE
pip install -e .

Additional analysis stuff/phylo mode

conda install biopython
pip install gseapy

```

Instead of separately installing pytorch, lately I've favored a full [RAPIDS install](https://docs.rapids.ai/install/#selector), as this should provide a robust environmnet setup and give you access access to fun tools like GPU accellerated UMAP.

```
# So in place of: conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda create -n antipode -c rapidsai -c conda-forge -c nvidia  \
    rapids=24.12 python=3.12 'cuda-version>=12.0,<=12.5' \
    'pytorch=*=*cuda*'
```


Please reach out to let me know if you try ANTIPODE on a dataset and it works (or doesn't work)... The model is (forever) a work in process!

Note that the model can be VRAM hungry, with parameters scaling by #covariates x #genes x #clusters|#modules... if you run out of vram, you might need to 1. fix a
GPU memory leak, 2. use fewer genes/latent dimensions/clusters/particles, 3. get a bigger GPU

## Coming hopefully soon
- ~~.Improved plotting functionality~~.
- Expanded tutorials
- PyPI release
- ~~.Gene expression histogram normalization~~.
- Phylogeny regression

## Unknown timeline challenges to take on
- Parameter variance estimation
- Improved clustering
- Full ancestral reconstruction

