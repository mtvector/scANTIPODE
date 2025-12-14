<h1 style="text-align: center;"> scANTIPODE</h1>
<a href="https://en.wikipedia.org/wiki/Antipodes"> 
<div align="center">
  <img src="assets/antipode_logo_alternate.png" alt="antipode logo!" width="200">
</div>
</a>

**S**ingle **C**ell **A**ncestral **N**ode **T**axonomy **I**nference by **P**artitioning **O**f **D**ifferential **E**xpression. The model is an extension of the SCVI paradigm--a structured generative, variational inference model developed for the simultaneous analysis (DE) and categorization (taxonomy generation) of cell types across evolution (or now any covariate) using single-cell RNA-seq data. Long ago it began as a hack of a simplified model of [scANVI](https://pyro.ai/examples/scanvi.html) and is built on the pytorch-based PPL [pyro](https://pyro.ai/). The model acts as an integration method, that learns interpretable differential expression in the process. Note that this means ANTIPODE will fail to integrate datasets of different datasets, or datasets with large disparities in quality or gene mean dispersions.

The complete procedure runs in 3 phases (but can also run fully supervised using only phase 2):

1. **The Fuzzy Phase:** Cells may belong to multiple types sampled from a bernoulli distribution, learns an integrated latent space with covariate effects, but is less straightforward to interpret.

2. **The Supervised Phase:** Discrete clustering is initialized from a supervised initialization (or defaults to a de novo k-means clustering in the latent space). Can take a supervised clustering and/or latent space for cells.

3. **The Free Phase:** All parameters are released for unconstrained learning.


You can read about the generative model in the publication and supplement [here](https://www.biorxiv.org/content/10.1101/2025.10.18.683238v2). You can look at example runs [here](examples/outputs/).


## Installation
First create a conda environment with python >= 3.10, then
```
To install the package from PyPI:
    pip install antipode


For development, clone the repository and run:
    conda create -n antipode -c rapidsai -c conda-forge -c nvidia  \
        rapids=24.12 python=3.12 'cuda-version>=12.0,<=12.5' \
        'pytorch=*=*cuda*'
    git clone https://github.com/mtvector/scANTIPODE.git
    cd scANTIPODE
    conda install jax jaxlib -c conda-forge
    pip install -e .

Optional phylogeny/analysis features:
    pip install antipode[extras]
```




Please reach out to let me know if you try ANTIPODE on a dataset and it works (or doesn't work)...

Note that the model can be VRAM hungry, with parameters scaling by #covariates x #genes x #clusters|#modules... if you run out of vram, you might need to 1. fix a
GPU memory leak, 2. use fewer genes/latent dimensions/clusters/particles, 3. get a bigger GPU

## Future development
- Phylogeny regression
- Parameter variance estimation
- Improved clustering
