<h1 style="text-align: center;"> scANTIPODE</h1>

![antipode logo!](assets/antipode_logo_alternate.png)

**S**ingle **C**ell **A**ncestral **N**ode **T**axonomy **I**nference by **P**arcellation **O**f **D**ifferential **E**xpression. The model is an extension of the SCVI paradigm--a structured generative, variational inference model developed for the simultaneous analysis (DE) and categorization (taxonomy generation) of cell types across evolution (or now any covariate) using single-cell RNA-seq data. It was originally developed from a simplified model of scANVI and is built on the pytorch-based PPL [pyro](https://pyro.ai/).

The model runs in 3 phases.


You can read about the generative model [here](https://www.overleaf.com/read/nmcmcjtvmfcb#acf7a4). You can look at example runs [here](examples/outputs/).


## Installation

```
git clone git@github.com:mtvector/scANTIPODE.git
#cuda 11.7 should work too
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install jax jaxlib -c conda-forge
cd scANTIPODE
pip install -e .
```

Please reach out to let me know if you try ANTIPODE on a dataset and it works (or doesn't work)... The model is (forever) a work in process!

Note that the model can be VRAM hungry, with parameters scaling by #covariates x #genes x #clusters|#modules... if you run out of vram, you might need to 1. fix a
GPU memory leak, 2. use fewer genes/latent dimensions/cluster, 3. get a bigger GPU

## Coming soon
- Improved plotting functionality
- Expanded tutorials
- PyPI release
- Gene expression histogram normalization
- Phylogeny regression


## Next challenges
- Parameter variance estimation
- Improved clustering
