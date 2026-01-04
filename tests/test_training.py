import numpy as np
import pandas as pd
import anndata
import pyro
import torch
from pyro import poutine

from antipode.antipode_model import ANTIPODE
from antipode.model_modules import SafeSVI


def _make_minimal_adata(n_obs=4, n_genes=3):
    x = np.ones((n_obs, n_genes), dtype=np.float32)
    adata = anndata.AnnData(x)
    adata.layers["counts"] = x.copy()
    adata.obs["discov"] = pd.Categorical(["a", "b", "a", "b"])
    adata.obs["batch"] = pd.Categorical(["b1", "b1", "b2", "b2"])
    return adata


def test_train_phase_runs_on_minimal_data():
    pyro.clear_param_store()
    adata = _make_minimal_adata()
    adata.obsm["discov_onehot"] = np.eye(2, dtype=np.float32)[
        adata.obs["discov"].cat.codes.to_numpy()
    ]
    adata.obsm["batch_onehot"] = np.eye(2, dtype=np.float32)[
        adata.obs["batch"].cat.codes.to_numpy()
    ]
    model = ANTIPODE(
        adata,
        ("obsm", "discov_onehot"),
        ("obsm", "batch_onehot"),
        layer="counts",
        level_sizes=[1, 2],
        num_latent=2,
        num_batch_embed=1,
        classifier_hidden=[2],
        encoder_hidden=[2],
        batch_embedder_hidden=[2],
        use_q_score=False,
        use_psi=False,
    )
    model.freeze_encoder = False

    s = torch.tensor(adata.layers["counts"])
    discov_ind = torch.tensor(adata.obsm["discov_onehot"])
    batch_ind = torch.tensor(adata.obsm["batch_onehot"])
    seccov = torch.zeros((s.shape[0], 1))

    optim = pyro.optim.ClippedAdam({"lr": 1e-3})
    elbo = pyro.infer.Trace_ELBO(num_particles=1)
    model_blocked = poutine.block(model.model, hide=["s"])
    svi = SafeSVI(model_blocked, model.guide, optim, elbo)

    loss = svi.step(
        s,
        discov_ind=discov_ind,
        batch_ind=batch_ind,
        seccov=seccov,
        step=torch.ones(1),
    )

    assert np.isfinite(loss)
    assert len(pyro.get_param_store()) > 0
