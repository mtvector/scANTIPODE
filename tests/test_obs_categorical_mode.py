"""Smoke-tests confirming that ANTIPODE runs without shape errors in both
obs-categorical and obsm-design-matrix discovery modes.

Regression guard for the RuntimeError at antipode_model.py caused by
discov_da sampling as [num_discov, num_discov, num_var] instead of
[num_discov, num_var] when the two plates were non-contiguous.
"""
import numpy as np
import pandas as pd
import anndata
import pyro
import torch
from pyro import poutine

from antipode.antipode_model import ANTIPODE
from antipode.model_modules import SafeSVI


def _make_adata(n_obs=64, n_genes=50, n_discov=4, n_batch=3):
    rng = np.random.default_rng(0)
    x = rng.poisson(5, size=(n_obs, n_genes)).astype(np.float32)
    adata = anndata.AnnData(x)
    adata.layers["counts"] = x.copy()
    discov_labels = [f"d{i}" for i in range(n_discov)]
    batch_labels = [f"b{i}" for i in range(n_batch)]
    adata.obs["discov"] = pd.Categorical(
        [discov_labels[i % n_discov] for i in range(n_obs)]
    )
    adata.obs["batch"] = pd.Categorical(
        [batch_labels[i % n_batch] for i in range(n_obs)]
    )
    # obsm one-hot views for design-matrix mode
    adata.obsm["discov_onehot"] = np.eye(n_discov, dtype=np.float32)[
        adata.obs["discov"].cat.codes.to_numpy()
    ]
    adata.obsm["batch_onehot"] = np.eye(n_batch, dtype=np.float32)[
        adata.obs["batch"].cat.codes.to_numpy()
    ]
    return adata


def _discov_real_means(adata, n_discov, n_genes):
    """Simple per-category mean expression used as the prior for discov_da."""
    return torch.ones(n_discov, n_genes)


def _run_steps(model, s, discov_ind, batch_ind, n_steps=3):
    """Run n_steps of SVI and return the last loss."""
    seccov = torch.zeros((s.shape[0], 1))
    optim = pyro.optim.ClippedAdam({"lr": 1e-3})
    elbo = pyro.infer.Trace_ELBO(num_particles=1)
    model_blocked = poutine.block(model.model, hide=["s"])
    svi = SafeSVI(model_blocked, model.guide, optim, elbo)
    loss = None
    for _ in range(n_steps):
        loss = svi.step(
            s,
            discov_ind=discov_ind,
            batch_ind=batch_ind,
            seccov=seccov,
            step=torch.ones(1),
        )
    return loss


def test_obs_categorical_mode_no_shape_error():
    """obs categorical mode: discov_da must sample as [num_discov, num_var]."""
    pyro.clear_param_store()
    n_discov, n_genes = 4, 50
    adata = _make_adata(n_discov=n_discov, n_genes=n_genes)

    model = ANTIPODE(
        adata,
        discov_pair=("obs", "discov"),
        batch_pair=("obs", "batch"),
        layer="counts",
        level_sizes=[1, 3],
        num_latent=4,
        num_batch_embed=2,
        classifier_hidden=[4],
        encoder_hidden=[4],
        batch_embedder_hidden=[4],
        use_q_score=False,
        use_psi=False,
        discov_real_means=_discov_real_means(adata, n_discov, n_genes),
    )
    model.freeze_encoder = False

    # Confirm discov_da plate is at dim=-2 (the fix)
    assert model.discov_da_plate.dim == -2, (
        f"discov_da_plate.dim should be -2, got {model.discov_da_plate.dim}"
    )

    s = torch.tensor(adata.layers["counts"])
    # The scvi CategoricalObsField loader returns shape [N, 1]; reproduce that here
    discov_ind = torch.tensor(
        adata.obs["discov"].cat.codes.to_numpy(), dtype=torch.long
    ).unsqueeze(-1)
    batch_ind = torch.tensor(
        adata.obs["batch"].cat.codes.to_numpy(), dtype=torch.long
    ).unsqueeze(-1)

    loss = _run_steps(model, s, discov_ind, batch_ind)
    assert np.isfinite(loss), f"Expected finite loss in obs mode, got {loss}"


def test_obsm_design_matrix_mode_no_shape_error():
    """obsm design-matrix mode: discov_da must also sample as [num_discov, num_var]."""
    pyro.clear_param_store()
    n_discov, n_genes = 4, 50
    adata = _make_adata(n_discov=n_discov, n_genes=n_genes)

    model = ANTIPODE(
        adata,
        discov_pair=("obsm", "discov_onehot"),
        batch_pair=("obsm", "batch_onehot"),
        layer="counts",
        level_sizes=[1, 3],
        num_latent=4,
        num_batch_embed=2,
        classifier_hidden=[4],
        encoder_hidden=[4],
        batch_embedder_hidden=[4],
        use_q_score=False,
        use_psi=False,
        discov_real_means=_discov_real_means(adata, n_discov, n_genes),
    )
    model.freeze_encoder = False

    s = torch.tensor(adata.layers["counts"])
    discov_ind = torch.tensor(adata.obsm["discov_onehot"])
    batch_ind = torch.tensor(adata.obsm["batch_onehot"])

    loss = _run_steps(model, s, discov_ind, batch_ind)
    assert np.isfinite(loss), f"Expected finite loss in obsm mode, got {loss}"


if __name__ == "__main__":
    test_obs_categorical_mode_no_shape_error()
    print("obs categorical mode: OK")
    test_obsm_design_matrix_mode_no_shape_error()
    print("obsm design-matrix mode: OK")
    print("All checks passed.")
