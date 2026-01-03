import numpy as np
import pandas as pd
import torch
import pyro
import pyro.poutine as poutine
import anndata

from antipode.model_modules import MAPLaplaceModule
from antipode.antipode_model import ANTIPODE


class DummyModel:
    prior_scale = 2.0


def test_map_laplace_prior_loc():
    pyro.clear_param_store()
    prior_loc = torch.tensor([[1.0, -2.0], [0.5, 0.0]])
    module = MAPLaplaceModule(
        DummyModel(),
        name="test",
        param_shape=list(prior_loc.shape),
        prior_loc=prior_loc,
    )
    trace = poutine.trace(module.model_sample).get_trace(s=torch.ones(1))
    dist = trace.nodes["test_sample"]["fn"]
    assert torch.allclose(dist.loc, prior_loc)


def _make_minimal_adata():
    x = np.ones((4, 3), dtype=np.float32)
    adata = anndata.AnnData(x)
    adata.layers["counts"] = x.copy()
    adata.obs["discov"] = pd.Categorical(["a", "b", "a", "b"])
    adata.obs["batch"] = pd.Categorical(["b1", "b1", "b2", "b2"])
    return adata


def test_discov_da_and_intercept_params():
    pyro.clear_param_store()
    adata = _make_minimal_adata()
    model = ANTIPODE(
        adata,
        ("obs", "discov"),
        ("obs", "batch"),
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
    discov_ind = torch.tensor(adata.obs["discov"].cat.codes.values).unsqueeze(-1)
    batch_ind = torch.tensor(adata.obs["batch"].cat.codes.values).unsqueeze(-1)
    seccov = torch.zeros((s.shape[0], 1))
    model.guide(s, discov_ind=discov_ind, batch_ind=batch_ind, seccov=seccov)
    model.model(s, discov_ind=discov_ind, batch_ind=batch_ind, seccov=seccov)

    pstore = pyro.get_param_store()
    assert "discov_da" in pstore
    assert tuple(pstore["discov_da"].shape) == (model.num_discov, model.num_var)
    assert "discov_constitutive_intercept" in pstore
    assert tuple(pstore["discov_constitutive_intercept"].shape) == (model.num_var,)
