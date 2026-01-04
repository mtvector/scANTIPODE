import numpy as np
import pandas as pd
import anndata

from antipode.antipode_model import ANTIPODE
from antipode.model_functions import numpy_hardmax


def _make_adata(n_obs=4, n_genes=3):
    x = np.arange(n_obs * n_genes, dtype=np.float32).reshape(n_obs, n_genes)
    adata = anndata.AnnData(x)
    adata.layers["counts"] = x.copy()
    adata.obs["discov"] = pd.Categorical(["a", "b", "a", "b"])
    adata.obs["batch"] = pd.Categorical(["b1", "b1", "b2", "b2"])
    adata.obs["level_1"] = pd.Categorical(["0", "1", "0", "1"])
    return adata


def _init_model(adata, discov_pair, batch_pair=("obs", "batch")):
    return ANTIPODE(
        adata,
        discov_pair,
        batch_pair,
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


def test_weighted_discov_leaf_aggregate_matches_onehot():
    adata = _make_adata()
    discov = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )
    adata.obsm["phylo_cov"] = discov
    adata.uns["phylo_cov_labels"] = ["d0", "d1"]
    adata.obsm["batch"] = np.eye(2, dtype=np.float32)[adata.obs["batch"].cat.codes.to_numpy()]

    model = _init_model(adata, ("obsm", "phylo_cov"), batch_pair=("obsm", "batch"))
    means, orders, weights = model._weighted_discov_leaf_aggregate(
        "level_1",
        layer="counts",
        normalize=False,
    )

    # Manual weighted mean per discov x leaf
    leaf_codes = adata.obs["level_1"].cat.codes.to_numpy()
    leaf_onehot = np.eye(2)[leaf_codes]
    sum_w = np.einsum("bd,bl->dl", discov, leaf_onehot)
    sum_x = np.einsum("bd,bl,bg->dlg", discov, leaf_onehot, adata.layers["counts"])
    manual = sum_x / np.maximum(sum_w[..., None], 1e-12)

    assert orders["level_1"] == ["0", "1"]
    assert np.allclose(weights, sum_w)
    assert np.allclose(means, manual)


def test_calculate_cluster_params_reconstruction_numpy():
    adata = _make_adata()
    model = _init_model(adata, ("obs", "discov"))

    num_discov = 2
    num_var = adata.shape[1]
    num_latent = 2
    sum_levels = sum(model.level_sizes)

    rng = np.random.default_rng(0)
    edges_0 = rng.normal(size=(model.level_sizes[-1], model.level_sizes[0]))
    locs = rng.normal(size=(sum_levels, num_latent))
    cluster_intercept = rng.normal(size=(sum_levels, num_var))
    z_decoder_weight = rng.normal(size=(num_latent, num_var))
    discov_dc = rng.normal(size=(num_discov, num_latent, num_var))
    discov_di = rng.normal(size=(num_discov, sum_levels, num_var))
    discov_dm = rng.normal(size=(num_discov, sum_levels, num_latent))
    discov_da = rng.normal(size=(num_discov, num_var))
    softmax_shift = rng.normal(size=(num_var,))

    adata.uns["param_store"] = {
        "edges_0": edges_0,
        "locs": locs,
        "cluster_intercept": cluster_intercept,
        "z_decoder_weight": z_decoder_weight,
        "discov_dc": discov_dc,
        "discov_di": discov_di,
        "discov_dm": discov_dm,
        "discov_da": discov_da,
        "softmax_shift": softmax_shift,
    }

    out = model.calculate_cluster_params(
        flavor="numpy",
        cluster_count_threshold=0,
    )
    (
        discov_cluster_params,
        cluster_params,
        cluster_labels,
        var_labels,
        discov_da_out,
        cluster_weights,
        (prop_taxon, prop_locs, prop_discov_di, prop_discov_dm),
    ) = out

    level_edges = [numpy_hardmax(edges_0, axis=-1)]
    eye_bottom = np.eye(model.level_sizes[-1])
    parent = eye_bottom @ level_edges[0]
    prop_taxon_manual = np.concatenate([parent, eye_bottom], axis=-1)

    prop_locs_manual = prop_taxon_manual @ locs
    prop_cluster_intercept = prop_taxon_manual @ cluster_intercept
    prop_discov_di_manual = np.einsum("pc,dcg->dpg", prop_taxon_manual, discov_di)
    prop_discov_dm_manual = np.einsum("pc,dcm->dpm", prop_taxon_manual, discov_dm)

    manual_cluster_params = (
        prop_locs_manual @ z_decoder_weight
        + prop_cluster_intercept
        + np.mean(discov_da, 0, keepdims=True)
    )

    manual_discov_cluster_params = (
        np.einsum(
            "dpm,dmg->dpg",
            prop_locs_manual + prop_discov_dm_manual,
            z_decoder_weight + discov_dc,
        )
        + (
            prop_cluster_intercept
            + prop_discov_di_manual
            + discov_da[:, None, :]
        )
        - softmax_shift
    )

    assert list(cluster_labels) == ["0", "1"]
    assert list(var_labels) == list(adata.var.index)
    assert np.allclose(cluster_weights, 1.0)
    assert np.allclose(discov_da_out, discov_da)
    assert np.allclose(prop_taxon, prop_taxon_manual)
    assert np.allclose(prop_locs, prop_locs_manual)
    assert np.allclose(prop_discov_di, prop_discov_di_manual)
    assert np.allclose(prop_discov_dm, prop_discov_dm_manual)
    assert np.allclose(cluster_params, manual_cluster_params)
    assert np.allclose(discov_cluster_params, manual_discov_cluster_params)
