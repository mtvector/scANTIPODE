try:
    from Bio import Phylo
except:
    print('no phylo')

from io import StringIO
import numpy as np
import torch
from .model_functions import *

def build_phylogeny_matrices(newick_str):
    """
    Parse Newick, return
      • leaf_names:   list[str]                 (the species)
      • node_names:   list[str]                 ([leaf_names] + internal clade names)
      • membership:   ndarray[n_leaves,n_nodes] binary 1 if leaf_i under node_j
      • weights:      ndarray[n_nodes]          = 1 for leaves, (#desc / total_leaves) for internals
    """
    # --- parse tree & collect leaves ---
    tree = Phylo.read(StringIO(newick_str), "newick")
    leaves = [t.name for t in tree.get_terminals()]
    n_leaves = len(leaves)

    # --- collect non-root internals and name them by their leaf list (kinda annoying i know) ---
    internals = [nd for nd in tree.get_nonterminals() if nd is not tree.root]
    internal_names = []
    internal_M_cols = []
    for nd in internals:
        desc = sorted([lf.name for lf in nd.get_terminals()])
        # join by underscore
        label = "_".join(desc)
        internal_names.append(label)
        # build one column for these descendants
        col = [1 if leaf in desc else 0 for leaf in leaves]
        internal_M_cols.append(col)

    # --- stack into arrays ---
    # identity for the leaves
    I = np.eye(n_leaves, dtype=int)
    # membership matrix for internals: shape (n_leaves, n_internal)
    M_int = np.array(internal_M_cols).T
    # full membership: leaves then internals
    M = np.concatenate([I, M_int], axis=1)

    # weights: 1 for each leaf, then (#desc / total_leaves) for internals
    # leaf_weights = np.ones(n_leaves, dtype=float)
    # internal_weights = M_int.sum(axis=0).astype(float) / n_leaves
    # weights = np.concatenate([leaf_weights, internal_weights], axis=0)
    weights = M.sum(0) / len(leaves)

    # node names: leaves first, then internal labels
    node_names = leaves + internal_names

    return leaves, node_names, M, weights

def add_node_obsm(
    adata,
    discov_key: str,
    newick_str: str,
    obsm_key: str = "node_binary",
):
    """
    Reads adata.obs[discov_key] as species names,
    builds membership & weights from Newick,
    and writes adata.obsm[obsm_key] = [n_obs, n_nodes] binary matrix.
    Returns (node_names, weights, membership).
    """
    leaves, nodes, M, weights = build_phylogeny_matrices(newick_str)
    # map each obs to leaf index
    adata_cats = set(adata.obs[discov_key].astype('category').cat.categories)
    if adata_cats != set(leaves):
        raise Exception("discov keys not identical to tree leaves")
    obs_leaves = adata.obs[discov_key].astype(str).values
    leaf2idx = {l:i for i,l in enumerate(leaves)}
    onehot = np.zeros((adata.n_obs, len(leaves)), dtype=int)
    for i, sp in enumerate(obs_leaves):
        onehot[i, leaf2idx[sp]] = 1
    # obs × nodes
    obsm = onehot.dot(M)
    adata.obsm[obsm_key] = obsm
    return leaves, nodes, weights, M

def node_params_to_leaf_params(node_params: torch.Tensor, membership: np.ndarray):
    """
    node_params: [n_nodes, ...] tensor
    membership:   [n_leaves, n_nodes] numpy array
    returns:      [n_leaves, ...] tensor
    """
    M = torch.from_numpy(membership.astype(float)).to(node_params.device)
    # each leaf gets sum of its nodes’ params
    return M @ node_params


def compute_descendant_fraction(membership: np.ndarray) -> np.ndarray:
    """
    Compute the fraction of leaves under each node.

    Parameters
    ----------
    membership
        Binary membership matrix shaped (n_leaves, n_nodes), where membership[i, j] == 1
        if leaf i is a descendant of node j.

    Returns
    -------
    np.ndarray
        Fraction of leaves under each node, shaped (n_nodes,).
    """
    membership = np.asarray(membership)
    return membership.sum(axis=0) / membership.shape[0]


def filter_nodes_by_coverage(nodes, membership, max_frac=0.99):
    """
    Drop internal nodes that cover too many leaves (quasi-intercept clades).

    Assumes the first n_leaves columns are leaves (as in build_phylogeny_matrices()).
    Leaves are always retained.
    """
    n_leaves = membership.shape[0]
    frac = compute_descendant_fraction(membership)
    keep = np.ones(membership.shape[1], dtype=bool)

    keep[:n_leaves] = True
    keep[n_leaves:] = frac[n_leaves:] <= max_frac

    nodes_keep = [n for k, n in zip(keep, nodes) if k]
    membership_keep = membership[:, keep]
    return nodes_keep, membership_keep, keep


def build_simplex_phylo_cov(
    adata,
    discov_key: str,
    leaves: list[str],
    nodes: list[str],
    membership: np.ndarray,
    gamma: float = 0.5,
    leaf_weight: float = 1.0,
    internal_weight: float = 1.0,
    eps: float = 1e-8,
):
    """
    Build a simplex-valued phylogeny covariate per observation.

    Parameters
    ----------
    adata
        AnnData with a discovery category column.
    discov_key
        Column in adata.obs containing discovery categories aligned to leaves.
    leaves
        Leaf names, ordered as in membership.
    nodes
        Node names (unused; kept for API symmetry).
    membership
        Binary membership matrix shaped (n_leaves, n_nodes).
    gamma
        Decay per step toward root (0 < gamma < 1).
    leaf_weight
        Weight for the leaf entry.
    internal_weight
        Base weight for internal nodes before decay.
    eps
        Small constant to guard division by zero.

    Returns
    -------
    np.ndarray
        Simplex matrix shaped (n_obs, n_nodes) where each row sums to 1.
    """
    sp = adata.obs[discov_key].astype("category")
    sp = sp.cat.reorder_categories(leaves, ordered=True)
    codes = sp.cat.codes.to_numpy()

    onehot = np.zeros((adata.n_obs, len(leaves)), dtype=np.float32)
    onehot[np.arange(adata.n_obs), codes] = 1.0

    x_bin = onehot @ membership.astype(np.float32)
    clade_size = membership.sum(axis=0).astype(np.float32)

    x = np.zeros_like(x_bin, dtype=np.float32)
    for i in range(adata.n_obs):
        leaf_idx = codes[i]
        active = np.where(x_bin[i] > 0)[0]
        internals = active[active != leaf_idx]

        if internals.size > 0:
            order = internals[np.argsort(clade_size[internals])]
            for k, j in enumerate(order):
                x[i, j] = internal_weight * (gamma ** k)

        x[i, leaf_idx] = leaf_weight
        s = x[i].sum()
        if s < eps:
            x[i, leaf_idx] = 1.0
            s = 1.0
        x[i] /= s

    return x


def build_discov_da_prior_with_internal_means(
    leaf_prior: np.ndarray,
    membership: np.ndarray,
    keep_mask: np.ndarray | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Build discovery-level DA priors for internal nodes using descendant means.

    Parameters
    ----------
    leaf_prior
        Leaf prior matrix shaped (n_leaves, n_var), aligned to leaves.
    membership
        Binary membership matrix shaped (n_leaves, n_nodes).
    keep_mask
        Optional boolean mask to subset nodes after aggregation.
    eps
        Small constant to avoid division by zero.

    Returns
    -------
    np.ndarray
        Node prior matrix shaped (n_nodes_kept, n_var).
    """
    leaf_prior = np.asarray(leaf_prior, dtype=np.float32)
    membership = np.asarray(membership, dtype=np.float32)

    desc_counts = membership.sum(axis=0, keepdims=True).T
    desc_counts = np.maximum(desc_counts, eps)
    node_prior = (membership.T @ leaf_prior) / desc_counts

    if keep_mask is not None:
        node_prior = node_prior[keep_mask]

    return node_prior.astype(np.float32)


def build_dcd_prior_with_internal_means(
    leaf_prior: np.ndarray,
    membership: np.ndarray,
    keep_mask: np.ndarray | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Backwards-compatible alias for build_discov_da_prior_with_internal_means().
    """
    return build_discov_da_prior_with_internal_means(
        leaf_prior=leaf_prior,
        membership=membership,
        keep_mask=keep_mask,
        eps=eps,
    )


def setup_phylo_covariates(
    adata,
    discov_key: str,
    newick_str: str,
    leaf_prior: np.ndarray,
    obsm_key: str = "phylo_cov",
    max_frac: float = 0.85,
    gamma: float = 0.5,
    leaf_weight: float = 1.0,
    internal_weight: float = 1.0,
    eps: float = 1e-8,
):
    """
    Convenience helper to build phylogeny covariates and discovery DA init values.

    This will:
      1) Parse the Newick tree and build leaf/internal membership.
      2) Filter internal nodes with large coverage.
      3) Build a simplex-valued covariate matrix stored in adata.obsm[obsm_key].
      4) Aggregate leaf priors into internal node init values aligned to filtered nodes.

    Parameters
    ----------
    adata
        AnnData with discovery categories in adata.obs[discov_key].
    discov_key
        Column in adata.obs containing discovery categories aligned to tree leaves.
    newick_str
        Newick string defining the phylogeny.
    leaf_prior
        Leaf init matrix shaped (n_leaves, n_var), aligned to leaf order.
    obsm_key
        Key to store the simplex covariate in adata.obsm.
    max_frac
        Maximum fraction of leaves allowed for internal nodes.
    gamma
        Decay per step toward root (0 < gamma < 1).
    leaf_weight
        Weight for the leaf entry in the simplex.
    internal_weight
        Base weight for internal nodes before decay.
    eps
        Small constant to avoid division by zero.

    Returns
    -------
    tuple
        (discov_da_init, leaves, nodes_f, membership_f, keep_mask)
    """
    leaves, nodes, weights, membership = add_node_obsm(
        adata,
        discov_key,
        newick_str,
        obsm_key=obsm_key,
    )
    adata.uns["phylo_nodes"] = nodes
    adata.uns["phylo_weights"] = weights
    adata.uns["phylo_M"] = membership
    adata.obs[discov_key] = adata.obs[discov_key].cat.reorder_categories(
        [x for x in nodes if x in adata.obs[discov_key].cat.categories]
    )

    nodes_f, membership_f, keep_mask = filter_nodes_by_coverage(
        nodes,
        membership,
        max_frac=max_frac,
    )
    x_phylo = build_simplex_phylo_cov(
        adata,
        discov_key=discov_key,
        leaves=leaves,
        nodes=nodes_f,
        membership=membership_f,
        gamma=gamma,
        leaf_weight=leaf_weight,
        internal_weight=internal_weight,
        eps=eps,
    )
    adata.obsm[obsm_key] = x_phylo
    adata.uns[f"{obsm_key}_labels"] = nodes_f

    discov_da_init = build_discov_da_prior_with_internal_means(
        leaf_prior=leaf_prior,
        membership=membership,
        keep_mask=keep_mask,
        eps=eps,
    )
    return discov_da_init, leaves, nodes_f, membership_f, keep_mask
