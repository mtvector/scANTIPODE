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
