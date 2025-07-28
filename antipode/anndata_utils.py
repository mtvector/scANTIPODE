# Author: mtvector a.k.a. Matthew Schmitz
import h5py
import anndata
from anndata import AnnData
from pathlib import Path
from typing import Union, Literal, List, Dict
from ete3 import Tree
from scipy.sparse import csr_matrix
import pandas as pd

def h5_tree(val):
    tree = {}
    for key, item in val.items():
        if isinstance(item, h5py._hl.group.Group):
            tree[key] = h5_tree(item)
        else:
            try:
                tree[key] = len(item)
            except TypeError:
                tree[key] = "scalar"
    return tree


def dict_to_ete3_tree(d, parent=None):
    if parent is None:
        parent = Tree(name="root")
    for k, v in d.items():
        child = parent.add_child(name=k)
        if isinstance(v, dict):
            dict_to_ete3_tree(v, child)
    return parent


def ete3_tree_to_dict(tree):
    def helper(node):
        if node.is_leaf():
            return node.name
        d = {}
        for child in node.get_children():
            d[child.name] = helper(child)
        return d

    root_dict = {}
    for child in tree.get_children():
        root_dict[child.name] = helper(child)
    return root_dict


def prune_tree(tree, keys):
    t = dict_to_ete3_tree(tree)

    nodes_to_keep = set()

    # Find all nodes matching the keys and collect their ancestors and descendants
    for key in keys:
        for node in t.search_nodes(name=key):
            nodes_to_keep.update(node.iter_ancestors())
            nodes_to_keep.update(node.iter_descendants())
            nodes_to_keep.add(node)

    # Prune the original tree by removing nodes that are not in nodes_to_keep
    for node in t.traverse("postorder"):
        if node not in nodes_to_keep and node.up:
            node.detach()

    pruned_dict = ete3_tree_to_dict(t)
    return pruned_dict


def read_h5_to_dict(h5_group, pruned_tree):
    def helper(group, subtree):
        result = {}
        for key, value in subtree.items():
            if isinstance(value, dict):
                # Ensure key exists in the group before trying to access it
                if key in group:
                    result[key] = helper(group[key], value)
                else:
                    result[key] = None  # Handle missing keys gracefully
            else:
                # Ensure key exists in the group before trying to access it
                if key in group:
                    if isinstance(group[key], h5py.Dataset):
                        if group[key].shape == ():
                            result[key] = group[key][()]  # Read scalar dataset
                        else:
                            data = group[key][:]
                            # Decode binary strings to regular strings if necessary
                            if data.dtype.kind == "S":
                                data = data.astype(str)
                            result[key] = data  # Read non-scalar dataset
                    else:
                        result[key] = None  # Handle non-dataset values gracefully
                else:
                    result[key] = None  # Handle missing keys gracefully
        return result

    return helper(h5_group, pruned_tree)


def convert_to_dataframe(data):
    df_dict = {}
    for key, value in data.items():
        if isinstance(value, dict) and "categories" in value and "codes" in value:
            categories = [
                cat.decode("utf-8") if isinstance(cat, bytes) else cat
                for cat in value["categories"]
            ]
            codes = value["codes"]
            df_dict[key] = pd.Categorical.from_codes(codes, categories)
        elif (
            isinstance(value, dict)
            and "data" in value
            and "indices" in value
            and "indptr" in value
        ):
            df_dict[key] = csr_matrix(
                (value["data"], value["indices"], value["indptr"])
            )
        else:
            if value.dtype.kind == "S":
                value = [
                    v.decode("utf-8") if isinstance(v, bytes) else v for v in value
                ]
            df_dict[key] = value
    df = pd.DataFrame(df_dict)
    return df


def handle_special_keys(data):
    if "obs" in data:
        data["obs"] = convert_to_dataframe(data["obs"])
    if "var" in data:
        data["var"] = convert_to_dataframe(data["var"])
    if ("layers" in data) or (data == "X"):
        try:
            for layer in data["layers"]:
                data["layers"][layer] = csr_matrix(
                    (
                        data["layers"][layer]["data"],
                        data["layers"][layer]["indices"],
                        data["layers"][layer]["indptr"],
                    )
                )
        except:
            pass
    return data


def read_h5ad_backed_selective(
    filename: Union[str, Path],
    mode: Literal["r", "r+"],
    selected_keys: List[str] = [],
    return_dict: bool = False,
) -> AnnData:
    f = h5py.File(filename, mode)

    selected_keys += ["_index"]

    tree = h5_tree(f)
    selected_tree = prune_tree(tree, selected_keys)
    with f:
        d = read_h5_to_dict(f, selected_tree)
        d = handle_special_keys(d)
    d["filename"] = filename
    d["filemode"] = mode
    if return_dict:
        return d
    else:
        adata = AnnData(**d)
        if "_index" in adata.obs.columns:
            index_series = adata.obs["_index"].astype('string')
            adata.obs.index = list(index_series)
            adata.obs.drop("_index",axis=1, inplace=True)
        if "_index" in adata.var.columns:
            index_series = adata.var["_index"].astype('string')
            adata.var.index = list(index_series)
            adata.var.drop("_index",axis=1, inplace=True)
        return adata