import os
import random
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from torch.backends import cudnn
from scipy.sparse.linalg import eigsh
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix


def fix_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def preprocess_adata(adata, n_top_genes=2000, counts_layer=None):
    """
    Basic ST preprocessing:
    1. HVG selection
    2. library-size normalization
    3. log1p
    4. scaling

    Notes:
    - For best practice, give raw counts via counts_layer
      (e.g. counts_layer="counts") or make sure adata.X is raw counts.
    """
    adata = adata.copy()

    if counts_layer is not None:
        if counts_layer not in adata.layers:
            raise ValueError(f"counts_layer='{counts_layer}' not found in adata.layers")
        adata.X = adata.layers[counts_layer].copy()

    if "highly_variable" not in adata.var.columns:
        sc.pp.highly_variable_genes(
            adata,
            flavor="seurat_v3",
            n_top_genes=n_top_genes
        )

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=True, max_value=10)
    return adata


def to_dense(x):
    if isinstance(x, (csc_matrix, csr_matrix)):
        return x.toarray()
    if sp.issparse(x):
        return x.A
    return np.asarray(x)


def get_expr_feature(adata, use_hvg=True):
    if use_hvg and "highly_variable" in adata.var.columns:
        adata_use = adata[:, adata.var["highly_variable"]]
    else:
        adata_use = adata

    feat = to_dense(adata_use.X).astype(np.float32)
    return feat


def get_pca_feature(adata, n_pcs=50, use_hvg=True, random_state=42):
    x = get_expr_feature(adata, use_hvg=use_hvg)
    n_pcs = min(n_pcs, x.shape[0] - 1, x.shape[1])
    n_pcs = max(2, n_pcs)
    pca = PCA(n_components=n_pcs, random_state=random_state)
    x_pca = pca.fit_transform(x).astype(np.float32)
    return x_pca, pca


def l2_normalize_rows(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norm + eps)


def normalize_adj_dense(adj):
    adj = np.asarray(adj, dtype=np.float32)
    rowsum = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(np.clip(rowsum, 1e-8, None), -0.5)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt
    return adj_norm.astype(np.float32)


def preprocess_adj_dense(adj, add_self_loop=True):
    adj = np.asarray(adj, dtype=np.float32)
    if add_self_loop:
        adj = adj + np.eye(adj.shape[0], dtype=np.float32)
    return normalize_adj_dense(adj)


def compute_normalized_laplacian(adj):
    adj = np.asarray(adj, dtype=np.float32)
    deg = np.sum(adj, axis=1)
    deg_inv_sqrt = np.power(np.clip(deg, 1e-8, None), -0.5)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    I = np.eye(adj.shape[0], dtype=np.float32)
    L = I - D_inv_sqrt @ adj @ D_inv_sqrt
    return L.astype(np.float32)


def build_spatial_knn_graph(coords, n_neighbors=6):
    """
    Build spatial KNN graph only from coordinates.
    For Visium-like data, n_neighbors=6 or 8 is usually reasonable.
    """
    coords = np.asarray(coords, dtype=np.float32)
    n = coords.shape[0]
    nn_k = min(n_neighbors + 1, n)

    nbrs = NearestNeighbors(n_neighbors=nn_k, metric="euclidean").fit(coords)
    dists, indices = nbrs.kneighbors(coords)

    adj = np.zeros((n, n), dtype=np.float32)

    if nn_k > 1:
        sigma = np.median(dists[:, 1:]) + 1e-8
    else:
        sigma = 1.0

    for i in range(n):
        for k in range(1, nn_k):
            j = indices[i, k]
            dist = dists[i, k]
            w = np.exp(-(dist ** 2) / (2 * sigma ** 2))
            adj[i, j] = w

    adj = np.maximum(adj, adj.T)
    return adj.astype(np.float32)


def reweight_spatial_graph(
    adj_spatial,
    expr_feat,
    morph_feat=None,
    expr_power=0.5,
    morph_power=1.0,
    sim_floor=0.05
):
    """
    Expression / morphology only reweight existing spatial edges,
    instead of creating long-range shortcut edges.
    """
    adj_spatial = np.asarray(adj_spatial, dtype=np.float32)
    n = adj_spatial.shape[0]

    src, dst = np.where(np.triu(adj_spatial > 0, k=1))
    if len(src) == 0:
        return adj_spatial.copy().astype(np.float32)

    expr_norm = l2_normalize_rows(expr_feat)
    sim_expr = np.sum(expr_norm[src] * expr_norm[dst], axis=1)
    sim_expr = np.clip((sim_expr + 1.0) / 2.0, sim_floor, 1.0)

    weight = adj_spatial[src, dst] * (sim_expr ** expr_power)

    if morph_feat is not None:
        morph_norm = l2_normalize_rows(morph_feat)
        sim_morph = np.sum(morph_norm[src] * morph_norm[dst], axis=1)
        sim_morph = np.clip((sim_morph + 1.0) / 2.0, sim_floor, 1.0)
        weight = weight * (sim_morph ** morph_power)

    adj_wave = np.zeros((n, n), dtype=np.float32)
    adj_wave[src, dst] = weight
    adj_wave[dst, src] = weight

    if adj_wave.max() > 0:
        adj_wave = adj_wave / (adj_wave.max() + 1e-8)

    return adj_wave.astype(np.float32)


def construct_st_graphs(
    adata,
    expr_feat_for_graph,
    n_neighbors_spatial=6,
    use_morphology=False,
    morph_key="image_feat",
    expr_power=1.0,
    morph_power=1.0
):
    """
    Returns:
        adj_spatial: local graph for local encoder / smoothness / refinement
        adj_wave:    reweighted spatial graph for global wave propagation
    """
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    adj_spatial = build_spatial_knn_graph(coords, n_neighbors=n_neighbors_spatial)

    if use_morphology and morph_key in adata.obsm:
        morph_feat = np.asarray(adata.obsm[morph_key], dtype=np.float32)
    else:
        morph_feat = None

    adj_wave = reweight_spatial_graph(
        adj_spatial=adj_spatial,
        expr_feat=expr_feat_for_graph,
        morph_feat=morph_feat,
        expr_power=expr_power,
        morph_power=morph_power
    )

    return {
        "adj_spatial": adj_spatial.astype(np.float32),
        "adj_wave": adj_wave.astype(np.float32)
    }


def spectral_decompose_laplacian_hybrid(adj, k=128, low_ratio=0.75):
    """
    Hybrid spectral decomposition:
    - low-frequency eigenpairs for global tissue structure
    - high-frequency eigenpairs for boundaries / local sharp changes
    """
    L = compute_normalized_laplacian(adj)
    n = L.shape[0]

    if n <= 3:
        eigvals, eigvecs = np.linalg.eigh(L)
        return eigvals.astype(np.float32), eigvecs.astype(np.float32)

    k = min(k, n - 1)
    k = max(2, k)

    k_low = max(1, int(round(k * low_ratio)))
    k_high = max(0, k - k_low)

    if n <= 512 or k >= n - 1:
        eigvals_all, eigvecs_all = np.linalg.eigh(L)
        idx_low = np.arange(k_low)
        if k_high > 0:
            idx_high = np.arange(n - k_high, n)
            idx = np.concatenate([idx_low, idx_high])
        else:
            idx = idx_low

        idx = np.unique(idx)
        eigvals = eigvals_all[idx]
        eigvecs = eigvecs_all[:, idx]
    else:
        L_sparse = sp.csr_matrix(L)

        eigvals_low, eigvecs_low = eigsh(L_sparse, k=k_low, which="SM")
        eigvals = eigvals_low
        eigvecs = eigvecs_low

        if k_high > 0:
            eigvals_high, eigvecs_high = eigsh(L_sparse, k=k_high, which="LA")
            eigvals = np.concatenate([eigvals_low, eigvals_high], axis=0)
            eigvecs = np.concatenate([eigvecs_low, eigvecs_high], axis=1)

    order = np.argsort(eigvals)
    eigvals = np.real(eigvals[order]).astype(np.float32)
    eigvecs = np.real(eigvecs[:, order]).astype(np.float32)

    return eigvals, eigvecs


def feature_masking(x, mask_rate=0.15):
    x = np.asarray(x, dtype=np.float32)
    mask = np.random.binomial(1, 1 - mask_rate, size=x.shape).astype(np.float32)
    return (x * mask).astype(np.float32)


def feature_noise(x, noise_std=0.1):
    x = np.asarray(x, dtype=np.float32)
    noise = np.random.normal(0.0, noise_std, size=x.shape).astype(np.float32)
    return (x + noise).astype(np.float32)


def build_two_feature_views(x, mask_rate=0.15, noise_std=0.1):
    x1 = feature_noise(feature_masking(x, mask_rate=mask_rate), noise_std=noise_std)
    x2 = feature_noise(feature_masking(x, mask_rate=mask_rate), noise_std=noise_std)
    return x1.astype(np.float32), x2.astype(np.float32)