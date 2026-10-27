"""E19 CSR graph operations. Preserve frozen graph semantics, including ties."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors
import torch


def build_spatial_knn_graph(coords, n_neighbors=6):
    coords = np.asarray(coords, dtype=np.float32)
    n = len(coords)
    if n == 0:
        raise ValueError('Empty coordinates')
    nk = min(n_neighbors + 1, n)
    d, ix = NearestNeighbors(n_neighbors=nk, metric='euclidean').fit(coords).kneighbors(coords)
    sigma = np.median(d[:, 1:]) + 1e-8 if nk > 1 else 1.0
    # Deliberately skip POSITION zero, matching the frozen implementation.
    row = np.repeat(np.arange(n), nk - 1)
    col = ix[:, 1:].ravel()
    distance = d[:, 1:].ravel()
    w = np.exp(-(distance ** 2) / (2 * sigma ** 2)).astype(np.float32)
    a = sp.csr_matrix((w, (row, col)), shape=(n, n))
    return a.maximum(a.T).tocsr()


def reweight_spatial_graph(adj_spatial, expr_feat, morph_feat=None,
                           expr_power=.5, morph_power=1., sim_floor=.05):
    from .preprocess import l2_normalize_rows
    a = sp.csr_matrix(adj_spatial, dtype=np.float32)
    upper = sp.triu(a, k=1, format='coo')
    if upper.nnz == 0:
        return a.copy()
    weight = upper.data.copy()
    for feature, power in [(expr_feat, expr_power), (morph_feat, morph_power)]:
        if feature is None:
            continue
        x = l2_normalize_rows(feature)
        # Chunked edge feature products avoid E x P temporary blowups.
        for start in range(0, upper.nnz, 65536):
            sl = slice(start, start + 65536)
            sim = np.sum(x[upper.row[sl]] * x[upper.col[sl]], axis=1)
            sim = np.clip((sim + 1.) / 2., sim_floor, 1.)
            weight[sl] *= sim ** power
    w = sp.csr_matrix((weight, (upper.row, upper.col)), shape=a.shape)
    w = (w + w.T).tocsr()
    maximum = w.data.max() if w.nnz else 0.
    if maximum > 0:
        w.data /= maximum + 1e-8
    return w.astype(np.float32)


def normalize_adj_dense(adj):
    a = sp.csr_matrix(adj, dtype=np.float32)
    deg = np.asarray(a.sum(axis=1)).ravel()
    inv = np.power(np.clip(deg, 1e-8, None), -.5)
    # Elementwise row/column scaling, no dense diagonal matrix.
    coo = a.tocoo()
    data = (inv[coo.row] * coo.data) * inv[coo.col]
    return sp.csr_matrix((data, (coo.row, coo.col)), shape=a.shape)


def preprocess_adj_dense(adj, add_self_loop=True):
    a = sp.csr_matrix(adj, dtype=np.float32)
    if add_self_loop:
        a = a + sp.eye(a.shape[0], dtype=np.float32, format='csr')
    return normalize_adj_dense(a)


def compute_normalized_laplacian(adj):
    a = sp.csr_matrix(adj, dtype=np.float32)
    return (sp.eye(a.shape[0], dtype=np.float32, format='csr') - normalize_adj_dense(a)).tocsr()


def spectral_decompose_laplacian_hybrid(adj, k=128, low_ratio=.75):
    lap = compute_normalized_laplacian(adj)
    n = lap.shape[0]
    if n <= 3:
        # Bounded <=3x3 compatibility exception, never a large-graph fallback.
        vals, vecs = np.linalg.eigh(lap.toarray())
        return vals.astype(np.float32), vecs.astype(np.float32)
    k = max(2, min(int(k), n - 1))
    if not 0 <= low_ratio <= 1:
        raise ValueError('low_ratio must be in [0,1]')
    kl = max(1, int(round(k * low_ratio)))
    kh = max(0, k - kl)
    v0 = np.random.RandomState(42).normal(size=n).astype(np.float32)
    vals, vecs = eigsh(lap, k=kl, which='SM', v0=v0, tol=1e-7, maxiter=max(1000, 20*n))
    if kh:
        vh, uh = eigsh(lap, k=kh, which='LA', v0=v0, tol=1e-7, maxiter=max(1000, 20*n))
        vals, vecs = np.r_[vals, vh], np.c_[vecs, uh]
    order = np.argsort(vals)
    return vals[order].astype(np.float32), vecs[:, order].astype(np.float32)


def construct_st_graphs(adata, expr_feat_for_graph, n_neighbors_spatial=6,
                        use_morphology=False, morph_key='image_feat',
                        expr_power=1., morph_power=1.):
    a = build_spatial_knn_graph(adata.obsm['spatial'], n_neighbors_spatial)
    morph = adata.obsm[morph_key] if use_morphology and morph_key in adata.obsm else None
    w = reweight_spatial_graph(a, expr_feat_for_graph, morph, expr_power, morph_power)
    return dict(adj_spatial=a, adj_wave=w)


def scipy_to_torch(a, device=None):
    a = a.tocoo()
    ix = torch.from_numpy(np.vstack([a.row, a.col]).astype(np.int64))
    data = torch.from_numpy(a.data.astype(np.float32))
    return torch.sparse_coo_tensor(ix, data, a.shape, device=device).coalesce()


def build_edge_index_from_adj(adj, threshold=0., add_self_loop=False):
    a = sp.triu(sp.csr_matrix(adj), k=1, format='coo')
    mask = a.data > threshold
    return (torch.from_numpy(np.vstack([a.row[mask], a.col[mask]]).astype(np.int64)),
            torch.from_numpy(a.data[mask].astype(np.float32)))


def refine_labels_by_spatial_neighbors(labels, adj_spatial, n_iter=2, self_weight=1.):
    labels = np.asarray(labels, dtype=np.int32).copy()
    a = sp.csr_matrix(adj_spatial, dtype=np.float32)
    ncls = int(labels.max()) + 1 if len(labels) else 0
    for _ in range(n_iter):
        new = labels.copy()
        for i in range(len(labels)):
            sl = slice(a.indptr[i], a.indptr[i+1])
            mask = a.data[sl] > 0
            ids, weight = a.indices[sl][mask], a.data[sl][mask]
            if not len(ids):
                continue
            votes = np.bincount(labels[ids], weights=weight, minlength=ncls)
            votes[labels[i]] += self_weight
            new[i] = int(np.argmax(votes))
        labels = new
    return labels
