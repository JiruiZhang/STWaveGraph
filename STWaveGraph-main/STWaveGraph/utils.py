import warnings
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


# =========================================================
# SSL / clustering losses
# =========================================================
def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def barlow_twins_loss(z1, z2, lambda_offdiag=5e-3):
    """
    Barlow Twins loss.
    """
    if z1.shape[0] < 2:
        return z1.new_tensor(0.0)

    z1 = (z1 - z1.mean(0)) / (z1.std(0, unbiased=False).clamp_min(1e-9))
    z2 = (z2 - z2.mean(0)) / (z2.std(0, unbiased=False).clamp_min(1e-9))

    n = z1.shape[0]
    c = (z1.T @ z2) / n

    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()

    return on_diag + lambda_offdiag * off_diag


def target_distribution(q):
    weight = (q ** 2) / (torch.sum(q, dim=0, keepdim=True) + 1e-8)
    return weight / (torch.sum(weight, dim=1, keepdim=True) + 1e-8)


def cluster_kl_loss(q):
    p = target_distribution(q).detach()
    return F.kl_div(torch.log(q + 1e-8), p, reduction="batchmean")


def cluster_balance_loss(q, prior=None):
    """
    Encourage cluster usage not to collapse.
    A softer version than forcing strict uniformity.
    """
    mean_q = torch.mean(q, dim=0)
    mean_q = mean_q / (mean_q.sum() + 1e-8)

    if prior is None:
        prior = torch.ones_like(mean_q) / mean_q.numel()
    else:
        prior = prior.to(mean_q.device)
        prior = prior / (prior.sum() + 1e-8)

    return torch.sum(prior * torch.log((prior + 1e-8) / (mean_q + 1e-8)))


# =========================================================
# Graph edge utilities
# =========================================================
def build_edge_index_from_adj(adj, threshold=0.0, add_self_loop=False):
    """
    Extract undirected upper-triangular edges from dense adjacency.

    Returns:
        edge_index: torch.LongTensor [2, E]
        edge_weight: torch.FloatTensor [E]
    """
    if isinstance(adj, np.ndarray):
        adj_t = torch.FloatTensor(adj)
    else:
        adj_t = adj.detach().cpu().float()

    adj_t = adj_t.clone()
    n = adj_t.shape[0]

    if not add_self_loop:
        adj_t[torch.arange(n), torch.arange(n)] = 0.0

    mask = torch.triu(adj_t > threshold, diagonal=1)
    edge_index = mask.nonzero(as_tuple=False).T

    if edge_index.numel() == 0:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32)
        )

    edge_weight = adj_t[edge_index[0], edge_index[1]].float()

    return edge_index.long(), edge_weight.float()


# =========================================================
# Edge-level losses
# =========================================================
def graph_smoothness_loss_edge(h, edge_index, edge_weight=None):
    """
    sum_ij w_ij ||h_i - h_j||^2 / sum_ij w_ij
    """
    if edge_index.numel() == 0:
        return h.new_tensor(0.0)

    src = edge_index[0]
    dst = edge_index[1]

    diff = h[src] - h[dst]
    dist2 = torch.sum(diff * diff, dim=1)

    if edge_weight is not None:
        ew = edge_weight.to(h.device)
        loss = (ew * dist2).sum() / (ew.sum() + 1e-8)
    else:
        loss = dist2.mean()

    return loss


def soft_boundary_loss_edge(
    h,
    q,
    edge_index,
    edge_weight=None,
    margin=1.0,
    detach_q=False
):
    """
    Soft boundary-aware loss on spatial edges.

    If q_i and q_j are similar:
        encourage h_i, h_j to be close

    If q_i and q_j are different:
        encourage h_i, h_j to be apart by margin
    """
    if edge_index.numel() == 0:
        return h.new_tensor(0.0)

    src = edge_index[0]
    dst = edge_index[1]

    hi = h[src]
    hj = h[dst]
    qi = q[src]
    qj = q[dst]

    sim_q = torch.sum(qi * qj, dim=1)
    if detach_q:
        sim_q = sim_q.detach()

    sim_q = sim_q.clamp(0.0, 1.0)
    dist = torch.norm(hi - hj, p=2, dim=1)

    loss_same = sim_q * (dist ** 2)
    loss_diff = (1.0 - sim_q) * F.relu(margin - dist).pow(2)

    loss_edge = loss_same + loss_diff

    if edge_weight is not None:
        ew = edge_weight.to(h.device)
        loss = (ew * loss_edge).sum() / (ew.sum() + 1e-8)
    else:
        loss = loss_edge.mean()

    return loss


# =========================================================
# Clustering helpers
# =========================================================
def reduce_embedding_dim(emb, pca_dim=None, random_state=42):
    """
    Optional PCA before clustering.
    """
    emb = np.asarray(emb, dtype=np.float32)

    if pca_dim is None:
        return emb, None

    n_comp = min(pca_dim, emb.shape[1], emb.shape[0] - 1)
    if n_comp < 2:
        return emb, None

    pca = PCA(n_components=n_comp, random_state=random_state)
    emb_red = pca.fit_transform(emb).astype(np.float32)
    return emb_red, pca


def mclust_R(X, num_cluster, modelNames='EEE', random_seed=2020):
    """
    Run R mclust via rpy2.
    """
    X = np.asarray(X, dtype=np.float64)

    try:
        import rpy2.robjects as robjects
        import rpy2.robjects.numpy2ri as numpy2ri
        numpy2ri.activate()
    except Exception as e:
        raise ImportError(
            "rpy2 is not available. Please install rpy2 and R first."
        ) from e

    try:
        robjects.r('suppressPackageStartupMessages(library(mclust))')
    except Exception as e:
        raise ImportError(
            "R package 'mclust' is not installed. "
            "Please run install.packages('mclust') in R."
        ) from e

    robjects.r['set.seed'](random_seed)
    rmclust = robjects.r['Mclust']

    X_r = numpy2ri.py2rpy(X)

    res = rmclust(X_r, num_cluster, modelNames=modelNames)

    labels = np.array(res.rx2('classification')).astype(np.int32) - 1

    probs = np.array(res.rx2('z')).astype(np.float32)
    if probs.ndim == 1:
        probs = probs[:, None]

    params = res.rx2('parameters')
    means = np.array(params.rx2('mean')).astype(np.float32)

    if means.ndim == 1:
        centers = means.reshape(1, -1)
    elif means.shape[0] == X.shape[1] and means.shape[1] == num_cluster:
        centers = means.T
    elif means.shape[0] == num_cluster:
        centers = means
    else:
        centers = means.reshape(num_cluster, -1)

    return labels, probs, centers


def cluster_embedding(
    emb,
    n_clusters,
    method="gmm",
    random_state=42,
    pca_dim=None,
    mclust_model="EEE",
    fallback_to_gmm=True
):
    """
    Robust downstream clustering on embedding.
    """
    method = method.lower()
    emb = np.asarray(emb, dtype=np.float32)

    emb_cluster, pca_model = reduce_embedding_dim(
        emb,
        pca_dim=pca_dim,
        random_state=random_state
    )

    if method == "gmm":
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            reg_covar=1e-5,
            random_state=random_state
        )
        gmm.fit(emb_cluster)
        probs = gmm.predict_proba(emb_cluster).astype(np.float32)
        labels = np.argmax(probs, axis=1).astype(np.int32)
        centers = gmm.means_.astype(np.float32)
        return labels, probs, centers

    elif method == "kmeans":
        km = KMeans(
            n_clusters=n_clusters,
            n_init=20,
            random_state=random_state
        )
        labels = km.fit_predict(emb_cluster).astype(np.int32)
        centers = km.cluster_centers_.astype(np.float32)

        probs = np.zeros((emb_cluster.shape[0], n_clusters), dtype=np.float32)
        probs[np.arange(emb_cluster.shape[0]), labels] = 1.0
        return labels, probs, centers

    elif method == "mclust":
        try:
            labels, probs, centers = mclust_R(
                emb_cluster,
                num_cluster=n_clusters,
                modelNames=mclust_model,
                random_seed=random_state
            )
            return labels, probs.astype(np.float32), centers.astype(np.float32)

        except Exception as e:
            if fallback_to_gmm:
                warnings.warn(
                    f"[cluster_embedding] mclust failed: {repr(e)}. "
                    f"Fallback to sklearn GaussianMixture.",
                    RuntimeWarning
                )
                gmm = GaussianMixture(
                    n_components=n_clusters,
                    covariance_type="full",
                    reg_covar=1e-5,
                    random_state=random_state
                )
                gmm.fit(emb_cluster)
                probs = gmm.predict_proba(emb_cluster).astype(np.float32)
                labels = np.argmax(probs, axis=1).astype(np.int32)
                centers = gmm.means_.astype(np.float32)
                return labels, probs, centers
            else:
                raise e

    else:
        raise ValueError("method must be 'gmm', 'kmeans', or 'mclust'")


def refine_labels_by_spatial_neighbors(labels, adj_spatial, n_iter=2, self_weight=1.0):
    """
    Spatial refinement by weighted majority vote on spatial neighbors.
    """
    labels = np.asarray(labels, dtype=np.int32).copy()
    adj = np.asarray(adj_spatial, dtype=np.float32)
    n = len(labels)
    n_cls = int(labels.max()) + 1 if len(labels) > 0 else 0

    for _ in range(n_iter):
        new_labels = labels.copy()
        for i in range(n):
            neigh = np.where(adj[i] > 0)[0]
            if len(neigh) == 0:
                continue

            weights = adj[i, neigh].astype(np.float32)
            votes = np.bincount(labels[neigh], weights=weights, minlength=n_cls)
            votes[labels[i]] += self_weight
            new_labels[i] = int(np.argmax(votes))

        labels = new_labels

    return labels

def cluster_consistency_loss(q1, q2):
    return 0.5 * (
        F.kl_div(torch.log(q1 + 1e-8), q2.detach(), reduction="batchmean") +
        F.kl_div(torch.log(q2 + 1e-8), q1.detach(), reduction="batchmean")
    )
# =========================================================
# Evaluation
# =========================================================
def evaluate_clustering(y_true, y_pred):
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    return {"ARI": ari, "NMI": nmi}