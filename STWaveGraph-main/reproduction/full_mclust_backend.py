"""Explicit mclust full-data hierarchical initialization variant; no implicit GMM."""
import numpy as np
from STWaveGraph.utils import cluster_embedding, reduce_embedding_dim


def cluster_embedding_fullinit(emb, n_clusters, method='mclust', random_state=42,
                              pca_dim=None, mclust_model='EEE', fallback_to_gmm=False):
    if method != 'mclust':
        return cluster_embedding(emb, n_clusters, method=method, random_state=random_state,
            pca_dim=pca_dim, mclust_model=mclust_model, fallback_to_gmm=False)
    if fallback_to_gmm:
        raise ValueError('Full-initialization variant prohibits implicit GMM fallback')
    import rpy2.robjects as ro
    import rpy2.robjects.numpy2ri as numpy2ri
    numpy2ri.activate()
    x, _ = reduce_embedding_dim(emb, pca_dim=pca_dim, random_state=random_state)
    x = np.asarray(x, dtype=np.float64)
    ro.r('suppressPackageStartupMessages(library(mclust))')
    ro.r['set.seed'](int(random_state))
    initialization = ro.ListVector({'subset': ro.IntVector(range(1, len(x)+1))})
    result = ro.r['Mclust'](numpy2ri.py2rpy(x), int(n_clusters),
        modelNames=mclust_model, initialization=initialization, warn=True, verbose=False)
    if bool(ro.r['is.null'](result)[0]):
        raise RuntimeError('mclust returned NULL with full-data initialization, model='+mclust_model)
    assert str(result.rx2('modelName')[0]) == mclust_model
    assert int(result.rx2('G')[0]) == n_clusters
    labels = np.asarray(result.rx2('classification')).astype(np.int32)-1
    probabilities = np.asarray(result.rx2('z'), dtype=np.float32)
    if probabilities.ndim == 1:
        probabilities = probabilities[:, None]
    means = np.asarray(result.rx2('parameters').rx2('mean'), dtype=np.float32)
    if means.ndim == 1:
        centers = means.reshape(1, -1)
    elif means.shape == (x.shape[1], n_clusters):
        centers = means.T
    elif means.shape[0] == n_clusters:
        centers = means
    else:
        centers = means.reshape(n_clusters, -1)
    return labels, probabilities, centers
