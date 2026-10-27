"""Strict external clustering with truthful per-run provenance."""
import time
import numpy as np


class ClusterExecutionError(RuntimeError):
    def __init__(self, message, record):
        super().__init__(message)
        self.record = record


def _validate_output(result, n, clusters, dimensions):
    labels, probabilities, centers = [np.asarray(x) for x in result]
    if labels.shape != (n,) or labels.dtype.kind not in 'iu':
        raise ValueError('Cluster labels must be an integer vector in observation order')
    if np.any(labels < 0) or np.any(labels >= clusters):
        raise ValueError('Cluster labels outside the configured range')
    if probabilities.shape != (n, clusters):
        raise ValueError('Unexpected assignment-probability shape')
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError('Non-finite or negative assignment probabilities')
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-4, rtol=1e-4):
        raise ValueError('Assignment probabilities are not normalized')
    if centers.shape != (clusters, dimensions) or not np.isfinite(centers).all():
        raise ValueError('Unexpected or non-finite cluster centers')
    return labels.astype(np.int32), probabilities.astype(np.float32), centers.astype(np.float32)


def cluster_with_record(embedding, n_clusters, method='mclust', seed=42,
                        pca_dim=None, mclust_model='EEE', allow_fallback=False,
                        backend=None):
    """Disable implicit fallback; any explicitly allowed fallback changes the variant name.

    The backend is the unchanged core cluster_embedding function. Tests can inject
    a backend without requiring R, sklearn or torch in the test environment.
    """
    emb = np.asarray(embedding, dtype=np.float32)
    if emb.ndim != 2 or emb.shape[0] < 2 or not np.isfinite(emb).all():
        raise ValueError('Embedding must be a finite N x D array with N >= 2')
    if not isinstance(n_clusters, (int, np.integer)) or not 1 <= n_clusters <= len(emb):
        raise ValueError('Invalid cluster count')
    method = str(method).lower()
    if method not in ['mclust', 'gmm', 'kmeans']:
        raise ValueError('Unsupported clustering method: '+method)
    if backend is None:
        from STWaveGraph.utils import cluster_embedding
        backend = cluster_embedding
    dims = emb.shape[1]
    if pca_dim is not None:
        components = min(int(pca_dim), dims, len(emb)-1)
        if components >= 2:
            dims = components
    record = dict(requested_method=method, actual_method=None,
        variant=method, seed=int(seed), n_clusters=int(n_clusters),
        pca_dim=pca_dim, mclust_model=mclust_model if method=='mclust' else None,
        fallback_allowed=bool(allow_fallback), fallback_used=False,
        fallback_reason=None, status='started', attempts=[])
    started = time.monotonic()

    def execute(actual):
        attempt = dict(method=actual, status='started')
        record['attempts'].append(attempt)
        t = time.monotonic()
        try:
            result = backend(emb, n_clusters=int(n_clusters), method=actual,
                random_state=int(seed), pca_dim=pca_dim,
                mclust_model=mclust_model, fallback_to_gmm=False)
            result = _validate_output(result, len(emb), int(n_clusters), dims)
            attempt['status'] = 'complete'
            attempt['elapsed_seconds'] = time.monotonic()-t
            return result
        except Exception as error:
            attempt.update(status='failed', error_type=type(error).__name__,
                           error=str(error), elapsed_seconds=time.monotonic()-t)
            raise

    try:
        try:
            labels, probabilities, centers = execute(method)
            record['actual_method'] = method
        except Exception as error:
            record['fallback_reason'] = type(error).__name__+': '+str(error)
            if method != 'mclust' or not allow_fallback:
                raise
            record.update(fallback_used=True, variant='mclust_failed__gmm_fallback')
            labels, probabilities, centers = execute('gmm')
            record['actual_method'] = 'gmm'
        record.update(status='complete', elapsed_seconds=time.monotonic()-started)
        return labels, probabilities, centers, record
    except Exception as error:
        record.update(status='failed', elapsed_seconds=time.monotonic()-started)
        raise ClusterExecutionError('Strict clustering failed: '+str(error), record) from error
