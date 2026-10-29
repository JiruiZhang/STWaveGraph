"""Shared configuration, training and evaluation functions for the three entry points."""
from pathlib import Path
import argparse, json, os, sys, hashlib, subprocess, time, traceback, copy, importlib, types
from types import SimpleNamespace
HERE = Path(__file__).resolve().parent
ROOT = None
OLD = None
CONFIG = None
CONFIG_PATH = None
BASELINE_SOURCES = None
METHODS = ['STWaveGraph', 'DeepGFT', 'GraphST', 'STAGATE', 'SpaGCN', 'SEDR', 'DeepST', 'GAAEST', 'SpaCAE', '3dOT']

def load_configs(path):
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    assert len(obj['benchmark']['samples']) == 17
    runs = obj['benchmark']['runs']
    keys = {(r['sample'], r['method'], int(r['seed'])) for r in runs}
    assert len(runs) == len(keys) == 850
    assert len(obj['ablation']['base']) == 15
    assert len(obj['ablation']['variants']) == 240
    return obj

def source_hashes():
    return {p.name: sha(p) for p in (ROOT / 'STWaveGraph').glob('*.py')}

def directory_hash(path):
    h = hashlib.sha256()
    for p in sorted(Path(path).rglob('*.py')):
        h.update(p.relative_to(path).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()

def runtime(root):
    global ROOT, np, pd, sp, sc, torch, nn, F, h5py, fcntl, random
    global cluster_embedding, reduce_embedding_dim, SpectralGraphWaveLayer, inverse_softplus
    global adjusted_rand_score, normalized_mutual_info_score, linear_sum_assignment, NearestNeighbors, cdist, cKDTree
    ROOT = Path(root).resolve()
    if not (ROOT / 'STWaveGraph/model.py').is_file():
        raise FileNotFoundError('Expected STWaveGraph/model.py under --root: ' + str(ROOT))
    sys.path.insert(0, str(ROOT))
    import numpy as np, pandas as pd, scipy.sparse as sp, scanpy as sc, torch, h5py, random
    import fcntl
    from torch import nn
    from torch.nn import functional as F
    from STWaveGraph.utils import cluster_embedding, reduce_embedding_dim
    from STWaveGraph.model import SpectralGraphWaveLayer, inverse_softplus
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from scipy.optimize import linear_sum_assignment
    from sklearn.neighbors import NearestNeighbors
    from scipy.spatial.distance import cdist
    from scipy.spatial import cKDTree
    install_ablation_classes()

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def save(p, v):
    Path(p).write_text(json.dumps(v, indent=2, allow_nan=False) + '\n')

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_paths(run, method):
    run = Path(run)
    paths = [ROOT]
    sub = {'STAGATE': 'STAGATE', 'SpaGCN': 'SpaGCN/SpaGCN_package', 'SpaCAE': 'SpaCAE/SpaCAE', '3dOT': '3dOT'}.get(method, method)
    if method != 'STWaveGraph':
        paths.insert(0, BASELINE_SOURCES / sub)
    for p in reversed(paths):
        sys.path.insert(0, str(p))
    return paths

def dense(x):
    return x.toarray() if sp.issparse(x) else np.asarray(x)

def continuous(a):
    return a.uns['input_kind'] == 'continuous_noncount'

def log_input(a, scale=False, target=10000.0):
    a = a.copy()
    if not continuous(a):
        sc.pp.normalize_total(a, target_sum=target)
    sc.pp.log1p(a)
    if scale:
        sc.pp.scale(a, zero_center=False, max_value=10)
    return a

def select_genes(a, n):
    a = a.copy()
    x = dense(a.X)
    v = np.var(x, axis=0) > 0
    if continuous(a) or a.n_vars <= n:
        a.var['highly_variable'] = v
    else:
        sc.pp.highly_variable_genes(a, flavor='seurat_v3', n_top_genes=n)
    assert a.var['highly_variable'].sum() > 1
    return a

def spatial_net(a, k):
    from sklearn.neighbors import NearestNeighbors
    n = min(k + 1, len(a))
    d, ix = NearestNeighbors(n_neighbors=n).fit(a.obsm['spatial']).kneighbors(a.obsm['spatial'])
    rr = []
    cc = []
    ds = []
    for i in range(len(a)):
        use = ix[i] != i
        cols = ix[i][use][:k]
        dist = d[i][use][:k]
        assert len(cols) == min(k, len(a) - 1)
        rr.extend([i] * len(cols))
        cc.extend(cols)
        ds.extend(dist)
    row = np.asarray(rr, dtype=int)
    col = np.asarray(cc, dtype=int)
    dd = np.asarray(ds)
    a.uns['Spatial_Net'] = pd.DataFrame({'Cell1': np.asarray(a.obs_names)[row], 'Cell2': np.asarray(a.obs_names)[col], 'Distance': dd})
    return a

def graph_from_coords(coords, k=6):
    from sklearn.neighbors import kneighbors_graph
    g = kneighbors_graph(coords, n_neighbors=min(k, len(coords) - 1), include_self=False, mode='connectivity')
    return g.maximum(g.T).tocsr()

def forward_fixture(model, args, operation, index):
    model.eval()
    with torch.no_grad():
        if operation == 'wave':
            return torch.nn.functional.normalize(model.inference(*args)[0], p=2, dim=1)
        if operation == 'decode':
            return model.decode(model.get_features(*args)).squeeze(0)
        y = model(*args)
        return y[index] if index is not None else y

def cpu_tree(x):
    if torch.is_tensor(x):
        return x.detach().cpu()
    if isinstance(x, tuple):
        return tuple((cpu_tree(v) for v in x))
    if isinstance(x, list):
        return [cpu_tree(v) for v in x]
    if isinstance(x, dict):
        return {k: cpu_tree(v) for k, v in x.items()}
    if hasattr(x, 'cpu'):
        return x.cpu()
    return x

def device_tree(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, tuple):
        return tuple((device_tree(v, device) for v in x))
    if isinstance(x, list):
        return [device_tree(v, device) for v in x]
    if isinstance(x, dict):
        return {k: device_tree(v, device) for k, v in x.items()}
    if hasattr(x, 'to'):
        return x.to(device)
    return x

def spatial_edges(coords, neighbors=6):
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2 or (not np.isfinite(coords).all()):
        raise ValueError('Expected finite N x 2 spatial coordinates')
    n = len(coords)
    if n < 2:
        raise ValueError('At least two observations required')
    nearest = NearestNeighbors(n_neighbors=min(neighbors + 1, n)).fit(coords)
    distances, indices = nearest.kneighbors(coords)
    pairs = np.stack([np.repeat(np.arange(n), indices.shape[1] - 1), indices[:, 1:].ravel()], axis=1)
    pairs.sort(axis=1)
    pairs = np.unique(pairs[pairs[:, 0] != pairs[:, 1]], axis=0)
    positive = distances[:, 1][distances[:, 1] > 0]
    if not len(positive):
        raise ValueError('No positive nearest-neighbor spacing')
    return (pairs, float(np.median(positive)))

def boundary_metrics(coords, truth, prediction, valid, edges, spacing):
    coords = np.asarray(coords, dtype=np.float64)
    truth, prediction = (np.asarray(truth), np.asarray(prediction))
    valid = np.asarray(valid, dtype=bool)
    edges = np.asarray(edges, dtype=np.int64)
    n = len(coords)
    if truth.shape != (n,) or prediction.shape != (n,) or valid.shape != (n,):
        raise ValueError('All labels/mask must match coordinate observation order')
    if edges.ndim != 2 or edges.shape[1] != 2 or np.any(edges < 0) or np.any(edges >= n):
        raise ValueError('Invalid evaluation edges')
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError('Invalid spacing')
    use = edges[valid[edges[:, 0]] & valid[edges[:, 1]]]
    reference = truth[use[:, 0]] != truth[use[:, 1]]
    estimated = prediction[use[:, 0]] != prediction[use[:, 1]]
    true_positive = int(np.count_nonzero(reference & estimated))
    nr, npred = (int(reference.sum()), int(estimated.sum()))
    result = dict(evaluation_edges=len(use), reference_boundary_edges=nr, predicted_boundary_edges=npred, boundary_true_positive=true_positive)
    if not nr:
        result.update(boundary_precision=None, boundary_recall=None, boundary_f1=None, boundary_assd_native=None, boundary_assd_spacing=None, boundary_status='no_reference_boundary')
    elif not npred:
        result.update(boundary_precision=0.0, boundary_recall=0.0, boundary_f1=0.0, boundary_assd_native=None, boundary_assd_spacing=None, boundary_status='empty_predicted_boundary')
    else:
        precision, recall = (true_positive / npred, true_positive / nr)
        midpoints = coords[use].mean(axis=1)
        ref_points, pred_points = (midpoints[reference], midpoints[estimated])
        d_pred = cKDTree(ref_points).query(pred_points)[0]
        d_ref = cKDTree(pred_points).query(ref_points)[0]
        assd = float((d_pred.mean() + d_ref.mean()) / 2)
        result.update(boundary_precision=precision, boundary_recall=recall, boundary_f1=2 * true_positive / (npred + nr), boundary_assd_native=assd, boundary_assd_spacing=assd / spacing, boundary_status='defined')
    return result

def cluster_embedding_fullinit(emb, n_clusters, method='mclust', random_state=42, pca_dim=None, mclust_model='EEE', fallback_to_gmm=False):
    if method != 'mclust':
        return cluster_embedding(emb, n_clusters, method=method, random_state=random_state, pca_dim=pca_dim, mclust_model=mclust_model, fallback_to_gmm=False)
    if fallback_to_gmm:
        raise ValueError('Full-initialization variant prohibits implicit GMM fallback')
    import rpy2.robjects as ro
    import rpy2.robjects.numpy2ri as numpy2ri
    numpy2ri.activate()
    x, _ = reduce_embedding_dim(emb, pca_dim=pca_dim, random_state=random_state)
    x = np.asarray(x, dtype=np.float64)
    ro.r('suppressPackageStartupMessages(library(mclust))')
    ro.r['set.seed'](int(random_state))
    initialization = ro.ListVector({'subset': ro.IntVector(range(1, len(x) + 1))})
    result = ro.r['Mclust'](numpy2ri.py2rpy(x), int(n_clusters), modelNames=mclust_model, initialization=initialization, warn=True, verbose=False)
    if bool(ro.r['is.null'](result)[0]):
        raise RuntimeError('mclust returned NULL with full-data initialization, model=' + mclust_model)
    assert str(result.rx2('modelName')[0]) == mclust_model
    assert int(result.rx2('G')[0]) == n_clusters
    labels = np.asarray(result.rx2('classification')).astype(np.int32) - 1
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
    return (labels, probabilities, centers)

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
    if not np.allclose(probabilities.sum(axis=1), 1, atol=0.0001, rtol=0.0001):
        raise ValueError('Assignment probabilities are not normalized')
    if centers.shape != (clusters, dimensions) or not np.isfinite(centers).all():
        raise ValueError('Unexpected or non-finite cluster centers')
    return (labels.astype(np.int32), probabilities.astype(np.float32), centers.astype(np.float32))

def cluster_with_record(embedding, n_clusters, method='mclust', seed=42, pca_dim=None, mclust_model='EEE', allow_fallback=False, backend=None):
    """Disable implicit fallback; any explicitly allowed fallback changes the variant name.

    The backend is the unchanged core cluster_embedding function. Tests can inject
    a backend without requiring R, sklearn or torch in the test environment.
    """
    emb = np.asarray(embedding, dtype=np.float32)
    if emb.ndim != 2 or emb.shape[0] < 2 or (not np.isfinite(emb).all()):
        raise ValueError('Embedding must be a finite N x D array with N >= 2')
    if not isinstance(n_clusters, (int, np.integer)) or not 1 <= n_clusters <= len(emb):
        raise ValueError('Invalid cluster count')
    method = str(method).lower()
    if method not in ['mclust', 'gmm', 'kmeans']:
        raise ValueError('Unsupported clustering method: ' + method)
    if backend is None:
        from STWaveGraph.utils import cluster_embedding
        backend = cluster_embedding
    dims = emb.shape[1]
    if pca_dim is not None:
        components = min(int(pca_dim), dims, len(emb) - 1)
        if components >= 2:
            dims = components
    record = dict(requested_method=method, actual_method=None, variant=method, seed=int(seed), n_clusters=int(n_clusters), pca_dim=pca_dim, mclust_model=mclust_model if method == 'mclust' else None, fallback_allowed=bool(allow_fallback), fallback_used=False, fallback_reason=None, status='started', attempts=[])
    started = time.monotonic()

    def execute(actual):
        attempt = dict(method=actual, status='started')
        record['attempts'].append(attempt)
        t = time.monotonic()
        try:
            result = backend(emb, n_clusters=int(n_clusters), method=actual, random_state=int(seed), pca_dim=pca_dim, mclust_model=mclust_model, fallback_to_gmm=False)
            result = _validate_output(result, len(emb), int(n_clusters), dims)
            attempt['status'] = 'complete'
            attempt['elapsed_seconds'] = time.monotonic() - t
            return result
        except Exception as error:
            attempt.update(status='failed', error_type=type(error).__name__, error=str(error), elapsed_seconds=time.monotonic() - t)
            raise
    try:
        try:
            labels, probabilities, centers = execute(method)
            record['actual_method'] = method
        except Exception as error:
            record['fallback_reason'] = type(error).__name__ + ': ' + str(error)
            if method != 'mclust' or not allow_fallback:
                raise
            record.update(fallback_used=True, variant='mclust_failed__gmm_fallback')
            labels, probabilities, centers = execute('gmm')
            record['actual_method'] = 'gmm'
        record.update(status='complete', elapsed_seconds=time.monotonic() - started)
        return (labels, probabilities, centers, record)
    except Exception as error:
        record.update(status='failed', elapsed_seconds=time.monotonic() - started)
        raise ClusterExecutionError('Strict clustering failed: ' + str(error), record) from error
VARIANTS = ['wave', 'heat', 'cheb5', 'pde_d4', 'local_only', 'wave_only', 'expr_off', 'velocity_zero', 'bands1', 'bands2', 'low_only', 'residual_off', 'fusion_mean', 'fusion_concat', 'boundary_off', 'balance_toggle']

def install_ablation_classes():
    global AlternativePropagation, WaveAblation

    class AlternativePropagation(nn.Module):

        def __init__(self, original, kind, trainer):
            super().__init__()
            self.kind = kind
            self.n_bands = original.n_bands
            self._band_masks = original._band_masks
            self.raw_res_scale = nn.Parameter(original.raw_res_scale.detach().clone())
            if kind == 'heat':
                self.raw_time = nn.Parameter(inverse_softplus([1.0] * self.n_bands))
            elif kind == 'cheb5':
                c = torch.zeros(6)
                c[0] = 1.0
                self.coefficients = nn.Parameter(c)
            elif kind == 'pde_d4':
                a = trainer.adj_wave_raw
                src, dst = np.where(np.triu(a > 0, 1))
                degree = np.maximum(a.sum(1), 1e-08)
                w = np.sqrt(a[src, dst])
                U = trainer.eigvecs_t
                gu = torch.as_tensor(w / np.sqrt(degree[src]), device=U.device)[:, None] * U[src] - torch.as_tensor(w / np.sqrt(degree[dst]), device=U.device)[:, None] * U[dst]
                self.register_buffer('GU', gu)
                self.raw_k = nn.Parameter(torch.full((4, original.hidden_dim), float(np.arctanh(0.5))))
                self.step = 0.2

        def forward(self, u0, v0, U, lam):
            uh = U.T @ u0
            if self.kind == 'pde_d4':
                c = uh
                for raw in self.raw_k:
                    k = 2 * torch.tanh(raw)
                    c = c - self.step * (self.GU.T @ (torch.tanh(self.GU @ c * k) * k))
                out = U @ c
            else:
                gain = torch.zeros_like(lam)
                for b, mask in enumerate(self._band_masks(lam)):
                    if self.kind == 'heat':
                        t = (F.softplus(self.raw_time[b]) + 1e-06).clamp(0.001, 3)
                        gain[mask] = torch.exp(-t * lam[mask])
                    else:
                        x = lam[mask] - 1
                        t0 = torch.ones_like(x)
                        t1 = x
                        g = self.coefficients[0] * t0 + self.coefficients[1] * t1
                        for j in range(2, 6):
                            t2 = 2 * x * t1 - t0
                            g = g + self.coefficients[j] * t2
                            t0, t1 = (t1, t2)
                        gain[mask] = g
                out = U @ (gain[:, None] * uh)
            scale = (F.softplus(self.raw_res_scale) + 1e-06).clamp(0, 2)
            return out + scale * (u0 - U @ uh)

    class WaveAblation(nn.Module):

        def __init__(self, original, zero_velocity=False, remove_residual=False):
            super().__init__()
            self.original = original
            self.zero_velocity = zero_velocity
            self.remove_residual = remove_residual

        def forward(self, u0, v0, U, lam):
            if self.zero_velocity:
                v0 = torch.zeros_like(v0)
            out = self.original(u0, v0, U, lam)
            if self.remove_residual:
                s = (F.softplus(self.original.raw_res_scale) + self.original.eps).clamp(0, 2)
                out = out - s * (u0 - U @ (U.T @ u0))
            return out

def altered_forward(self, feat, adj_local, eigvecs, eigvals):
    h0 = self.input_proj(feat)
    mode = self.ablation_mode
    if mode == 'local_only':
        context = self.local_encoder(h0, adj_local)
    elif mode == 'wave_only':
        context = self.wave_layer(h0, self.velocity_proj(h0), eigvecs, eigvals)
    else:
        hw = self.wave_layer(h0, self.velocity_proj(h0), eigvecs, eigvals)
        hl = self.local_encoder(h0, adj_local)
        if mode == 'fusion_mean':
            context = 0.5 * (hw + hl)
        else:
            context = self.concat_projection(torch.cat([hw, hl], dim=1))
    skip = torch.sigmoid(self.skip_gate)
    h = self.out_norm(skip * h0 + (1 - skip) * context)
    return (h, F.normalize(self.projector(h), p=2, dim=1), self.decoder(h))

def configure_model(trainer, variant):
    e = trainer.model.encoder
    if variant in ['heat', 'cheb5', 'pde_d4']:
        e.wave_layer = AlternativePropagation(e.wave_layer, variant, trainer).to(trainer.device)
    elif variant in ['velocity_zero', 'residual_off']:
        e.wave_layer = WaveAblation(e.wave_layer, variant == 'velocity_zero', variant == 'residual_off')
    elif variant in ['bands1', 'bands2']:
        e.wave_layer = SpectralGraphWaveLayer(trainer.hidden_dim, n_bands=int(variant[-1])).to(trainer.device)
    elif variant in ['local_only', 'wave_only', 'fusion_mean', 'fusion_concat']:
        e.ablation_mode = variant
        if variant == 'fusion_concat':
            e.concat_projection = nn.Linear(2 * trainer.hidden_dim, trainer.hidden_dim).to(trainer.device)
            with torch.no_grad():
                e.concat_projection.weight.zero_()
                e.concat_projection.bias.zero_()
                eye = torch.eye(trainer.hidden_dim, device=trainer.device) * 0.5
                e.concat_projection.weight[:, :trainer.hidden_dim] = eye
                e.concat_projection.weight[:, trainer.hidden_dim:] = eye
        e.forward_once = types.MethodType(altered_forward, e)
    if variant in ['heat', 'cheb5', 'pde_d4', 'local_only', 'velocity_zero']:
        for p in e.velocity_proj.parameters():
            p.requires_grad_(False)
    if variant == 'local_only':
        for p in e.wave_layer.parameters():
            p.requires_grad_(False)
    if variant == 'wave_only':
        for p in e.local_encoder.parameters():
            p.requires_grad_(False)
    if variant in ['local_only', 'wave_only', 'fusion_mean', 'fusion_concat']:
        for p in e.fuse_gate.parameters():
            p.requires_grad_(False)
    if variant == 'residual_off':
        e.wave_layer.original.raw_res_scale.requires_grad_(False)
    trainer.optimizer = torch.optim.AdamW([p for p in trainer.model.parameters() if p.requires_grad], lr=trainer.lr, weight_decay=trainer.weight_decay)
    return dict(total=sum((p.numel() for p in trainer.model.parameters())), trainable=sum((p.numel() for p in trainer.model.parameters() if p.requires_grad)), propagation_trainable=sum((p.numel() for p in e.wave_layer.parameters() if p.requires_grad)))

def prepare_deepgft(run, sample, a, family):
    run = Path(run)
    directory = run / 'preprocessing_cache/DeepGFT' / sample
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'author_preprocessed_inputs.pt'
    identity = dict(input_sha256=sha(run / 'inputs' / (sample + '.h5ad')), source_sha256=directory_hash(BASELINE_SOURCES), adapter_sha256=sha(Path(__file__)), preprocessing_seed=42)
    with (directory / 'cache.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            record = json.loads((directory / 'manifest.json').read_text())
            assert record['identity'] == identity and record['sha256'] == sha(path)
            result = torch.load(path, map_location='cpu', weights_only=False)
            return (result, record)
        seed_all(42)
        u = importlib.import_module('DeepGFT.utils')
        gn = importlib.import_module('DeepGFT.genenet')
        small = a.n_vars < 2000
        iscont = continuous(a)
        if not small and (not iscont):
            u.prefilter_genes(a, min_cells=3)
        csvg = 0.1 if family == 'Breast' else 0.0001
        c1 = 0.5 if family == 'Breast' else 5 if family == 'STARmap' else 0.05
        c2 = 0.001 if family == 'Breast' else 0.005
        signal = min(256 if family == 'STARmap' else 1500, a.n_obs - 1, a.n_vars - 1)
        middle = min(2 if family == 'STARmap' else 3, signal - 1)
        cluster = family != 'STARmap'
        if iscont or small:
            a = log_input(a)
            a0 = a.copy()
            u.low_pass_enhancement(a, ratio_low_freq=15, c=csvg, spatial_info='spatial', ratio_neighbors=0.3, inplace=True)
        else:
            a, a0 = u.svg(a, svg_method='gft_top', n_top=3000, csvg=csvg)
        a = a.copy()
        a.X = sp.csr_matrix(a.X)
        a0 = a0.copy()
        assert list(a.obs.columns) == []
        u.obtain_spotnet(a, knn_method='KNN', k_cutoff=6, prune=False)
        gf, ge, gl = u.f2s_gene(a, gene_signal=signal, c1=c1)
        gn.obtain_genenet(a)
        sf, se, sl = u.f2s_spot(a, spot_signal=signal, middle=middle, c2=c2)
        if cluster:
            u.obtain_pre_spotnet(a, a0)
        result = dict(adata=a, gf=gf, ge=ge, sf=sf, se=se, signal=signal, middle=middle, cluster=cluster, csvg=csvg, c1=c1, c2=c2)
        for x in [a.X.data, gf, ge, sf, se]:
            assert np.isfinite(x).all()
        tmp = directory / 'author_preprocessed_inputs.pending.pt'
        torch.save(result, tmp)
        tmp.rename(path)
        record = dict(identity=identity, sha256=sha(path), gene_count=int(a.n_vars), observation_count=int(a.n_obs), genes=list(a.var_names), path=str(path), annotations_used=False)
        save(directory / 'manifest.json', record)
        return (result, record)

def ah(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def prepare_benchmark_inputs(run):
    RUN = Path(run)
    OUT = RUN / 'inputs'
    OUT.mkdir(parents=True, exist_ok=False)
    configs = CONFIG['benchmark']['inherited']
    rows = []
    specs = [('DLPFC_' + s, 'DLPFC', 'Data/DLPFC/' + s, 5 if s in ['151669', '151670', '151671', '151672'] else 7) for s in ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673', '151674', '151675', '151676']]
    specs += [('Breast', 'Breast', 'Data/Human_Breast_Cancer', 20), ('MouseBrain', 'MouseBrain', 'Data/Mouse_Brain_Anterior', 52), ('osmFISH', 'osmFISH', 'Data/osmFISHresult/osmFISH_cortex.h5', 11), ('MERFISH', 'MERFISH', 'Data/adata.h5ad', 8), ('STARmap', 'STARmap', 'Data/raw/adata.h5ad', 7)]
    for sample, family, rel, k in specs:
        p = ROOT / rel
        rawfiles = []
        if p.is_dir():
            a = sc.read_visium(p, count_file='filtered_feature_bc_matrix.h5', load_images=True)
            m = pd.read_csv(p / 'metadata.tsv', sep='\t')
            key = {'DLPFC': 'barcode', 'Breast': 'ID', 'MouseBrain': 'Unnamed: 0'}[family]
            col = 'layer_guess' if family == 'DLPFC' else 'ground_truth'
            assert not m[key].astype(str).duplicated().any()
            m[key] = m[key].astype(str)
            assert not pd.Index(a.obs_names.astype(str)).difference(m[key]).size
            truth = m.set_index(key)[col].reindex(a.obs_names.astype(str))
            truth.index = a.obs_names
            rawfiles = [p / 'filtered_feature_bc_matrix.h5', p / 'metadata.tsv'] + list((p / 'spatial').glob('*positions*.csv'))
            donor_info = {c: sorted(m[c].astype(str).unique().tolist()) for c in m.columns if any((s in c.lower() for s in ['donor', 'subject', 'brnum', 'patient']))}
        elif family == 'osmFISH':
            with h5py.File(p) as f:
                dec = lambda xs: [x.decode() if isinstance(x, bytes) else str(x) for x in xs]
                a = sc.AnnData(sp.csr_matrix(f['X'][:]), obs=pd.DataFrame(index=['cell_' + str(i) for i in range(len(f['X']))]), var=pd.DataFrame(index=dec(f['Genes'][:])))
                a.obsm['spatial'] = f['Pos'][:]
                truth = pd.Series(dec(f['Region'][:]), index=a.obs_names)
            rawfiles = [p]
            donor_info = {}
            full = sc.read_h5ad(ROOT / 'Data/osmFISHresult/osmFISH_codeluppi2018spatial_cortex_data.h5ad')
            keep = full.obs['Region'].astype(str) != 'Excluded'
            full = full[keep].copy()
            assert full.shape == a.shape
            ix = np.arange(a.n_obs)
            distances = np.linalg.norm(np.asarray(full.obsm['spatial'], dtype=np.float32) - np.asarray(a.obsm['spatial'], dtype=np.float32), axis=1)
            assert np.allclose(np.asarray(full.obsm['spatial']), a.obsm['spatial'], rtol=2 * np.finfo(np.float32).eps, atol=1e-05)
            full = full[ix, a.var_names].copy()
            assert np.array_equal(np.asarray(full.X.toarray() if sp.issparse(full.X) else full.X), a.X.toarray())
            assert np.array_equal(full.obs['Region'].astype(str).to_numpy(), truth.to_numpy())
            pd.DataFrame(dict(custom_barcode=a.obs_names, reference_barcode=full.obs_names)).to_csv(RUN / 'osmFISH_barcode_identity.csv', index=False)
            (RUN / 'osmFISH_identity_check.json').write_text(json.dumps(dict(reference=str(ROOT / 'Data/osmFISHresult/osmFISH_codeluppi2018spatial_cortex_data.h5ad'), reference_sha256=sha(ROOT / 'Data/osmFISHresult/osmFISH_codeluppi2018spatial_cortex_data.h5ad'), exclusion='Region == Excluded', n_excluded=int((~keep).sum()), matched_cells=int(a.n_obs), identity_policy='Original row order; joint exact expression/gene/float32-coordinate comparison; reference IDs retained; nearest-neighbor identity is ambiguous because coordinates repeat.', mapping_same_order=True, coordinate_max_difference_float32=float(distances.max()), coordinate_raw_max_difference=float(np.max(np.abs(np.asarray(full.obsm['spatial']) - a.obsm['spatial']))), coordinate_precision_policy='Reference is float32; relative tolerance 2*float32 epsilon plus absolute 1e-5; original custom coordinates retained and converted to float32 for models/metrics, as in prior protocol.', expression_exact=True, region_exact=True), indent=2))
        else:
            a = sc.read_h5ad(p)
            truth = a.obs['cluster'].copy()
            rawfiles = [p]
            donor_info = {}
        a.var_names_make_unique()
        assert a.obs_names.is_unique and a.var_names.is_unique
        x = sp.csr_matrix(a.X, dtype=np.float32)
        coords = np.asarray(a.obsm['spatial'], dtype=np.float32)
        assert np.isfinite(x.data).all() and x.data.min() >= 0 and np.isfinite(coords).all()
        unique_coords = np.unique(coords, axis=0).shape[0]
        valid = truth.notna() & ~truth.astype(str).isin(['', 'nan', 'NA', '__MISSING__'])
        ref = pd.DataFrame(dict(truth=truth.astype(object).fillna('__MISSING__').astype(str), valid_annotation=valid, x=coords[:, 0], y=coords[:, 1]), index=a.obs_names.astype(str))
        ref.index.name = 'barcode'
        ref.to_csv(OUT / (sample + '_reference.csv'))
        clean = sc.AnnData(x, obs=pd.DataFrame(index=a.obs_names.astype(str)), var=pd.DataFrame(index=a.var_names.astype(str)))
        clean.obsm['spatial'] = coords
        clean.uns['input_kind'] = 'continuous_noncount' if family == 'MERFISH' else 'integer_counts'
        clean.write_h5ad(OUT / (sample + '.h5ad'), compression='gzip')
        noninteger = float(np.mean(np.abs(x.data - np.round(x.data)) > 1e-05))
        if family != 'MERFISH':
            assert noninteger == 0, (sample, noninteger)
        cfg = dict(configs[family]['config'], n_clusters=k, mclust_fallback_to_gmm=False)
        cfg['random_seed'] = 42
        sample_cfg = CONFIG_PATH
        rec = dict(sample=sample, family=family, n_clusters=k, shape=list(clean.shape), n_evaluated=int(valid.sum()), input_kind=clean.uns['input_kind'], noninteger_fraction_nonzero=noninteger, unique_coordinate_count=int(unique_coords), duplicate_coordinate_extra_rows=int(len(coords) - unique_coords), raw_files={str(q): sha(q) for q in rawfiles}, input_h5ad_sha256=sha(OUT / (sample + '.h5ad')), reference_sha256=sha(OUT / (sample + '_reference.csv')), wave_config_sha256=sha(sample_cfg), barcode_sha256=hashlib.sha256('\n'.join(clean.obs_names).encode()).hexdigest(), gene_sha256=hashlib.sha256('\n'.join(clean.var_names).encode()).hexdigest(), coordinate_sha256=ah(coords), donor_metadata=donor_info, preprocessing_annotation_policy='Reference stored separately; no reference columns in training AnnData; no score-based tuning.')
        rows.append(rec)
        print('G2_INPUT_READY', sample, clean.shape, flush=True)
    (RUN / 'input_manifest.json').write_text(json.dumps(rows, indent=2))
    print('G2_INPUT_PREPARATION_COMPLETE', len(rows), flush=True)

def train_benchmark_one(args):
    RUN = Path(args.run)
    OUT = RUN / args.phase / args.sample / args.method / ('seed' + str(args.seed))
    OUT.mkdir(parents=True, exist_ok=False)
    paths = setup_paths(RUN, args.method)
    torch.set_num_threads(4)
    seed_all(args.seed)
    assert os.environ['PYTHONHASHSEED'] == str(args.seed)
    start = time.monotonic()
    history = []
    model = None
    raw_backward = torch.Tensor.backward

    def traced_backward(self, *a, **kw):
        if self.numel() == 1:
            value = float(self.detach().cpu())
            assert np.isfinite(value), 'Nonfinite training loss'
            history.append(dict(step=len(history) + 1, loss=value))
        return raw_backward(self, *a, **kw)
    torch.Tensor.backward = traced_backward
    spec = next((r for r in json.loads((RUN / 'input_manifest.json').read_text()) if r['sample'] == args.sample))
    save(OUT / 'request.json', dict(sample=args.sample, method=args.method, seed=args.seed, smoke=args.smoke, input_manifest=spec, config_collection_sha256=sha(CONFIG_PATH), utility_sha256=sha(Path(__file__)), source_bundle_sha256=directory_hash(BASELINE_SOURCES), python=sys.version, torch=torch.__version__, scanpy=sc.__version__, device=args.device))
    if args.device.startswith('cuda'):
        torch.cuda.set_device(args.device)
        torch.cuda.reset_peak_memory_stats()
    try:
        a = sc.read_h5ad(RUN / 'inputs' / (args.sample + '.h5ad'))
        assert list(a.obs.columns) == []
        assert sha(RUN / 'inputs' / (args.sample + '.h5ad')) == spec['input_h5ad_sha256']
        family = spec['family']
        iscont = continuous(a)
        small = a.n_vars < 2000
        method = args.method
        dev = torch.device(args.device)
        budget = {}
        native = {}
        internal = None
        seed_all(42)
        if method == 'STWaveGraph':
            from STWaveGraph.STWaveGraph import STWaveGraph
            import STWaveGraph.STWaveGraph as swmod
            cfg = copy.deepcopy(CONFIG['benchmark']['samples'][args.sample])
            cfg.update(random_seed=args.seed, mclust_fallback_to_gmm=False)
            if small:
                a.var['highly_variable'] = np.var(dense(a.X), axis=0) > 0
            if iscont:
                swmod = importlib.import_module('STWaveGraph.STWaveGraph')

                def preprocess_continuous(adata, n_top_genes=2000, counts_layer=None):
                    x = adata.copy()
                    x.var['highly_variable'] = np.var(dense(x.X), axis=0) > 0
                    sc.pp.log1p(x)
                    sc.pp.scale(x, zero_center=True, max_value=10)
                    return x
                swmod.preprocess_adata = preprocess_continuous
            if args.smoke:
                cfg.update(pretrain_epochs=2, finetune_epochs=22)
            prep_cfg = dict(cfg, random_seed=42)
            t = STWaveGraph(a, device=dev, **prep_cfg)
            t.random_seed = args.seed
            seed_all(args.seed)
            t._build_model()
            t.fit(verbose=False)
            model = t.model
            fixture_args = (t.x, t.adj_local_t, t.eigvecs_t, t.eigvals_t)
            op = 'wave'
            index = None
            with torch.no_grad():
                internal = t.model.inference(*fixture_args)[3].argmax(1).cpu().numpy()
            budget = cfg
            native = dict(pca=cfg['cluster_pca_dim'], covariance=cfg['mclust_model'], refine='wave', refine_iters=cfg['refine_iters'], spatial_adj=t.adj_spatial_raw)
        elif method in ['GraphST', 'GAAEST']:
            module = importlib.import_module(method + '.' + method)
            cls = getattr(module, method)
            if iscont or small:
                a = select_genes(a, 3000)
                a = log_input(a, scale=True)
            kw = dict(device=dev, epochs=2 if args.smoke else 600, random_seed=args.seed)
            if method == 'GAAEST':
                kw['save_model_file'] = str(OUT / 'author_saved_model.pt')
                module.construct_interaction_KNN(a, n_neighbors=3)
                spatial_net(a, 3)
                ndf = a.uns['Spatial_Net']
                ids = {x: i for i, x in enumerate(a.obs_names)}
                directed = sp.csr_matrix((np.ones(len(ndf)), (ndf.Cell1.map(ids), ndf.Cell2.map(ids))), shape=(a.n_obs, a.n_obs))
                a.obsm['graph_neigh'] = directed.toarray()
                a.obsm['adj'] = directed.maximum(directed.T).toarray()
            t = cls(a, **kw)
            if method == 'GAAEST':
                observed = sp.csr_matrix((np.ones(t.data.edge_index.shape[1]), tuple(t.data.edge_index.cpu().numpy())), shape=(a.n_obs, a.n_obs))
                expected = (directed + sp.eye(a.n_obs)).astype(bool)
                assert (observed.astype(bool) != expected).nnz == 0
                save(OUT / 'graph_support_check.json', dict(passed=True, nonself_neighbors_min=int(np.asarray(directed.sum(1)).min()), nonself_neighbors_max=int(np.asarray(directed.sum(1)).max()), graph_neigh_and_GAT_edges_consistent=True, zero_distance_edges_retained=True))
            aa = t.train()
            model = t.model
            if method == 'GraphST':
                fixture_args = (t.features, t.features_a, t.adj)
            else:
                fixture_args = (t.features, t.features_a, t.data.edge_index)
            op = 'index'
            index = 1
            budget = kw
            native = dict(pca=20, covariance='EEE', refine='majority', neighbors=50 if method == 'GraphST' else 35)
        elif method == 'STAGATE':
            pkg = importlib.import_module('STAGATE_pyG')
            tm = importlib.import_module('STAGATE_pyG.Train_STAGATE')
            a = select_genes(a, 3000)
            a = log_input(a)
            spatial_net(a, 6)
            original = tm.STAGATE
            captured = []

            def factory(*aa, **kk):
                m = original(*aa, **kk)
                captured.append(m)
                return m
            tm.STAGATE = factory
            pkg.train_STAGATE(a, n_epochs=2 if args.smoke else 1000, random_seed=args.seed, device=dev, verbose=False)
            model = captured[0]
            data = tm.Transfer_pytorch_Data(a[:, a.var['highly_variable']]).to(dev)
            fixture_args = (data.x, data.edge_index)
            op = 'index'
            index = 0
            budget = dict(epochs=2 if args.smoke else 1000, hidden=[512, 30], lr=0.001, neighbors=6)
            save(OUT / 'graph_support_check.json', dict(passed=True, nonself_neighbors_min=int(a.uns['Spatial_Net'].groupby('Cell1').size().min()), nonself_neighbors_max=int(a.uns['Spatial_Net'].groupby('Cell1').size().max()), self_removed_by_index=True, zero_distance_edges_retained=True))
            native = dict(pca=0, covariance='EEE', refine='none')
        elif method == 'DeepGFT':
            u = importlib.import_module('DeepGFT.utils')
            tm = importlib.import_module('DeepGFT.train')
            gn = importlib.import_module('DeepGFT.genenet')
            preprocessed, cache_record = prepare_deepgft(RUN, args.sample, a, family)
            a = preprocessed['adata']
            gf, ge, sf, se = [preprocessed[k] for k in ['gf', 'ge', 'sf', 'se']]
            signal, middle, cluster, csvg, c1, c2 = [preprocessed[k] for k in ['signal', 'middle', 'cluster', 'csvg', 'c1', 'c2']]
            save(OUT / 'preprocessing_cache.json', cache_record)
            original = tm.SPOT
            captured = []

            def factory(*aa, **kk):
                m = original(*aa, **kk)
                captured.append(m)
                return m
            tm.SPOT = factory
            seed_all(args.seed)
            epochs = 2 if args.smoke else 700 if family == 'Breast' else 200 if family == 'STARmap' else 600
            _, _, native_emb, _, _ = tm.train_spot(a, gf, ge, sf, se, alpha=20, device=dev, epoch_max=epochs, cluster=cluster)
            model = captured[0]
            data = u.Transfer_pytorch_Data(a, dev, ['spotnet', 'spotnet_cluster'] if cluster else ['spotnet'])
            import torch_sparse
            edge = [torch_sparse.SparseTensor(row=d.edge_index[0], col=d.edge_index[1]) for d in data]
            fixture_args = tuple((torch.tensor(np.asarray(x), dtype=torch.float32, device=dev) for x in [gf, ge, sf, se])) + tuple(edge)
            op = 'index'
            index = 2
            budget = dict(epochs=epochs, alpha=20, signal=signal, middle=middle, cluster=cluster, csvg=csvg, c1=c1, c2=c2, neighbors=6, precluster_resolution=0.6, svg='author gft_top; all nonconstant genes for small/continuous panels')
            native = dict(pca=0, covariance='EEE', refine='majority' if family == 'DLPFC' else 'none', neighbors=30)
        elif method == 'SEDR':
            sedr = importlib.import_module('SEDR')
            if iscont or small:
                aa = log_input(select_genes(a, 2000))
                sc.pp.scale(aa)
                n = min(200, aa.n_vars - 1, aa.n_obs - 1)
                sc.pp.pca(aa, n_comps=n, random_state=42)
            else:
                aa = sedr.adata_preprocess(a, pca_n_comps=min(200, a.n_obs - 1, a.n_vars - 1))
            graph = sedr.graph_construction(aa, n=6, mode='KNN')
            seed_all(args.seed)
            t = sedr.Sedr(np.asarray(aa.obsm['X_pca']), graph, mode='clustering', device=dev)
            epochs = 2 if args.smoke else 200
            if args.smoke:
                original_pretrain = t.train_without_dec
                t.train_without_dec = lambda *aa, **kw: original_pretrain(epochs=2)
            t.train_with_dec(epochs=epochs)
            model = t.model
            fixture_args = (t.X, t.adj_norm)
            op = 'index'
            index = 0
            seed_all(args.seed)
            z, q, _, _ = t.process()
            internal = q.argmax(1)
            budget = dict(pretrain_epochs=epochs, dec_epochs=epochs, neighbors=6, pca=aa.obsm['X_pca'].shape[1], evaluation_seed=args.seed, stochastic_inference='Author encoding_mask_noise also runs in eval; freeze RNG before every recorded inference.')
            native = dict(pca=0, covariance='EEE', refine='none')
        elif method == 'DeepST':
            dt = importlib.import_module('deepstkit')
            dm = importlib.import_module('deepstkit.main')
            tr = importlib.import_module('deepstkit.trainer')
            d = dm.run(save_path=str(OUT / 'author_outputs'), pre_epochs=2 if args.smoke else 500, epochs=2 if args.smoke else 500, use_gpu=True)
            aa = d._get_augment(a.copy(), spatial_type='BallTree', use_morphological=False, n_components=min(100, a.n_vars - 1, a.n_obs - 1))
            graph = d._get_graph(a.obsm['spatial'], distType='KDTree')
            nc = min(200, a.n_vars - 1, a.n_obs - 1)
            if iscont:
                from sklearn.decomposition import PCA
                x = np.log1p(np.maximum(aa.obsm['augment_gene_data'], 0))
                x = (x - x.mean(0)) / (x.std(0) + 1e-08)
                data = PCA(n_components=nc, random_state=args.seed).fit_transform(x)
            else:
                data = d._data_process(aa, pca_n_comps=nc)
            seed_all(args.seed)
            model = dt.DeepST_model(input_dim=data.shape[1], Conv_type='GATConv', linear_encoder_hidden=[32, 20], linear_decoder_hidden=[32], conv_hidden=[32, 8], p_drop=0.01, dec_cluster_n=20)
            t = tr.train(data, graph, model, pre_epochs=d.pre_epochs, epochs=d.epochs, kl_weight=1, mse_weight=1, bce_kld_weight=1, domain_weight=1, use_gpu=True)
            t.fit()
            model = t.model
            fixture_args = (t.data, t.adj)
            op = 'index'
            index = 0
            z, q = t.process()
            internal = q.argmax(1)
            budget = dict(pretrain_epochs=d.pre_epochs, epochs=d.epochs, pca=nc, morphology=False)
            native = dict(pca=0, covariance='EEE', refine='majority', neighbors=50, native_clustering='Leiden-prior supplementary remains separate from common mclust')
        elif method == 'SpaCAE':
            pkg = importlib.import_module('SpaCAE')
            util = importlib.import_module('utils')
            a = select_genes(a, 2000)
            n = min(14, a.n_vars - 1, a.n_obs - 1)
            t = pkg.SpaCAE(a, num_cluster=spec['n_clusters'], input_dim=n, z_dim=n, output_dim=n, spacae_seed=args.seed, device=dev, n_epochs=2 if args.smoke else 40000)
            seed_all(42)
            aa, ori, aug, g = util.spatial_reconstruction(a, alpha=0.5, n_components=n, n_pcs=min(15, a.n_vars - 1, a.n_obs - 1), normalize_total=not iscont)
            seed_all(args.seed)
            t.train_SpaCAE(ori, aug, g)
            model = t.cgae
            fixture_args = tuple((torch.as_tensor(x, dtype=torch.float32, device=dev) for x in [ori, aug, g]))
            op = 'index'
            index = 0
            budget = dict(epochs=t.epochs, dim=n, alpha=0.5, lr=0.0001)
            native = dict(pca=0, covariance='EEE', refine='conditional', neighbors=10)
        elif method == '3dOT':
            u = importlib.import_module('lib_3d_OT.utils')
            tm = importlib.import_module('lib_3d_OT.single_modialty')
            a = select_genes(a, 3000)
            a = a[:, a.var['highly_variable']].copy()
            a = log_input(a, scale=True, target=None)
            a.obsm['feat'] = a.X
            a.obs['truth'] = 'UNUSED_CONSTANT_PLACEHOLDER'
            seed_all(args.seed)
            g = u.prepare_data(a, location='spatial', nb_neighbors=16).to(dev)
            model = tm.extractMODEL(args=None, input_dim=g.express.shape[-1]).to(dev)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            tm.train_graph_extractor(g, model, optimizer, dev, epochs=2 if args.smoke else 1150)
            fixture_args = (g,)
            op = 'decode'
            index = None
            budget = dict(epochs=2 if args.smoke else 1150, lr=0.001, neighbors=16, output='final reconstruction expression')
            native = dict(pca=30, covariance='EEE', refine='majority', neighbors=50)
        elif method == 'SpaGCN':
            sg = importlib.import_module('SpaGCN')
            from scipy.spatial.distance import cdist
            from functools import partial
            from sklearn.decomposition import PCA as FixedPCA
            importlib.import_module('SpaGCN.SpaGCN').PCA = partial(FixedPCA, random_state=42)
            a = select_genes(a, 3000)
            a = a[:, a.var['highly_variable']].copy()
            a = log_input(a)
            adj = cdist(a.obsm['spatial'], a.obsm['spatial'])
            l = sg.search_l(0.5, adj, start=0.01, end=float(adj.max()) + 1, tol=0.01, max_run=100)
            assert l is not None
            seed_all(args.seed)
            t = sg.SpaGCN()
            t.set_l(l)
            t.train(a, adj, num_pcs=min(50, a.n_vars - 1, a.n_obs - 1), init='kmeans', n_clusters=spec['n_clusters'], init_spa=True, max_epochs=2 if args.smoke else 200, lr=0.05, tol=0.005)
            model = t.model
            fixture_args = (torch.as_tensor(t.embed, dtype=torch.float32), torch.as_tensor(t.adj_exp, dtype=torch.float32))
            op = 'index'
            index = 0
            internal = t.predict()[0]
            budget = dict(max_epochs=2 if args.smoke else 200, lr=0.05, tol=0.005, init='kmeans', spatial_length_scale=float(l))
            native = dict(pca=0, covariance='EEE', refine='none', native_labels='internal')
        else:
            raise ValueError(method)
        model.eval()
        seed_all(args.seed)
        before = forward_fixture(model, fixture_args, op, index).detach().cpu().numpy().astype(np.float32)
        assert before.shape[0] == spec['shape'][0] and np.isfinite(before).all()
        if method == 'DeepGFT':
            assert np.allclose(before, native_emb, atol=0.0002, rtol=0.0002)
        states = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        assert all((torch.isfinite(v).all() for v in states.values()))
        model.load_state_dict(states, strict=True)
        seed_all(args.seed)
        after = forward_fixture(model, fixture_args, op, index).detach().cpu().numpy().astype(np.float32)
        diff = float(np.max(np.abs(before - after)))
        assert np.allclose(before, after, atol=0.0002, rtol=0.0002), (method, diff)
        np.savez_compressed(OUT / 'model_outputs.npz', embedding=before, **{'internal': internal} if internal is not None else {})
        native_small = {k: v for k, v in native.items() if k != 'spatial_adj'}
        if 'spatial_adj' in native:
            np.savez_compressed(OUT / 'native_spatial_graph.npz', adj=native['spatial_adj'])
        save(OUT / 'recorded_config.json', next((r for r in CONFIG['benchmark']['runs'] if r['sample'] == args.sample and r['method'] == args.method and (r['seed'] == args.seed))))
        save(OUT / 'config.json', dict(method=method, sample=args.sample, seed=args.seed, budget={k: str(v) if isinstance(v, (torch.device, Path)) else v for k, v in budget.items()}, native=native_small, input_kind=spec['input_kind'], smoke=args.smoke))
        save(OUT / 'history.json', history)
        constructor = None
        if method == '3dOT':
            constructor = dict(class_name='extractMODEL', input_dim=int(fixture_args[0].express.shape[-1]))
        elif method == 'SEDR':
            constructor = dict(class_name='SEDR_module', input_dim=int(fixture_args[0].shape[-1]))
        torch.save(dict(model=None if constructor else model, constructor=constructor, args=fixture_args, operation=op, index=index, source_method=method, state_dict=states, evaluation_seed=args.seed, config=json.loads((OUT / 'config.json').read_text())), OUT / 'checkpoint_with_fixture.pt')
        save(OUT / 'result.json', dict(status='training_complete', sample=args.sample, method=method, seed=args.seed, smoke=args.smoke, embedding_shape=list(before.shape), strict_reload_max_difference=diff, parameters=sum((p.numel() for p in model.parameters())), trainable_parameters=sum((p.numel() for p in model.parameters() if p.requires_grad)), optimizer_steps=len(history), elapsed_seconds=time.monotonic() - start, checkpoint_sha256=sha(OUT / 'checkpoint_with_fixture.pt'), peak_gpu_allocated_bytes=int(torch.cuda.max_memory_allocated()) if method != 'SpaGCN' and torch.cuda.is_available() else None))
        print('G2_TRAINING_COMPLETE', args.sample, method, args.seed, flush=True)
    except Exception as e:
        save(OUT / 'failure.json', dict(status='failed', sample=args.sample, method=args.method, seed=args.seed, smoke=args.smoke, error=repr(e), traceback=traceback.format_exc(), elapsed_seconds=time.monotonic() - start, completed_optimizer_steps=len(history)))
        save(OUT / 'history_partial.json', history)
        traceback.print_exc()
        sys.exit(1)

def evaluate_benchmark_one(a):
    R = Path(a.run)
    seed_all(a.seed)
    torch.set_num_threads(2)
    out = R / a.phase / a.sample / a.method / ('seed' + str(a.seed))
    ev = out / 'evaluation'
    ev.mkdir(exist_ok=False)
    spec = next((x for x in json.loads((R / 'input_manifest.json').read_text()) if x['sample'] == a.sample))
    cfg = json.loads((out / 'config.json').read_text())
    outputs = np.load(out / 'model_outputs.npz')
    emb = outputs['embedding']
    adata = sc.read_h5ad(R / 'inputs' / (a.sample + '.h5ad'))
    coords = np.asarray(adata.obsm['spatial'], dtype=np.float32)
    names = adata.obs_names.astype(str)
    assert emb.shape[0] == len(names) and np.isfinite(emb).all()
    pred = {}
    records = {}
    failures = {}

    def fit(stage, dim, cov):
        try:
            y, probs, centers, record = cluster_with_record(emb, spec['n_clusters'], method='mclust', seed=a.seed, pca_dim=dim, mclust_model=cov, allow_fallback=False, backend=cluster_embedding_fullinit)
            record['initialization'] = 'all observations; no subset, no fallback'
            save(ev / (stage + '_clustering.json'), record)
            records[stage] = record
            np.savez_compressed(ev / (stage + '_assignments.npz'), labels=y, probabilities=probs, centers=centers)
            return y
        except ClusterExecutionError as e:
            save(ev / (stage + '_failure.json'), dict(error=repr(e), record=e.record, traceback=traceback.format_exc()))
            failures[stage] = e.record
            return None

    def common_refine(y):
        g = graph_from_coords(coords, 6)
        classes = np.unique(y)
        oh = (y[:, None] == classes[None, :]).astype(np.float32)
        votes = g @ oh + oh
        ix = votes.argmax(1)
        for i in range(len(y)):
            tied = np.flatnonzero(votes[i] == votes[i].max())
            if len(tied) > 1:
                current = np.flatnonzero(classes == y[i])[0]
                if current in tied:
                    ix[i] = current
                else:
                    for j in g.indices[g.indptr[i]:g.indptr[i + 1]]:
                        jj = np.flatnonzero(classes == y[j])[0]
                        if jj in tied:
                            ix[i] = jj
                            break
        return classes[ix]

    def majority(y, k, conditional=False, include_self=False):
        k = min(k, len(y) - 1)
        d, ix = NearestNeighbors(n_neighbors=k + 1).fit(coords).kneighbors(coords)
        ans = np.asarray(y).copy()
        for i in range(len(y)):
            neigh = y[ix[i, 0 if include_self else 1:]]
            cl, counts = np.unique(neigh, return_counts=True)
            mx = counts.max()
            candidate = next((x for x in neigh if x in cl[counts == mx]))
            if not conditional or (np.count_nonzero(neigh == y[i]) < k / 2 and mx > k / 2):
                ans[i] = candidate
        return ans
    common_dim = min(20, emb.shape[1], len(emb) - 1)
    y = fit('common_external', common_dim, 'EEE')
    if y is not None:
        pred['common_external'] = y
        pred['common_refined'] = common_refine(y)
    if 'internal' in outputs:
        pred['internal'] = outputs['internal'].astype(int)
    elif 'model_q' in outputs:
        pred['internal'] = outputs['model_q'].argmax(1)
    native = cfg['native']
    dim = native.get('pca', 0) or None
    cov = native.get('covariance', 'EEE')
    if a.method == 'DeepST':
        try:
            np.save(ev / 'native_coordinates.npy', coords)
            ne = os.environ.copy()
            ne.pop('PYTHONPATH', None)
            with (ev / 'native_deepst.log').open('x') as log:
                pr = subprocess.run([os.environ.get('DEEPST_PYTHON', sys.executable), str(HERE / 'run_benchmark.py'), '--native-deepst', str(out), str(spec['n_clusters'])], env=ne, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            assert pr.returncode == 0, 'Native DeepST clustering exited ' + str(pr.returncode)
            yy = np.load(ev / 'native_deepst_predictions.npz')
            pred['native_external'] = yy['external']
            pred['native_refined'] = yy['refined']
        except Exception as e:
            save(ev / 'native_external_failure.json', dict(error=repr(e), traceback=traceback.format_exc()))
            failures['native_external'] = dict(error=repr(e))
    elif a.method == 'SpaGCN':
        if 'internal' in pred:
            pred['native_external'] = pred['internal'].copy()
            pred['native_refined'] = pred['native_external'].copy()
    else:
        same = cov == 'EEE' and (dim == common_dim or (dim is None and emb.shape[1] <= common_dim))
        if same:
            yn = pred.get('common_external')
            save(ev / 'native_external_clustering.json', dict(reuses='common_external', same_transformed_input=True, covariance=cov, pca_dim=dim))
        else:
            yn = fit('native_external', dim, cov)
        if yn is not None:
            pred['native_external'] = yn
            mode = native.get('refine', 'none')
            if mode == 'wave':
                from STWaveGraph.utils import refine_labels_by_spatial_neighbors
                adj = np.load(out / 'native_spatial_graph.npz')['adj']
                pred['native_refined'] = refine_labels_by_spatial_neighbors(yn, adj, n_iter=native['refine_iters'], self_weight=1)
            elif mode == 'none':
                pred['native_refined'] = yn.copy()
            else:
                pred['native_refined'] = majority(yn, native['neighbors'], conditional=mode == 'conditional')
    pd.DataFrame(pred, index=pd.Index(names, name='barcode')).to_csv(ev / 'predictions_unannotated.csv')
    if a.smoke:
        save(ev / 'smoke_integration.json', dict(prediction_stages=list(pred), actual_clusters={k: int(len(np.unique(v))) for k, v in pred.items()}, n_observations=len(names), failures=failures, reference_opened=False, actual_R_mclust=bool(records)))
        print('G2_SMOKE_EVALUATION', a.sample, a.method, flush=True)
        sys.exit(0)
    ref = pd.read_csv(R / 'inputs' / (a.sample + '_reference.csv'), index_col='barcode')
    assert list(ref.index.astype(str)) == list(names)
    valid = ref['valid_annotation'].to_numpy(dtype=bool)
    truth = ref['truth'].astype(str).to_numpy()
    edges, spacing = spatial_edges(coords, neighbors=6)
    rows = []
    domains = []
    for stage, yp in pred.items():
        ytrue = truth[valid]
        ypart = np.asarray(yp)[valid]
        u, ui = np.unique(ytrue, return_inverse=True)
        v, vi = np.unique(ypart, return_inverse=True)
        cm = np.zeros((len(u), len(v)), dtype=np.int64)
        np.add.at(cm, (ui, vi), 1)
        ri, ci = linear_sum_assignment(-cm)
        mapping = dict(zip(ri, ci))
        ious = []
        dices = []
        for i, c in enumerate(u):
            j = mapping.get(i)
            tp = int(cm[i, j]) if j is not None else 0
            nt = int(cm[i].sum())
            npred = int(cm[:, j].sum()) if j is not None else 0
            iou = tp / (nt + npred - tp)
            dice = 2 * tp / (nt + npred)
            ious.append(iou)
            dices.append(dice)
            domains.append(dict(sample=a.sample, method=a.method, seed=a.seed, stage=stage, domain=str(c), n_true=nt, n_matched_predicted=npred, IoU=iou, Dice=dice, recall=tp / nt))
        comb = lambda x: np.sum(x * (x - 1) / 2)
        tp = comb(cm)
        den = comb(cm.sum(0)) + comb(cm.sum(1)) - tp
        rows.append(dict(sample=a.sample, family=spec['family'], method=a.method, seed=a.seed, stage=stage, n_nodes=len(names), n_evaluated=int(valid.sum()), n_predicted_clusters=int(len(np.unique(yp))), ARI=float(adjusted_rand_score(ytrue, ypart)), NMI=float(normalized_mutual_info_score(ytrue, ypart, average_method='arithmetic')), pairwise_JI=float(tp / den) if den else None, matched_macro_IoU=float(np.mean(ious)), matched_macro_Dice=float(np.mean(dices)), **boundary_metrics(coords, truth, yp, valid, edges, spacing)))
    table = ref.copy()
    for k, v in pred.items():
        table[k] = v
    table.to_csv(ev / 'predictions.csv')
    save(ev / 'metrics.json', rows)
    save(ev / 'domain_metrics.json', domains)
    save(ev / 'evaluation_result.json', dict(sample=a.sample, method=a.method, seed=a.seed, status='complete' if 'common_external' in pred else 'primary_failed', stages=list(pred), failures=failures, primary='common_external', fallback_used=False, reference_sha256=sha(R / 'inputs' / (a.sample + '_reference.csv')), embedding_file_sha256=sha(out / 'model_outputs.npz')))
    print('G2_EVALUATION_COMPLETE', a.sample, a.method, a.seed, flush=True)

def native_deepst(output, clusters):
    o = Path(output)
    data = np.load(o / 'model_outputs.npz')
    emb = data['embedding']
    coords = np.load(o / 'evaluation/native_coordinates.npy')
    k = int(clusters)
    a = sc.AnnData(np.zeros((len(emb), 1), dtype=np.float32))
    a.obsm['emb'] = emb
    sc.pp.neighbors(a, use_rep='emb', random_state=0)
    search = []
    chosen = 1.0
    for resolution in sorted(np.arange(0.1, 2.5, 0.01), reverse=True):
        sc.tl.leiden(a, random_state=0, resolution=float(resolution), flavor='igraph', n_iterations=2, directed=False)
        nc = len(a.obs['leiden'].unique())
        search.append(dict(resolution=float(resolution), clusters=nc))
        if nc == k:
            chosen = float(resolution)
            break
    sc.tl.leiden(a, key_added='native', resolution=chosen, random_state=0, flavor='igraph', n_iterations=2, directed=False)
    y = a.obs['native'].astype(int).to_numpy()
    distance = cdist(coords, coords)
    refined = []
    for i in range(len(y)):
        ix = np.argsort(distance[i], kind='stable')[:7]
        neigh = y[ix]
        counts = pd.Series(neigh).value_counts()
        own = int(np.count_nonzero(neigh == y[i]))
        refined.append(int(counts.idxmax()) if own < 3 and int(counts.max()) > 3 else int(y[i]))
    np.savez_compressed(o / 'evaluation/native_deepst_predictions.npz', external=y, refined=np.asarray(refined))
    (o / 'evaluation/native_leiden_record.json').write_text(json.dumps(dict(requested_clusters=k, actual_clusters=int(len(np.unique(y))), resolution=chosen, search=search, scanpy=sc.__version__, python=sys.version, clustering_seed=0, labels_accessed=False, fallback=False, policy='Author prior-K resolution search, igraph Leiden two iterations, six-neighbor conditional refinement.'), indent=2))

def array_hash(x):
    return hashlib.sha256(x.tobytes()).hexdigest()

def evaluate_ablation(t, a, data, dataset, out, seed, cfg):
    import numpy as np, pandas as pd, torch
    import torch.nn.functional as F
    from STWaveGraph import utils
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
    from scipy.optimize import linear_sum_assignment
    t.model.eval()
    with torch.no_grad():
        h, z, rec, q = t.model.inference(t.x, t.adj_local_t, t.eigvecs_t, t.eigvals_t)
        emb = F.normalize(h, p=2, dim=1).cpu().numpy().astype(np.float32)
        q = q.cpu().numpy().astype(np.float32)
    np.savez_compressed(out / 'model_outputs.npz', embedding=emb, model_q=q)
    try:
        raw, probs, centers, record = cluster_with_record(emb, cfg['n_clusters'], method='mclust', seed=seed, pca_dim=cfg['cluster_pca_dim'], mclust_model=cfg['mclust_model'], allow_fallback=False, backend=cluster_embedding_fullinit)
    except ClusterExecutionError as e:
        save(out / 'clustering_record.json', e.record)
        raise
    record['initialization'] = 'all observations; no subset, no fallback'
    save(out / 'clustering_record.json', record)
    refined = utils.refine_labels_by_spatial_neighbors(raw, t.adj_spatial_raw, n_iter=cfg['refine_iters'], self_weight=1.0)
    pred = {'internal': q.argmax(1), 'external': raw, 'refined': refined}
    names = pd.Index(a.obs_names.astype(str), name='barcode')
    meta_path = data / 'metadata.tsv'
    meta = pd.read_csv(meta_path, sep='\t')
    key = {'DLPFC': 'barcode', 'Breast': 'ID', 'MouseBrain': 'Unnamed: 0'}[dataset]
    col = 'layer_guess' if dataset == 'DLPFC' else 'ground_truth'
    assert not meta[key].astype(str).duplicated().any()
    meta[key] = meta[key].astype(str)
    assert len(names.difference(meta[key])) == 0
    truth = meta.set_index(key)[col].reindex(names)
    valid = truth.notna().to_numpy()
    truth_values = truth.astype(object).fillna('__MISSING__').astype(str).to_numpy()
    coords = np.asarray(a.obsm['spatial'], dtype=np.float32)
    edges, spacing = spatial_edges(coords, neighbors=6)
    lib = np.asarray(a.X.sum(1)).ravel()
    low = lib <= np.quantile(lib, 0.1)
    table = pd.DataFrame(dict(truth=truth_values, valid_annotation=valid, x=coords[:, 0], y=coords[:, 1], library_size=lib, low_library_proxy=low, **pred), index=names)
    table.to_csv(out / 'predictions.csv')
    rows = []
    domains = []
    for stage, labels in pred.items():
        y = truth_values[valid]
        yp = np.asarray(labels)[valid]
        un_y, y_idx = np.unique(y, return_inverse=True)
        un_p, p_idx = np.unique(yp, return_inverse=True)
        cm = np.zeros((len(un_y), len(un_p)), dtype=np.int64)
        np.add.at(cm, (y_idx, p_idx), 1)
        ri, ci = linear_sum_assignment(-cm)
        mapping = {int(c): int(r) for r, c in zip(ri, ci)}
        mapped = np.array([mapping.get(int(c), -1) for c in p_idx])
        correct = mapped == y_idx
        for r, label in enumerate(un_y):
            matches = [c for c, rr in mapping.items() if rr == r]
            c = matches[0] if matches else None
            tp = int(cm[r, c]) if c is not None else 0
            ntrue = int(cm[r].sum())
            npred = int(cm[:, c].sum()) if c is not None else 0
            domains.append(dict(stage=stage, domain=str(label), n_true=ntrue, n_matched_predicted=npred, recall=tp / ntrue, IoU=tp / (ntrue + npred - tp), matched_cluster=str(un_p[c]) if c is not None else None))
        src, dst = edges.T
        interior = valid[src] & valid[dst] & (truth_values[src] == truth_values[dst])
        false_boundary = (labels[src] != labels[dst]) & interior
        counts = np.unique(labels, return_counts=True)[1]
        proportions = counts / counts.sum()
        row = dict(dataset=dataset, stage=stage, seed=seed, n_nodes=len(names), n_evaluated=int(valid.sum()), ARI=float(adjusted_rand_score(y, yp)), NMI=float(normalized_mutual_info_score(y, yp)), matched_error=float(1 - correct.mean()), low_library_error=float(1 - correct[low[valid]].mean()), other_library_error=float(1 - correct[~low[valid]].mean()), n_interior_edges=int(interior.sum()), n_false_interior_boundaries=int(false_boundary.sum()), interior_false_boundary_rate=float(false_boundary.sum() / interior.sum()), cluster_sizes=counts.tolist(), cluster_entropy=float(-(proportions * np.log(proportions)).sum()), **boundary_metrics(coords, truth_values, labels, valid, edges, spacing))
        rows.append(row)
    save(out / 'metrics.json', rows)
    save(out / 'domain_metrics.json', domains)
    save(out / 'annotation_alignment.json', dict(file=str(meta_path), sha256=sha(meta_path), key=key, column=col, barcode_sha256=hashlib.sha256('\n'.join(names).encode()).hexdigest(), n_valid=int(valid.sum()), policy='annotations accessed after predictions; library size decile is a QC proxy only'))
    return rows

def run_seed(args):
    import numpy as np, torch, scanpy as sc
    from STWaveGraph.STWaveGraph import STWaveGraph
    from STWaveGraph.preprocess import fix_seed, spectral_decompose_laplacian_hybrid, reweight_spatial_graph
    run = Path(args.run)
    seed = args.seed
    dataset = args.dataset
    assert os.environ['PYTHONHASHSEED'] == str(seed)
    torch.set_num_threads(4)
    sources = {p.name: sha(p) for p in (ROOT / 'STWaveGraph').glob('*.py')}
    source_cp = CONFIG_PATH
    cfg = copy.deepcopy(CONFIG['ablation']['base'][dataset + '/seed' + str(seed)])
    cfg.update(random_seed=seed, mclust_fallback_to_gmm=False)
    rel = {'DLPFC': 'Data/DLPFC/151673', 'Breast': 'Data/Human_Breast_Cancer', 'MouseBrain': 'Data/Mouse_Brain_Anterior'}[dataset]
    data = ROOT / rel
    raw = [data / 'filtered_feature_bc_matrix.h5', data / 'metadata.tsv'] + list((data / 'spatial').glob('*positions*.csv'))
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in raw}
    seed_dir = run / 'formal' / dataset / ('seed' + str(seed))
    if args.smoke:
        seed_dir = run / args.smoke_name / dataset / ('seed' + str(seed))
    seed_dir.mkdir(parents=True, exist_ok=False)
    save(seed_dir / 'base_config.json', cfg)
    save(seed_dir / 'provenance.json', dict(source_hashes=sources, data_hashes=hashes, config_source_checkpoint=str(source_cp), config_source_checkpoint_sha256=sha(source_cp), python=sys.version, torch=torch.__version__, scanpy=sc.__version__, device=args.device, config_collection_sha256=sha(CONFIG_PATH), utility_sha256=sha(Path(__file__))))
    a = sc.read_visium(data, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    a.var_names_make_unique()
    assert a.obs_names.is_unique
    vals = a.X.data if hasattr(a.X, 'data') else np.asarray(a.X).ravel()
    input_audit = dict(shape=list(a.shape), min=float(vals.min()), max=float(vals.max()), noninteger_fraction=float(np.mean(np.abs(vals - np.round(vals)) > 1e-05)), finite=bool(np.isfinite(vals).all()), coordinates_finite=bool(np.isfinite(a.obsm['spatial']).all()))
    save(seed_dir / 'input_audit.json', input_audit)
    assert input_audit['min'] >= 0 and input_audit['noninteger_fraction'] == 0 and input_audit['finite'] and input_audit['coordinates_finite']
    t = STWaveGraph(a, device=torch.device(args.device), **cfg)
    U0 = t.eigvecs_t.clone()
    L0 = t.eigvals_t.clone()
    A0 = t.adj_wave_raw.copy()
    fix_seed(seed)
    noexpr = reweight_spatial_graph(t.adj_spatial_raw, t.x_pca, expr_power=0, morph_power=t.morph_power)
    le, ue = spectral_decompose_laplacian_hybrid(noexpr, k=t.spectral_k, low_ratio=t.spectral_low_ratio)
    fix_seed(seed)
    ll, ul = spectral_decompose_laplacian_hybrid(A0, k=t.spectral_k, low_ratio=1.0)
    cache = seed_dir / 'shared_inputs.pt'
    torch.save(dict(x=t.x.cpu(), adj_local=t.adj_local_t.cpu(), adj_spatial=t.adj_spatial_raw, adj_wave=A0, eigenvalues=L0.cpu(), eigenvectors=U0.cpu(), expr_off_eigenvalues=le, expr_off_eigenvectors=ue, low_only_eigenvalues=ll, low_only_eigenvectors=ul, observation_names=list(a.obs_names), config=cfg), cache)
    cache_hash = sha(cache)
    save(seed_dir / 'graph_interventions.json', dict(x_sha256=array_hash(t.x_pca), spatial_sha256=array_hash(t.adj_spatial_raw), wave_sha256=array_hash(A0), expr_off_sha256=array_hash(noexpr), common_spatial_support=True, expr_off_equals_spatial_up_to_global_scale=bool(np.allclose(noexpr, t.adj_spatial_raw / (t.adj_spatial_raw.max() + 1e-08))), spectral_budget=int(len(L0)), base_eigenvalues=L0.cpu().tolist(), expr_off_eigenvalues=le.tolist(), low_only_eigenvalues=ll.tolist(), cache_sha256=cache_hash))
    t.eigvecs_t = U0
    t.eigvals_t = L0
    fix_seed(seed)
    t._build_model()
    init = {k: v.detach().cpu().clone() for k, v in t.model.state_dict().items()}
    variants = VARIANTS if not args.smoke else ['wave', 'heat', 'cheb5', 'pde_d4', 'velocity_zero', 'fusion_concat']
    if args.variant_list:
        assert args.smoke, 'Subset selection is only allowed for integration checks'
        variants = args.variant_list.split(',')
        assert set(variants).issubset(VARIANTS)
    summary = []
    for variant in variants:
        out = seed_dir / variant
        out.mkdir(exist_ok=False)
        start = time.monotonic()
        try:
            for k, v in cfg.items():
                setattr(t, k, v)
            t.eigvecs_t = U0
            t.eigvals_t = L0
            t.adj_wave_raw = A0
            if variant == 'expr_off':
                t.expr_power = 0
                t.adj_wave_raw = noexpr
                t.eigvecs_t = torch.tensor(ue, device=t.device)
                t.eigvals_t = torch.tensor(le, device=t.device)
            elif variant == 'low_only':
                t.spectral_low_ratio = 1.0
                t.eigvecs_t = torch.tensor(ul, device=t.device)
                t.eigvals_t = torch.tensor(ll, device=t.device)
            elif variant == 'boundary_off':
                t.lambda_boundary = 0
            elif variant == 'balance_toggle':
                t.lambda_balance = 0.05 if cfg['lambda_balance'] == 0 else 0.0
            if args.smoke:
                t.pretrain_epochs = 2
                t.finetune_epochs = 22
            fix_seed(seed)
            t._build_model()
            assert all((torch.equal(v.detach().cpu(), init[k]) for k, v in t.model.state_dict().items()))
            capacity = configure_model(t, variant)
            current = t.model.state_dict()
            compared = [k for k in init if k in current and (not k.startswith('encoder.wave_layer.'))]
            assert all((torch.equal(current[k].cpu(), init[k]) for k in compared))
            actual = dict(cfg, lambda_boundary=t.lambda_boundary, lambda_balance=t.lambda_balance, expr_power=t.expr_power, spectral_low_ratio=t.spectral_low_ratio, n_bands=int(variant[-1]) if variant in ['bands1', 'bands2'] else cfg['n_bands'], pretrain_epochs=t.pretrain_epochs, finetune_epochs=t.finetune_epochs)
            save(out / 'config.json', dict(variant=variant, config=actual, capacity=capacity, identical_shared_initial_tensors=len(compared), seed=seed, cache=str(cache), cache_sha256=cache_hash, config_source_checkpoint=str(source_cp), formal=not args.smoke, zero_hyperparameter_search=True))
            fix_seed(seed)
            if t.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(t.device)
            hist = t.fit(verbose=False)
            save(out / 'history.json', hist)
            assert all((np.isfinite(r['loss']) for section in hist.values() for r in section))
            trained = dict(state_dict={k: v.detach().cpu().clone() for k, v in t.model.state_dict().items()}, variant=variant, config=actual, shared_cache=str(cache), shared_cache_sha256=cache_hash, source_hashes=sources, seed=seed)
            torch.save(trained, out / 'trained_checkpoint.pt')
            t.model.eval()
            with torch.no_grad():
                before = t.model.inference(t.x, t.adj_local_t, t.eigvecs_t, t.eigvals_t)[0].clone()
            t.model.load_state_dict(torch.load(out / 'trained_checkpoint.pt', map_location=t.device, weights_only=False)['state_dict'], strict=True)
            with torch.no_grad():
                after = t.model.inference(t.x, t.adj_local_t, t.eigvecs_t, t.eigvals_t)[0]
            assert torch.equal(before, after)
            if variant in ['wave', 'velocity_zero']:
                from STWaveGraph.model import damped_wave_transfer
                with torch.no_grad():
                    e = t.model.encoder
                    h0 = e.input_proj(t.x)
                    v0 = e.velocity_proj(h0)
                    w = e.wave_layer.original if hasattr(e.wave_layer, 'original') else e.wave_layer
                    uh = t.eigvecs_t.T @ h0
                    vh = t.eigvecs_t.T @ v0
                    gh = torch.zeros_like(t.eigvals_t)
                    gv = torch.zeros_like(gh)
                    for b, m in enumerate(w._band_masks(t.eigvals_t)):
                        gh[m], gv[m] = damped_wave_transfer(t.eigvals_t[m], w._positive(w.raw_alpha)[b].clamp(0.001, 2), w._positive(w.raw_beta)[b].clamp(0.001, 5), w._positive(w.raw_time)[b].clamp(0.001, 3))
                    np.savez_compressed(out / 'transfer_response.npz', eigenvalues=t.eigvals_t.cpu().numpy(), gain_h=gh.cpu().numpy(), gain_v=gv.cpu().numpy(), state_energy=(gh[:, None] * uh).square().sum(1).cpu().numpy(), effective_velocity_energy=((gv[:, None] * vh).square().sum(1) * (0 if variant == 'velocity_zero' else 1)).cpu().numpy())
            rows = evaluate_ablation(t, a, data, dataset, out, seed, actual)
            result = dict(status='complete', dataset=dataset, variant=variant, seed=seed, smoke=args.smoke, elapsed_seconds=time.monotonic() - start, capacity=capacity, checkpoint_sha256=sha(out / 'trained_checkpoint.pt'), peak_gpu_allocated_bytes=int(torch.cuda.max_memory_allocated(t.device)) if t.device.type == 'cuda' else None, strict_checkpoint_reload_equal=True, metrics=rows)
            save(out / 'result.json', result)
            save(out / 'manifest.json', {p.name: sha(p) for p in out.iterdir() if p.is_file()})
            summary.append(dict(variant=variant, status='complete', elapsed_seconds=result['elapsed_seconds']))
            print('ABLATION_COMPLETE', dataset, seed, variant, result['elapsed_seconds'], flush=True)
        except Exception:
            save(out / 'failure.json', dict(status='failed', dataset=dataset, seed=seed, variant=variant, traceback=traceback.format_exc()))
            summary.append(dict(variant=variant, status='failed'))
            traceback.print_exc()
        save(seed_dir / 'seed_summary.json', summary)
        if args.smoke and summary[-1]['status'] == 'failed':
            raise RuntimeError('Smoke failed; formal queue prohibited')
    assert sources == {p.name: sha(p) for p in (ROOT / 'STWaveGraph').glob('*.py')}
    if any((r['status'] != 'complete' for r in summary)):
        raise RuntimeError('One or more ablation variants failed; see seed_summary.json')
    print('SEED_COMPLETE', dataset, seed, json.dumps(summary), flush=True)

def prepare_sensitivity(seed, device, parameter, value):
    import numpy as np, torch, scanpy as sc
    from STWaveGraph.STWaveGraph import STWaveGraph
    from STWaveGraph.preprocess import fix_seed, construct_st_graphs, preprocess_adj_dense, spectral_decompose_laplacian_hybrid
    from STWaveGraph.utils import build_edge_index_from_adj
    from STWaveGraph.model import SpectralGraphWaveLayer
    base = OLD / 'formal/DLPFC' / ('seed' + str(seed))
    cache = base / 'shared_inputs.pt'
    cfg = json.loads((base / 'base_config.json').read_text())
    shared = torch.load(cache, map_location='cpu', weights_only=False)
    assert sha(cache) == json.loads((base / 'graph_interventions.json').read_text())['cache_sha256']
    for rel, digest in json.loads((base / 'provenance.json').read_text())['data_hashes'].items():
        assert sha(ROOT / rel) == digest, rel
    cfg[parameter] = value if parameter != 'none' else cfg.get(parameter)
    if parameter == 'none':
        cfg.pop('none')
    data = ROOT / 'Data/DLPFC/151673'
    a = sc.read_visium(data, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    a.var_names_make_unique()
    assert list(a.obs_names) == shared['observation_names']
    t = STWaveGraph.__new__(STWaveGraph)
    for k, v in cfg.items():
        setattr(t, k, v)
    t.device = torch.device(device)
    t.x = shared['x'].to(t.device)
    t.x_pca = shared['x'].numpy()
    t.adata = a
    t.adj_spatial_raw = shared['adj_spatial']
    t.adj_wave_raw = shared['adj_wave']
    t.adj_local_t = shared['adj_local'].to(t.device)
    U = shared['eigenvectors']
    lam = shared['eigenvalues']
    graph_changed = parameter in ['n_neighbors_spatial', 'expr_power']
    if parameter == 'expr_power' and value == 0:
        U = shared['expr_off_eigenvectors']
        lam = shared['expr_off_eigenvalues']
    elif parameter == 'spectral_low_ratio' and value == 1:
        U = shared['low_only_eigenvectors']
        lam = shared['low_only_eigenvalues']
    elif parameter in ['n_neighbors_spatial', 'expr_power', 'spectral_k', 'spectral_low_ratio']:
        fix_seed(seed)
        if graph_changed:
            g = construct_st_graphs(a, t.x_pca, n_neighbors_spatial=t.n_neighbors_spatial, use_morphology=t.use_morphology, morph_key=t.morph_key, expr_power=t.expr_power, morph_power=t.morph_power)
            t.adj_wave_raw = g['adj_wave']
            if parameter == 'n_neighbors_spatial':
                t.adj_spatial_raw = g['adj_spatial']
                t.adj_local_t = torch.as_tensor(preprocess_adj_dense(t.adj_spatial_raw, add_self_loop=True), device=t.device, dtype=torch.float32)
        fix_seed(seed)
        lam, U = spectral_decompose_laplacian_hybrid(t.adj_wave_raw, k=t.spectral_k, low_ratio=t.spectral_low_ratio)
    t.eigvecs_t = torch.as_tensor(U, device=t.device, dtype=torch.float32)
    t.eigvals_t = torch.as_tensor(lam, device=t.device, dtype=torch.float32)
    edges, weights = build_edge_index_from_adj(t.adj_spatial_raw, threshold=0, add_self_loop=False)
    t.edge_index_t = edges.to(t.device)
    t.edge_weight_t = weights.to(t.device)
    fix_seed(seed)
    original_bands = json.loads((base / 'base_config.json').read_text())['n_bands']
    t.n_bands = original_bands
    t._build_model()
    initial = {k: v.detach().cpu().clone() for k, v in t.model.state_dict().items()}
    if parameter == 'n_bands':
        t.n_bands = int(value)
        t.model.encoder.wave_layer = SpectralGraphWaveLayer(t.hidden_dim, n_bands=int(value)).to(t.device)
        t.optimizer = torch.optim.AdamW(t.model.parameters(), lr=t.lr, weight_decay=t.weight_decay)
    compared = [k for k in initial if not k.startswith('encoder.wave_layer.')]
    assert all((torch.equal(initial[k], t.model.state_dict()[k].cpu()) for k in compared))
    return (t, a, data, cfg, cache, dict(initial_tensors_compared=len(compared), base_cache_sha256=sha(cache)))

def run_sensitivity_one(args):
    import numpy as np, torch
    from STWaveGraph.preprocess import fix_seed
    group = Path(args.run)
    point = CONFIG['sensitivity']['unique_points'][args.point]
    seed = args.seed
    assert os.environ['PYTHONHASHSEED'] == str(seed)
    torch.set_num_threads(4)
    out = group / ('validation' if args.mode != 'formal' else 'sensitivity') / (args.mode + '_' + point['id'] if args.mode != 'formal' else point['id']) / ('seed' + str(seed))
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    try:
        expected = json.loads((OLD / 'formal/DLPFC' / ('seed' + str(seed)) / 'provenance.json').read_text())['source_hashes']
        actual = {p.name: sha(p) for p in (ROOT / 'STWaveGraph').glob('*.py')}
        assert expected == actual
        t, a, data, cfg, cache, checks = prepare_sensitivity(seed, args.device, point['parameter'], point['value'])
        if args.mode == 'smoke':
            t.pretrain_epochs = 2
            t.finetune_epochs = 22
            cfg.update(pretrain_epochs=2, finetune_epochs=22)
        cfg['random_seed'] = seed
        save(out / 'config.json', dict(config=cfg, point=point, shared_inputs=str(cache), checks=checks, source_hashes=actual, zero_search=True))
        torch.save(dict(eigenvectors=t.eigvecs_t.cpu(), eigenvalues=t.eigvals_t.cpu(), adj_spatial=t.adj_spatial_raw if point['parameter'] == 'n_neighbors_spatial' else None, adj_local=t.adj_local_t.cpu() if point['parameter'] == 'n_neighbors_spatial' else None), out / 'cache_patch.pt')
        save(out / 'spectral_diagnostics.json', dict(k=int(len(t.eigvals_t)), requested_k=t.spectral_k, low_ratio=t.spectral_low_ratio, eigenvalues=t.eigvals_t.cpu().tolist(), spatial_sha256=hashlib.sha256(t.adj_spatial_raw.tobytes()).hexdigest(), base_cache_sha256=checks['base_cache_sha256'], residual='retained: learned scale times (I-UU^T)H0'))
        fix_seed(seed)
        hist = t.fit(verbose=False)
        save(out / 'history.json', hist)
        assert len(hist['pretrain']) == t.pretrain_epochs and len(hist['finetune']) == t.finetune_epochs
        assert all((np.isfinite(r['loss']) for sec in hist.values() for r in sec))
        torch.save(dict(state_dict={k: v.detach().cpu() for k, v in t.model.state_dict().items()}, config=cfg, source_hashes=actual, shared_inputs=str(cache), shared_inputs_sha256=sha(cache), cache_patch_sha256=sha(out / 'cache_patch.pt'), point=point, seed=seed), out / 'trained_checkpoint.pt')
        rows = evaluate_ablation(t, a, data, 'DLPFC', out, seed, cfg)
        save(out / 'result.json', dict(status='complete', point=point, seed=seed, metrics=rows, elapsed_seconds=time.monotonic() - start, checkpoint_sha256=sha(out / 'trained_checkpoint.pt'), pretrain_epochs=len(hist['pretrain']), finetune_epochs=len(hist['finetune'])))
        if args.mode == 'reference':
            old = np.load(OLD / 'formal/DLPFC' / ('seed' + str(seed)) / 'wave/model_outputs.npz')
            new = np.load(out / 'model_outputs.npz')
            diffs = {k: float(np.max(np.abs(old[k] - new[k]))) for k in ['embedding', 'model_q']}
            save(out / 'reference_retrain_check.json', dict(differences=diffs, tolerance=2e-05, passed=all((x <= 2e-05 for x in diffs.values()))))
            assert all((x <= 2e-05 for x in diffs.values())), diffs
        save(out / 'manifest.json', {p.name: sha(p) for p in out.iterdir() if p.is_file()})
        print('SENSITIVITY_COMPLETE', point['id'], seed, flush=True)
    except Exception:
        save(out / 'failure.json', dict(point=point, seed=seed, traceback=traceback.format_exc()))
        traceback.print_exc()
        raise

def job_plan(kind, cfg, samples=None, methods=None, seeds=None, points=None):
    seeds=[int(x) for x in (seeds or '42,43,44,45,46').split(',')]
    if not seeds or len(set(seeds))!=len(seeds) or not set(seeds)<=set(range(42,47)):
        raise ValueError('Choose unique recorded seeds from 42 through 46')
    if kind=='benchmark':
        ss=(samples or ','.join(cfg['benchmark']['samples'])).split(',')
        mm=(methods or ','.join(METHODS)).split(',')
        if not set(ss)<=set(cfg['benchmark']['samples']) or not set(mm)<=set(METHODS):raise ValueError('Unknown sample or method')
        return [dict(sample=s,method=m,seed=seed) for seed in seeds for s in ss for m in mm]
    if kind=='ablation':
        ss=(samples or 'DLPFC,Breast,MouseBrain').split(',')
        if not set(ss)<={'DLPFC','Breast','MouseBrain'}:raise ValueError('Unknown ablation dataset')
        return [dict(dataset=s,seed=seed) for seed in seeds for s in ss]
    pp=cfg['sensitivity']['unique_points']
    ids=list(range(len(pp))) if not points else [i for i,p in enumerate(pp) if p['id'] in points.split(',')]
    if points and {pp[i]['id'] for i in ids}!=set(points.split(',')):raise ValueError('Unknown sensitivity point')
    return [dict(point=i,point_name=pp[i]['id'],seed=seed) for seed in seeds for i in ids]

def collect_metrics(kind, run):
    import csv
    run=Path(run);rows=[]
    if kind=='benchmark':
        for p in sorted(run.glob('formal/*/*/seed*/evaluation/metrics.json')):rows.extend(json.loads(p.read_text()))
    else:
        pattern='formal/*/seed*/*/result.json' if kind=='ablation' else 'sensitivity/*/seed*/result.json'
        for p in sorted(run.glob(pattern)):
            result=json.loads(p.read_text())
            for row in result.get('metrics',[]):
                row=dict(row)
                if kind=='ablation':row['variant']=result['variant']
                else:row['point']=result['point']['id']
                rows.append(row)
    if rows:
        fields=sorted({k for r in rows for k in r})
        with (run/'metrics_all.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    return len(rows)

def cli(kind):
    global CONFIG,CONFIG_PATH,BASELINE_SOURCES,OLD,np,pd,sc,cdist
    p=argparse.ArgumentParser(description=kind+' experiment runner; use --dry-run to inspect the job list without scientific dependencies')
    p.add_argument('--configs',default=str(HERE/'configs.json'))
    p.add_argument('--root',default=str(HERE.parent/'STWaveGraph-main'),help='Directory containing STWaveGraph/ and Data/')
    p.add_argument('--run',default=str(HERE/'outputs'/kind),help='Output directory; existing job outputs are never overwritten')
    p.add_argument('--samples',help='Comma-separated sample names; for ablations use DLPFC,Breast,MouseBrain')
    p.add_argument('--seeds',default='42,43,44,45,46')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--single',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--smoke',action='store_true',help='Short integration run, never a paper result')
    if kind=='benchmark':
        p.add_argument('--methods')
        p.add_argument('--author-sources',help='Parent of the upstream baseline source folders')
        p.add_argument('--prepare',action='store_true',help='Prepare 17 inputs from Data/ without training')
        p.add_argument('--evaluate-only',action='store_true')
        p.add_argument('--native-deepst',nargs=2,metavar=('OUTPUT','CLUSTERS'),help=argparse.SUPPRESS)
    elif kind=='sensitivity':
        p.add_argument('--points',help='Comma-separated parameter point IDs; omit to run all')
        p.add_argument('--ablation-run',default=str(HERE/'outputs/ablation'))
    a=p.parse_args()
    if kind=='benchmark' and a.native_deepst:
        import numpy as np, pandas as pd, scanpy as sc
        from scipy.spatial.distance import cdist
        native_deepst(*a.native_deepst);return
    CONFIG_PATH=Path(a.configs).resolve();CONFIG=load_configs(CONFIG_PATH)
    try:jobs=job_plan(kind,CONFIG,a.samples,getattr(a,'methods',None),a.seeds,getattr(a,'points',None))
    except ValueError as e:p.error(str(e))
    if a.dry_run:
        print(json.dumps({'experiment':kind,'job_count':len(jobs),'config_sha256':sha(CONFIG_PATH),'jobs':jobs},indent=2));return
    a.root=str(Path(a.root).resolve());a.run=str(Path(a.run).resolve())
    if kind=='benchmark' and a.prepare:
        runtime(a.root);prepare_benchmark_inputs(a.run);return
    if not a.single:
        run=Path(a.run);run.mkdir(parents=True,exist_ok=True)
        if (run/'jobs.json').exists():p.error('This output directory already has a job plan; choose a new --run directory or inspect the previous run.')
        save(run/'jobs.json',dict(kind=kind,smoke=a.smoke,jobs=jobs,config_sha256=sha(CONFIG_PATH)))
        status=[]
        for job in jobs:
            interpreter=sys.executable
            if kind=='benchmark':interpreter=os.environ.get('STWAVEGRAPH_PYTHON' if job['method']=='STWaveGraph' else 'BASELINE_PYTHON',sys.executable)
            command=[interpreter,str(HERE/('run_'+kind+'.py')),'--single','--configs',str(CONFIG_PATH),'--root',a.root,'--run',a.run,'--seeds',str(job['seed']),'--device',a.device]
            if kind=='benchmark':
                command+=['--samples',job['sample'],'--methods',job['method']]
                if a.author_sources:command+=['--author-sources',str(Path(a.author_sources).resolve())]
                if a.evaluate_only:command+=['--evaluate-only']
            elif kind=='ablation':command+=['--samples',job['dataset']]
            else:command+=['--points',job['point_name'],'--ablation-run',str(Path(a.ablation_run).resolve())]
            if a.smoke:command+=['--smoke']
            env=dict(os.environ,PYTHONHASHSEED=str(job['seed']),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            result=subprocess.run(command,env=env)
            status.append(dict(job=job,exit_code=result.returncode));save(run/'job_status.json',status)
        count=collect_metrics(kind,run)
        print(json.dumps({'jobs':len(status),'failed':sum(x['exit_code']!=0 for x in status),'metric_rows':count}))
        if any(x['exit_code'] for x in status):raise SystemExit(1)
        return
    if len(jobs)!=1:p.error('--single requires exactly one job')
    job=jobs[0]
    if os.environ.get('PYTHONHASHSEED')!=str(job['seed']):p.error('Worker PYTHONHASHSEED mismatch; use the normal batch entry.')
    runtime(a.root)
    if kind=='benchmark':
        BASELINE_SOURCES=Path(a.author_sources).resolve() if a.author_sources else Path(a.run)/'author_sources'
        if job['method']!='STWaveGraph' and not BASELINE_SOURCES.is_dir():raise FileNotFoundError('Provide --author-sources for baseline training')
        args=SimpleNamespace(**job,run=a.run,phase='smoke' if a.smoke else 'formal',device=a.device,smoke=a.smoke)
        if not a.evaluate_only:train_benchmark_one(args)
        evaluate_benchmark_one(args)
        if not a.smoke:
            state=json.loads((Path(a.run)/args.phase/args.sample/args.method/('seed'+str(args.seed))/'evaluation/evaluation_result.json').read_text())
            if state['status']!='complete':raise RuntimeError('Primary clustering failed; inspect evaluation outputs')
    elif kind=='ablation':
        run_seed(SimpleNamespace(**job,run=a.run,device=a.device,smoke=a.smoke,smoke_name='smoke',variant_list=None))
    else:
        OLD=Path(a.ablation_run).resolve()
        run_sensitivity_one(SimpleNamespace(run=a.run,point=job['point'],seed=job['seed'],device=a.device,mode='smoke' if a.smoke else 'formal'))
