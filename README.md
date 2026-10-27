# STWaveGraph: Spatial Domain Identification in Spatial Transcriptomics via Multi-Band Damped Graph Wave Propagation

## Overview

Spatial transcriptomics (ST) enables transcriptome-wide molecular profiling while preserving the spatial organization of tissue sections. Accurately identifying spatial domains is a fundamental task for understanding tissue architecture, cellular neighborhoods, and disease-associated microenvironments. However, existing graph-based methods often rely on repeated neighborhood aggregation, which may behave like diffusion-like smoothing and blur sharp molecular transitions at anatomical or pathological boundaries.

**STWaveGraph** is a frequency-adaptive graph spectral learning framework designed for robust and boundary-aware spatial domain identification in spatial transcriptomics. It features:

- **Expression-Reweighted Wave Graph:** Constructs spatial graphs from physical coordinates and reweights spatial edges using transcriptomic similarity, reducing potential cross-boundary signal mixing between transcriptionally dissimilar neighboring spots or cells.
- **Multi-Band Damped Graph Wave Propagation:** Introduces a graph spectral propagation module based on damped wave dynamics. Unlike conventional diffusion-like graph aggregation, STWaveGraph allows retained spectral components to follow learnable band-specific propagation behaviors, capturing both smooth tissue-level organization and localized spatial transitions.
- **Residual Local Graph Encoder:** Preserves short-range spatial continuity and neighborhood-level coherence through a residual graph convolutional branch, complementing the spectral wave branch.
- **Adaptive Spectral-Local Fusion:** Combines the spectral wave representation, local graph representation, and initial latent features through a learnable fusion mechanism to balance global graph spectral structure and local spatial consistency.
- **Boundary-Aware Clustering Refinement:** Optimizes spatial domain embeddings through representation pretraining and clustering-oriented fine-tuning, including reconstruction, cross-view consistency, spatial coherence, clustering alignment, cluster balance, and boundary-aware regularization.

![overview](STWaveGraph-main/Overview/overview.png)

## Installation

We recommend using [Conda](https://docs.conda.io/en/latest/) to manage your environment. STWaveGraph is implemented in Python and relies on PyTorch, Scanpy, Anndata, Rpy2, and common scientific computing packages.

```python
# 1. Clone the repository
git clone https://github.com/JiruiZhang/STWaveGraph.git
cd STWaveGraph

# 2. Create a conda environment
conda create -n stwavegraph_env python=3.8
conda activate stwavegraph_env

# 3. Install PyTorch
# Please install the PyTorch version according to your CUDA/cuDNN environment.
# The following command is an example for CUDA 10.2.
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

# Or install PyTorch by pip
pip install torch>=1.8.0

# 4. Install required Python packages
pip install numpy==1.22.3
pip install scanpy==1.9.1
pip install anndata==0.8.0
pip install rpy2==3.4.1
pip install pandas==1.4.2
pip install scipy==1.8.1
pip install scikit-learn==1.1.1
pip install tqdm==4.64.0
pip install matplotlib==3.4.2
```

# Requirements

- python == 3.8
- torch >= 1.8.0
- cudnn >= 10.2
- numpy == 1.22.3
- scanpy == 1.9.1
- anndata == 0.8.0
- rpy2 == 3.4.1
- pandas == 1.4.2
- scipy == 1.8.1
- scikit-learn == 1.1.1
- tqdm == 4.64.0
- matplotlib == 3.4.2
- R == 4.0.3

## Tutorial

A Jupyter Notebook of the tutorial is accessible from :

https://github.com/JiruiZhang/STWaveGraph/blob/main/STWaveGraph-main/10X%20Visium.ipynb


---

## Revision update (2026-10-27)

The original README above is retained. For the revised implementation, use the recorded environment and instructions below; the earlier installation section describes the historical snapshot. Run commands after `cd STWaveGraph-main`. Unless stated otherwise, paths below are relative to that directory.

Revised model source and six historical dataset tutorials, assembled 2026-10-27.
The model and graph implementation files are copied byte-for-byte from the
validated frozen revision archive. The dense implementation includes the
corrected damped-wave response at critical and near-critical damping.

### Contents

- `STWaveGraph/`: revised dense reference implementation.
- `STWaveGraphSparse/`: E19 sparse implementation, retaining the validated graph semantics.
- `configs/`: 17 frozen tissue/sample configurations; these are distinct from constructor defaults.
- `reproduction/`: strict clustering, explicit all-observation mclust initialization, and clustering-record tests.
- `tests/numerics/`: original solver regression tests.
- `requirements.txt`: concise recorded dependency versions.
- Six root-level `.ipynb` files: historical tutorials with saved outputs.
- `SOURCE_MANIFEST.json`: provenance and SHA256 checksums for this upload directory.

Raw data, trained checkpoints, manuscript files, third-party baseline source
trees, and complete experiment logs are separate assets. This directory is
the model-code update, not the full 850-run benchmark reproduction archive.
Retain the existing repository license and attribution files when uploading.

### Environment

The recorded execution environment used Linux, Python 3.8.15, PyTorch
2.4.1+cu124, R 4.0.3, rpy2 3.4.1 and mclust 6.0.0. Install a suitable PyTorch
build, the packages in `requirements.txt`, and R/mclust separately. If needed,
set `R_HOME` to the installed R directory before importing rpy2. Verify the
installed R package version with `packageVersion("mclust")`.

The original environment was restored and checked offline. This concise list
does not claim a tested fresh online installation on every platform. The full
reproduction archive retains platform-specific environment snapshots.

### Minimal use with raw integer counts

Run from the repository root with that directory on Python's import path.
`adata.X` must contain unnormalized counts, with coordinates in
`adata.obsm["spatial"]`. The example configuration is specific to DLPFC section 151673;
provide the appropriate configuration and a known or prespecified cluster
count for other data. Annotation labels are not model features.

```python
import json
import scanpy as sc
from STWaveGraph import STWaveGraph

adata = sc.read_h5ad("data/DLPFC_151673.h5ad")
with open("configs/DLPFC_151673_wave_config.json") as f:
    config = json.load(f)
config["mclust_fallback_to_gmm"] = False
trainer = STWaveGraph(adata, **config)
trainer.fit()
result = trainer.infer(refine=True)
# result.obsm["emb"], result.obs["pred_domain_raw"], result.obs["pred_domain"]
```

The frozen core retains the historical API default allowing a GMM fallback;
the supplied configurations explicitly disable it. Always pass
`mclust_fallback_to_gmm=False` for reported mclust experiments. The direct API
example uses the core/native inference path; it is not the common-benchmark
readout described below.

For the sparse implementation, use `from STWaveGraphSparse import STWaveGraph`.
Sparse graph operations do not imply that every subsequent clustering step
is linear in the number of observations.

### Preprocessing and readout protocols

The raw-count example is not a universal preprocessing recipe. In the frozen
common benchmark, panels with fewer than 2000 genes retained nonconstant
genes; noninteger continuous inputs used nonconstant genes, log1p and scaling
without count normalization. These were explicit campaign adapters, not
changes to the immutable core. Preprocessing was fixed at seed 42; training
and inference used the specified run seed. Reproducing those experiments
requires their archived adapters and inputs, rather than blindly applying the
raw-count example to an already normalized matrix.

For strict external clustering of a saved row-normalized embedding:

```python
from reproduction.full_mclust_backend import cluster_embedding_fullinit
from reproduction.tracked_clustering import cluster_with_record

labels, probabilities, centers, record = cluster_with_record(
    embedding, n_clusters=known_C, method="mclust", seed=42,
    pca_dim=min(20, embedding.shape[1], len(embedding)-1),
    mclust_model="EEE", allow_fallback=False,
    backend=cluster_embedding_fullinit,
)
```

This explicitly uses all observations for hierarchical initialization and
raises on failure. Save the returned record alongside predictions; on failure
save `ClusterExecutionError.record`. Common external, common spatially refined,
native and internal predictions are distinct readouts. The frozen common
benchmark uses its own six-neighbor refinement, not the native `infer()`
refinement. The real-scale sparse experiment instead used a prespecified
2000-observation initialization subset followed by full-data EM; the supplied
all-observation backend does not implement that separate protocol.

Unknown cluster-count selection has not been validated. The revision does not
establish universal superiority or that every component improves every metric.

### Checks

```bash
python tests/numerics/test_wave_solver.py
python reproduction/test_tracked_clustering.py
```

The numerical suite checks matrix-exponential agreement, critical damping,
gradients, initial conditions and network forward/backward execution. The
clustering-record suite injects test backends and does not require R; it does
not replace a real R/mclust integration check. Frozen integration evidence is
retained in the full reproduction archive. This upload preparation does not
retrain models or replace the frozen experimental results.

### Historical tutorials and saved outputs

The following files were preserved exactly as supplied in the desktop folder:

| Notebook | Dataset | Original figure |
| --- | --- | --- |
| [DLPFC_151673.ipynb](STWaveGraph-main/DLPFC_151673.ipynb) | DLPFC section 151673 | Fig. 3 |
| [Breast.ipynb](STWaveGraph-main/Breast.ipynb) | Human breast cancer | Fig. 4 |
| [MouseBrain.ipynb](STWaveGraph-main/MouseBrain.ipynb) | Mouse brain | Fig. 4 |
| [osmFISH.ipynb](STWaveGraph-main/osmFISH.ipynb) | osmFISH | Fig. 5 |
| [MERFISH.ipynb](STWaveGraph-main/MERFISH.ipynb) | MERFISH | Fig. 5 |
| [STARmap.ipynb](STWaveGraph-main/STARmap.ipynb) | STARmap | Fig. 6 |

These notebooks retain their saved code, outputs and execution counts. They
use historical dataset-specific settings, which can differ from `configs/`
and the constructor defaults. The saved outputs have not been regenerated
with the revised core. Notebook output alone does not bind a historical
runtime to a particular model checkpoint or prove which fallback backend ran.

`STARmap.ipynb` explicitly allows a GMM fallback; other historical tutorials
may inherit the core's fallback default. A permitted fallback does not imply
that it actually occurred. Keep these outputs separate from the revised
strict mclust comparisons, which prohibit fallback. Historical notebook
paths point to the original Linux data layout and must be adapted locally.
Some notebooks retain draft cells; they are archival tutorials rather than
an assertion that Run All reproduces every manuscript panel.

The saved osmFISH text reports ARI 0.6294 and NMI 0.7076, whereas the original
figure reports 0.630/0.708. The ARI difference remains unresolved; neither the
saved output nor the original figure was altered to conceal it. Complete
baseline, marker-analysis and additional-panel matrices are not embedded in
these six notebooks.

### Reproduction material accompanying the revision

The full frozen experiment reproduction archive is supplied separately with
the revision materials. It contains the fixed-protocol benchmark, baseline
settings, seed-level predictions, ablation/sensitivity/resource workflows,
figure/table scripts and environment evidence. This source directory alone
does not contain those complete assets. No public archive URL or release tag
is asserted here before actual publication.

Keep the repository's existing LICENSE and attribution files when merging
these files. This directory does not introduce a new licensing grant.
