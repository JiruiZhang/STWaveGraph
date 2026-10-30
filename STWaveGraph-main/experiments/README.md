# Experiments

| File | Purpose |
| --- | --- |
| `configs.json` | Benchmark, ablation and sensitivity configurations |
| `run_benchmark.py` | Prepare inputs and run benchmarks |
| `run_ablation.py` | Run operator and component ablations |
| `run_sensitivity.py` | Run parameter sensitivity experiments |
| `experiment_utils.py` | Shared training, clustering and evaluation functions |

## Commands

Start at the repository root after completing [installation](../../README.md).
Place input data under `STWaveGraph-main/Data/`. 

```bash
cd STWaveGraph-main/experiments

# Inspect plans without loading data or training
python run_benchmark.py --root .. --dry-run
python run_ablation.py --root .. --dry-run
python run_sensitivity.py --root .. --dry-run

# Prepare the 17 benchmark inputs
python run_benchmark.py --root .. --prepare --run ./outputs/benchmark

# Train STWaveGraph on the prepared inputs
python run_benchmark.py --root .. --run ./outputs/benchmark --methods STWaveGraph

# Run the three-tissue ablation experiment
python run_ablation.py --root .. --run ./outputs/ablation

# Run sensitivity experiments after the DLPFC ablations have completed
python run_sensitivity.py --root .. --run ./outputs/sensitivity --ablation-run ./outputs/ablation
```

To run all ten methods, use a separate output directory and provide the upstream
baseline source folders:

```bash
python run_benchmark.py --root .. --prepare --run ./outputs/benchmark_all
export STWAVEGRAPH_PYTHON="$(command -v python)"
export BASELINE_PYTHON=/absolute/path/to/baseline/environment/bin/python
python run_benchmark.py --root .. --run ./outputs/benchmark_all --author-sources /absolute/path/to/baseline_sources
```

The source folder contains `DeepGFT`, `GraphST`, `STAGATE`,
`SpaGCN/SpaGCN_package`, `SEDR`, `DeepST`, `GAAEST`,
`SpaCAE/SpaCAE` and `3dOT`.
 Add `--device cpu` for CPU runs.

## Input paths

All paths below are relative to `STWaveGraph-main/`.

| Dataset | Input |
| --- | --- |
| DLPFC | `Data/DLPFC/<section>/`, including `151673` for ablations |
| Breast | `Data/Human_Breast_Cancer/` |
| Mouse brain | `Data/Mouse_Brain_Anterior/` |
| osmFISH | `Data/osmFISHresult/osmFISH_cortex.h5` and `osmFISH_codeluppi2018spatial_cortex_data.h5ad` in the same folder |
| MERFISH | `Data/adata.h5ad` |
| STARmap | `Data/raw/adata.h5ad` |

