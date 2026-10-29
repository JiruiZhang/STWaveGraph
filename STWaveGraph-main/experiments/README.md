# Experiments

| File | Purpose |
| --- | --- |
| `configs.json` | Benchmark, ablation and sensitivity configurations |
| `run_benchmark.py` | Prepare inputs and run benchmark experiments |
| `run_ablation.py` | Run operator and component ablations |
| `run_sensitivity.py` | Run parameter sensitivity experiments |
| `experiment_utils.py` | Shared training, clustering and evaluation functions |

## Commands

Run from the repository root. Place input data under `STWaveGraph-main/Data/` and supply the baseline source directory.

```bash
cd experiments

# Check experiment plans without training
python run_benchmark.py --dry-run
python run_ablation.py --dry-run
python run_sensitivity.py --dry-run

# Prepare benchmark inputs
python run_benchmark.py --prepare --run ./outputs/benchmark

# Run benchmarks; replace the path with your baseline source directory
python run_benchmark.py --run ./outputs/benchmark --author-sources /path/to/baseline_sources

# Run ablations
python run_ablation.py --run ./outputs/ablation

# Run sensitivity experiments after the DLPFC ablations finish
python run_sensitivity.py --run ./outputs/sensitivity --ablation-run ./outputs/ablation
```
