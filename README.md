# Master-Thesis

The `Thesis` folder contains cosmology model experiments, notebooks, reports, and data for CPL, LCDM, and QCDM analyses.

### Top-level contents

- `.git/` — Git repository metadata
- `README.md` — project documentation
- Python scripts:
  - `CPL_benchmarking.py`
  - `LCDM_bench_cost.py`
  - `LCDM_benchmarking.py`
  - `CompressedSNLikelihood.py`
  - `train_CPL.py`
  - `train_LCDM.py`
  - `train_LCDM_cost.py`
- Jupyter notebooks:
  - `LCDM-datos.ipynb`
  - `inferencia.ipynb`
  - `training_lcdm.ipynb`
- Data files:
  - `cov_mu.txt`
  - `datos_mu.txt`
- PDF reports and figures for model evaluation and training results
- Model/data directories and checkpoints for CPL and LCDM experiments

### Subfolders

- `CPL/`
  - Includes CPL-specific benchmarking, training scripts, notebooks, parameter folders, and a `readme`
- `LCDM/`
  - Contains LCDM-specific benchmarking and training scripts, cost study folders, parameter dictionaries, and a `readme`
- `QCDM/`
  - Holds quintessence-related notebooks, data, scripts, a `readme`, and a `quintaesencia` folder

### Notes

- The repository is centered on cosmological model training, inference, and benchmarking.
- `CPL`, `LCDM`, and `QCDM` are the main thematic subprojects in this workspace.